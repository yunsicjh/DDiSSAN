from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from transformers import BertModel
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.f_beta import F1Score
from torchmetrics.classification.confusion_matrix import ConfusionMatrix
from torch_geometric.data import Data, Batch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.models.components.bilstm_enhancer import BiLSTMEnhancer
from src.models.components.absa_token_aggregator import ABSATokenAggregator
from src.models.components.diffgraph_atten import DifferentialGraphTransformerLayer
from src.models.components.hybrid_graph_attention import HybridGraphTransformerLayer
from src.models.components.diffcross_atten import MultiHeadDifferentialAttention
from src.models.components.semantic_enhancer import (
    MultiLayerCrossAttention,
    GlobalFeatureEnhancer,
    MultiLayerDifferentialCrossAttention,
    PreFusionModule,
    FinalDifferentialAttention,
    ProgressiveDimensionReduction,
)
from src.models.components.cross_modal_attention import (
    GlobalSemanticEnhancer,
    CrossModalInteraction,
)


class FocalLoss(nn.Module):
    """
    Focal Loss实现：解决类别不平衡问题
    论文：Focal Loss for Dense Object Detection
    支持类别权重的增强版本
    """

    def __init__(self, alpha=1.0, gamma=2.0, class_weights=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 确保class_weights在正确的设备上
        weights = self.class_weights
        if weights is not None:
            weights = weights.to(inputs.device)

        # 计算加权交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, weight=weights, reduction="none")
        # 计算pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        # 计算focal loss = alpha * (1 - pt)^gamma * ce_loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class MixedLoss(nn.Module):
    """
    混合损失策略：结合Focal Loss、加权交叉熵和标签平滑
    专门用于处理类别不平衡问题
    支持针对特定类别的额外关注（如neutral类）
    """

    def __init__(
        self,
        focal_alpha=1.0,
        focal_gamma=2.0,
        class_weights=None,
        label_smoothing=0.15,
        loss_weights=(0.5, 0.3, 0.2),
        neutral_boost=0.0,  # neutral类额外权重系数
    ):
        super(MixedLoss, self).__init__()
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma, class_weights)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.loss_weights = loss_weights  # (focal, weighted_ce, label_smooth)
        self.neutral_boost = neutral_boost  # neutral类增强系数

    def _label_smooth_loss(self, logits, targets):
        """标签平滑损失"""
        confidence = 1.0 - self.label_smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, targets, reduction="none")
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.label_smoothing * smooth_loss
        return loss.mean()

    def forward(self, logits, targets):
        # 1. Focal Loss - 关注困难样本
        focal_loss = self.focal_loss(logits, targets)

        # 2. 加权交叉熵 - 平衡类别
        weights = self.class_weights
        if weights is not None:
            weights = weights.to(logits.device)
        weighted_ce_loss = F.cross_entropy(logits, targets, weight=weights)

        # 3. 标签平滑损失 - 正则化
        smooth_loss = self._label_smooth_loss(logits, targets)

        # 4. Neutral类额外关注（如果启用）
        neutral_penalty = 0.0
        if self.neutral_boost > 0:
            neutral_mask = targets == 1  # neutral类索引为1
            if neutral_mask.sum() > 0:
                neutral_logits = logits[neutral_mask]
                neutral_targets = targets[neutral_mask]

                # 使用更强的focal loss参数处理neutral样本
                neutral_focal = FocalLoss(
                    alpha=self.focal_loss.alpha * 1.5,  # 增强alpha
                    gamma=self.focal_loss.gamma + 1.0,  # 增强gamma
                    class_weights=self.class_weights,
                )
                neutral_penalty = neutral_focal(neutral_logits, neutral_targets)
                neutral_penalty = neutral_penalty * self.neutral_boost

        # 混合损失
        mixed_loss = (
            self.loss_weights[0] * focal_loss
            + self.loss_weights[1] * weighted_ce_loss
            + self.loss_weights[2] * smooth_loss
            + neutral_penalty  # 添加neutral类额外损失
        )

        return mixed_loss, {
            "focal_loss": focal_loss.item(),
            "weighted_ce_loss": weighted_ce_loss.item(),
            "smooth_loss": smooth_loss.item(),
            "mixed_loss": mixed_loss.item(),
        }


class SSANLitModule(LightningModule):
    """
    SSAN模型的LightningModule封装
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        dep_embed_dim: int = 30,
        num_heads: int = 8,
        num_classes: int = 3,
        # 参数初始化
        init_enable: bool = True,
        init_method: str = "uniform",  
        init_range: float = 0.1,  
        init_exclude_pretrained: bool = True,  
        # BiLSTM参数
        bilstm_hidden_dim: int = 128,  # 固定为128维
        bilstm_num_layers: int = 1,
        bilstm_dropout: float = 0.5,  # 增强正则化
        # 跨模态注意力参数
        cross_modal_layers: int = 1,
        cross_modal_dropout: float = 0.3,  # 增强正则化
        # 分类器参数
        classifier_dropout: float = 0.6,  # 增强正则化
        use_layer_norm: bool = True,
        # 新增正则化参数
        token_aggregator_dropout: float = 0.2,
        bert_dropout: float = 0.1,
        gradient_clip_val: float = 1.0,
        label_smoothing: float = 0.15,
        # 类别不平衡处理参数
        use_class_weights: bool = True,
        use_focal_loss: bool = True,
        use_mixed_loss: bool = True,  
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        mixed_loss_weights: tuple = (
            0.5,
            0.3,
            0.2,
        ),  # (focal, weighted_ce, label_smooth)
        neutral_boost: float = 0.0,  # neutral类额外关注系数 (0.0=不启用, 0.3-0.5=推荐值)
        dataset_name: str = "restaurants",  # 用于自动设置类别权重
        # 图注意力类型选择
        graph_attention_type: str = "hybrid",  # "differential" 或 "hybrid"
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # BERT编码器 - 增加dropout正则化
        self.bert = BertModel.from_pretrained(bert_model_name)
        if hasattr(self.hparams, "bert_dropout") and self.hparams.bert_dropout > 0:
            self.bert.config.hidden_dropout_prob = self.hparams.bert_dropout
            self.bert.config.attention_probs_dropout_prob = self.hparams.bert_dropout

        # 核心组件（在configure_model中初始化）
        self.token_aggregator = None
        self.bilstm_enhancer = None
        self.diffgraph_attention = None
        self.hybrid_graph_attention = None
        self.dep_embedding = None

        # 新的跨模态组件
        self.global_semantic_enhancer = None
        self.cross_modal_interaction = None
        self.pre_fusion_module = None
        self.classifier = None

        # 类别不平衡处理：设置类别权重
        self.class_weights = None
        if use_class_weights:
            
            dataset_weights = {
                "restaurants": torch.tensor(
                    [4.46, 5.65, 1.67]
                ),  
                "laptops": torch.tensor([2.68, 5.02, 2.34]),  
                "tweets": torch.tensor([3.96, 2.01, 2.01]),  
            }
            weights = dataset_weights.get(
                dataset_name.lower(), torch.tensor([1.0, 1.0, 1.0])
            )
            # 权重归一化
            self.class_weights = weights / weights.sum() * num_classes

        # 损失函数配置 - 支持混合损失策略
        if use_mixed_loss:
            self.criterion = MixedLoss(
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                class_weights=self.class_weights,
                label_smoothing=label_smoothing,
                loss_weights=mixed_loss_weights,
                neutral_boost=neutral_boost,  
            )
        elif use_focal_loss:
            self.criterion = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                class_weights=self.class_weights,
                reduction="mean",
            )
            
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                weight=self.class_weights, label_smoothing=label_smoothing
            )
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        self.test_confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

        # 存储每个类别的F1分数
        self.test_f1_per_class = F1Score(
            task="multiclass", num_classes=num_classes, average=None
        )

        # 存储每个类别的准确率
        self.test_acc_per_class = Accuracy(
            task="multiclass", num_classes=num_classes, average=None
        )

        # 定义类别标签映射
        self.class_names = ["negative", "neutral", "positive"]  # 根据实际情况调整

        # 存储预测和目标用于详细分析
        self.test_predictions = []
        self.test_targets = []
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()
        self.test_f1_best = MaxMetric()
        # 参数初始化状态标记（避免重复初始化）
        self._params_initialized = False

    def on_after_backward(self) -> None:
        """在反向传播后进行梯度裁剪"""
        if (
            hasattr(self.hparams, "gradient_clip_val")
            and self.hparams.gradient_clip_val > 0
        ):
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.hparams.gradient_clip_val
            )

    def configure_model(self) -> None:
        """
        根据新prompt.txt配置完整的多模态模型架构
        """
        if self.token_aggregator is not None:
            return  # 已配置

        # 获取依赖词汇表大小
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            dep_vocab_size = self.trainer.datamodule.dep_vocab_size
            print(f"Configuring model with dep_vocab_size: {dep_vocab_size}")
        else:
            dep_vocab_size = 100
            print(f"Warning: Using default dep_vocab_size: {dep_vocab_size}")

       
        self.token_aggregator = ABSATokenAggregator(
            hidden_dim=self.bert.config.hidden_size,
            sentence_aggregation="attention",
            aspect_aggregation="mean",
            use_position_encoding=True,
            dropout=self.hparams.token_aggregator_dropout,
        )

       
        self.bilstm_enhancer = BiLSTMEnhancer(
            input_dim=self.bert.config.hidden_size,
            hidden_dim=self.hparams.bilstm_hidden_dim,
            num_layers=self.hparams.bilstm_num_layers,
            dropout=self.hparams.bilstm_dropout,
            bidirectional=True,
        )

        
        self.dep_embedding = nn.Embedding(dep_vocab_size, self.hparams.dep_embed_dim)

       
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            pos_vocab_size = self.trainer.datamodule.pos_vocab_size
            print(f"Configuring POS embedding with vocab_size: {pos_vocab_size}")
        else:
            pos_vocab_size = 50  
            print(f"Warning: Using default pos_vocab_size: {pos_vocab_size}")

        self.pos_embedding = nn.Embedding(pos_vocab_size, 30)  
        self.position_embedding = nn.Embedding(
            self.hparams.get("max_seq_len", 128), 30
        )  

       
        self.structure_fusion = nn.Linear(768 + 30 + 30, 768)

       
        if self.hparams.graph_attention_type == "hybrid":
            # 使用混合图注意力
            self.hybrid_graph_attention = HybridGraphTransformerLayer(
                in_channels=self.hparams.bilstm_hidden_dim,  
                out_channels=self.hparams.bilstm_hidden_dim, 
                edge_dim=self.hparams.dep_embed_dim,
                heads=self.hparams.num_heads,
                lambda_init=0.8,
                dropout=self.hparams.cross_modal_dropout,  
                concat=False,  
            )
            
        else:
            # 使用传统差分图注意力
            self.diffgraph_attention = DifferentialGraphTransformerLayer(
                in_channels=self.hparams.bilstm_hidden_dim,  
                out_channels=self.hparams.bilstm_hidden_dim,  
                edge_dim=self.hparams.dep_embed_dim,
                heads=self.hparams.num_heads,
                lambda_init=0.8,
                dropout=self.hparams.cross_modal_dropout,  
                concat=False,  
            )
           

       
        self.semantic_cross_attention = MultiLayerCrossAttention(
            d_model=self.hparams.bilstm_hidden_dim,  
            num_heads=self.hparams.num_heads,
            num_layers=3, 
            dropout=self.hparams.cross_modal_dropout,
        )

       
        self.global_feature_enhancer = GlobalFeatureEnhancer(
            d_model=self.hparams.bilstm_hidden_dim, 
            num_heads=self.hparams.num_heads,
            num_layers=3,  
            dropout=self.hparams.cross_modal_dropout,
        )

       
        self.diff_cross_attention_sem = MultiLayerDifferentialCrossAttention(
            d_model=self.hparams.bilstm_hidden_dim,  
            num_heads=self.hparams.num_heads,
            num_layers=3, 
            dropout=self.hparams.cross_modal_dropout,
            lambda_init=0.8,
        )

        self.diff_cross_attention_syn = MultiLayerDifferentialCrossAttention(
            d_model=self.hparams.bilstm_hidden_dim,  
            num_heads=self.hparams.num_heads,
            num_layers=3,  
            dropout=self.hparams.cross_modal_dropout,
            lambda_init=0.8,
        )

        
        self.pre_fusion_module = PreFusionModule(
            d_model=self.hparams.bilstm_hidden_dim  
        )

      
        fusion_dim = self.hparams.bilstm_hidden_dim * 2  
        self.final_diff_attention = FinalDifferentialAttention(
            d_model=fusion_dim,
            num_heads=self.hparams.num_heads,
            num_layers=3, 
            dropout=self.hparams.cross_modal_dropout,
            lambda_init=0.8,
        )

        
        self.progressive_reduction = ProgressiveDimensionReduction(
            input_dim=fusion_dim, 
            hidden_dims=[128, 64], 
            output_dim=32, 
            dropout=self.hparams.classifier_dropout * 0.5,
        )

       
        classifier_layers = [
            nn.Dropout(self.hparams.classifier_dropout * 0.5),
            nn.Linear(32, 16),  
        ]

        if self.hparams.use_layer_norm:
            classifier_layers.append(nn.LayerNorm(16))

        classifier_layers.extend(
            [
                nn.ReLU(),
                nn.Dropout(self.hparams.classifier_dropout * 0.3),
                nn.Linear(16, self.hparams.num_classes),  
            ]
        )

        self.classifier = nn.Sequential(*classifier_layers)

        graph_attention_name = (
            "混合图注意力"
            if self.hparams.graph_attention_type == "hybrid"
            else "差分图注意力"
        )

        

        if getattr(self.hparams, "init_enable", True) and not self._params_initialized:
            init_method = self.hparams.get("init_method", "uniform")
            exclude_pretrained = getattr(self.hparams, "init_exclude_pretrained", True)

            if init_method == "uniform":
                self._initialize_parameters_uniform(
                    init_range=self.hparams.get("init_range", 0.1),
                    exclude_pretrained=exclude_pretrained,
                )
               
            elif init_method == "xavier":
                self._initialize_parameters_xavier(
                    exclude_pretrained=exclude_pretrained
                )
               
            else:
                pass
            self._params_initialized = True

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播流程实现SSAN模型
        """
        
        if self.token_aggregator is None:
            self.configure_model()

        batch_size = batch["bert_input_ids"].shape[0]

       
        bert_output = self.bert(
            input_ids=batch["bert_input_ids"],
            attention_mask=batch["bert_attention_mask"],
        ).last_hidden_state  # [batch_size, seq_len, 768]

       
        sentence_word_features, aspect_word_features, aspect_word_mask = (
            self.token_aggregator(
                bert_output=bert_output,
                sentence_mask=batch["sentence_mask"],
                aspect_in_sentence_mask=batch.get("aspect_in_sentence_mask"),
                token_to_subtoken_maps=batch["token_to_subtoken_maps"],
            )
        ) 

       
        batch_size, seq_len = sentence_word_features.shape[:2]

        
        pos_ids = batch["pos_ids"][:, :seq_len]     
        pos_features = self.pos_embedding(pos_ids)  # [batch_size, seq_len, 30]

     
        position_ids = (
            torch.arange(seq_len, device=sentence_word_features.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        position_features = self.position_embedding(
            position_ids
        )  # [batch_size, seq_len, 30]

        
        enhanced_sentence_features = torch.cat(
            [
                sentence_word_features,  # [batch_size, seq_len, 768]
                pos_features,  # [batch_size, seq_len, 30]
                position_features,  # [batch_size, seq_len, 30]
            ],
            dim=-1,
        )  # [batch_size, seq_len, 828]

       
        sentence_word_features = self.structure_fusion(
            enhanced_sentence_features
        )  # [batch_size, seq_len, 768]

        
        sentence_word_mask = sentence_word_features.abs().sum(dim=-1) > 1e-6
        c_s_bi = self.bilstm_enhancer.enhance_word_features(
            sentence_word_features, sentence_word_mask
        )  # [batch_size, seq_len, 128]

       
        batch_size, seq_len = c_s_bi.shape[:2]

        
        graph_outputs = []

        for i in range(batch_size):
            # 当前样本的特征和图结构
            node_feats = c_s_bi[i]  # [seq_len, 128]
            adj_matrix = batch["adj_matrix"][i]  # [seq_len, seq_len]
            rel_matrix = batch["dep_rel_matrix"][i]  # [seq_len, seq_len]

            # 构建边索引
            edge_index = adj_matrix.nonzero().t().contiguous()

            # 构建边属性
            if edge_index.size(1) > 0:
                edge_types = rel_matrix[edge_index[0], edge_index[1]]
                edge_attr = self.dep_embedding(edge_types)
            else:
                edge_attr = None

            # 应用图注意力
            if edge_index.size(1) > 0:
                if self.hybrid_graph_attention is not None:
                    # 使用混合图注意力
                    graph_output = self.hybrid_graph_attention(
                        node_feats, edge_index, edge_attr
                    )
                else:
                    # 使用差分图注意力
                    graph_output = self.diffgraph_attention(
                        node_feats, edge_index, edge_attr
                    )
            else:
                # 无边的情况，直接使用原始特征
                graph_output = node_feats

            graph_outputs.append(graph_output)

       
        H_syn = torch.stack(graph_outputs, dim=0)  # [batch_size, seq_len, 128]

        
        valid_mask = sentence_word_mask.unsqueeze(
            -1
        ).float()  # [batch_size, seq_len, 1]
        G = (c_s_bi * valid_mask).sum(dim=1) / (
            valid_mask.sum(dim=1) + 1e-8
        )  # [batch_size, 128]

       
        H_sem = c_s_bi  # [batch_size, seq_len, 128]
        H_sem_1 = self.semantic_cross_attention(H_sem, G)  # [batch_size, seq_len, 128]

       
        G_out = self.global_feature_enhancer(G, H_sem)  # [batch_size, 128]

        
        H_sem_out = self.diff_cross_attention_sem(
            H_sem_1, H_syn, H_syn
        )  # [batch_size, seq_len, 128]

       
        H_syn_c = self.diff_cross_attention_syn(
            H_syn, H_sem, H_sem
        )  # [batch_size, seq_len, 128]

       
        H_syn_out = self.pre_fusion_module(H_syn_c, G_out)  # [batch_size, seq_len, 128]

        
        H_out = torch.cat([H_sem_out, H_syn_out], dim=-1)  # [batch_size, seq_len, 256]

        
        H_out_f, H_out_residual = self.final_diff_attention(
            H_out
        )  # 都是[batch_size, seq_len, 256]

       
        H_reduced = self.progressive_reduction(
            H_out_residual
        )  # [batch_size, seq_len, 32]


        aspect_in_sentence_mask = batch.get("aspect_in_sentence_mask")
        if aspect_in_sentence_mask is not None:
            # 使用方面词mask提取特征
            aspect_mask_expanded = aspect_in_sentence_mask.unsqueeze(
                -1
            ).float()  # [batch_size, seq_len, 1]

            # 提取方面词特征并池化
            aspect_features = (
                H_reduced * aspect_mask_expanded
            )  # [batch_size, seq_len, 32]
            aspect_representation = aspect_features.sum(dim=1) / (
                aspect_mask_expanded.sum(dim=1) + 1e-8
            )  # [batch_size, 32]

        else:
            # 如果没有方面词mask，使用全局平均池化
            valid_mask_reduced = valid_mask[
                :, : H_reduced.size(1), :
            ]  # 确保维度匹配 [batch_size, seq_len, 1]
            aspect_representation = (H_reduced * valid_mask_reduced).sum(dim=1) / (
                valid_mask_reduced.sum(dim=1) + 1e-8
            )  # [batch_size, 32]

        
        logits = self.classifier(aspect_representation)

        return {
            "logits": logits,
            "representations": {
                "H_sem_out": H_sem_out,
                "H_syn_out": H_syn_out,
                "H_out": H_out,
                "H_out_f": H_out_f,
                "aspect_features": aspect_representation,
            },
        }

    def _initialize_parameters_uniform(
        self, init_range: float = 0.1, exclude_pretrained: bool = True
    ) -> None:
        """使用均匀分布初始化模型参数。

        注意：默认跳过预训练BERT的参数，仅对自定义层（Linear/Embedding/LSTM/GRU等）执行初始化。

        Args:
            init_range: 均匀分布范围 [-init_range, init_range]
            exclude_pretrained: 是否排除预训练权重（如BERT）
        """
        low, high = -init_range, init_range

        def should_skip(name: str, module: nn.Module) -> bool:
            if not exclude_pretrained:
                return False
            # 跳过BERT及其子模块
            if name.startswith("bert") or isinstance(module, (BertModel,)):
                return True
            return False

        # 针对模块类型的权重初始化
        for module_name, module in self.named_modules():
            if should_skip(module_name, module):
                continue

            # Linear 层
            if isinstance(module, nn.Linear):
                if module.weight is not None:
                    nn.init.uniform_(module.weight, low, high)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, low, high)

            # Embedding 层
            elif isinstance(module, nn.Embedding):
                if module.weight is not None:
                    nn.init.uniform_(module.weight, low, high)

            # RNN 类
            elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                for name, param in module.named_parameters(recurse=False):
                    if param is not None:
                        nn.init.uniform_(param, low, high)

            # 

    def _initialize_parameters_xavier(self, exclude_pretrained: bool = True) -> None:
        """使用 Xavier Uniform 初始化模型参数。

        Xavier 初始化有助于保持前向传播和反向传播中的信号稳定，适用于大多数激活函数。   

        Args:
            exclude_pretrained: 是否排除预训练权重（如BERT）
        """

        def should_skip(name: str, module: nn.Module) -> bool:
            if not exclude_pretrained:
                return False
            if name.startswith("bert") or isinstance(module, (BertModel,)):
                return True
            return False

        # 针对模块类型的权重初始化
        for module_name, module in self.named_modules():
            if should_skip(module_name, module):
                continue

            # Linear 层 - Xavier Uniform
            if isinstance(module, nn.Linear):
                if module.weight is not None:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

            # Embedding 层 - Normal Xavier
            elif isinstance(module, nn.Embedding):
                if module.weight is not None:
                    nn.init.xavier_normal_(module.weight)

            # LSTM 层 - 
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters(recurse=False):
                    if "weight_ih" in name: 
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:  
                        
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:  # 偏置初始化为0
                        nn.init.constant_(param, 0.0)

            # GRU 和 RNN
            elif isinstance(module, (nn.GRU, nn.RNN)):
                for name, param in module.named_parameters(recurse=False):
                    if "weight" in name:
                        # 所有权重都用 Xavier Uniform（MPS 兼容）
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0.0)

    def _initialize_parameters_kaiming(self, exclude_pretrained: bool = True) -> None:
        """使用 Kaiming (He) Normal 初始化，针对 ReLU 激活函数优化。

        Args:
            exclude_pretrained: 是否排除预训练权重（如BERT）
        """

        def should_skip(name: str, module: nn.Module) -> bool:
            if not exclude_pretrained:
                return False
            if name.startswith("bert") or isinstance(module, (BertModel,)):
                return True
            return False

        for module_name, module in self.named_modules():
            if should_skip(module_name, module):
                continue

            # Linear 层
            if isinstance(module, nn.Linear):
                if module.weight is not None:
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_in", nonlinearity="relu"
                    )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

            # Embedding 层
            elif isinstance(module, nn.Embedding):
                if module.weight is not None:
                    nn.init.normal_(
                        module.weight, mean=0, std=1.0 / (module.weight.size(1) ** 0.5)
                    )

            # LSTM 层 
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters(recurse=False):
                    if "weight_ih" in name:
                        nn.init.kaiming_normal_(
                            param, mode="fan_in", nonlinearity="relu"
                        )
                    elif "weight_hh" in name:
                        # 改用 Xavier Uniform 以确保 MPS 兼容性
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0.0)

            # GRU 和 RNN
            elif isinstance(module, (nn.GRU, nn.RNN)):
                for name, param in module.named_parameters(recurse=False):
                    if "weight_ih" in name:
                        nn.init.kaiming_normal_(
                            param, mode="fan_in", nonlinearity="relu"
                        )
                    elif "weight_hh" in name:
                        # 改用 Xavier Uniform 以确保 MPS 兼容性
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0.0)

    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """模型步骤：支持混合损失策略、Focal Loss和类别加权"""
        output = self.forward(batch)
        logits = output["logits"]
        targets = batch["polarity"]


        if isinstance(self.criterion, MixedLoss):

            loss_result = self.criterion(logits, targets)
            if isinstance(loss_result, tuple):
                loss, loss_details = loss_result

                self.loss_details = loss_details
            else:
                loss = loss_result
        elif isinstance(self.criterion, FocalLoss):

            loss = self.criterion(logits, targets)
        elif self.class_weights is not None and not isinstance(
            self.criterion, (FocalLoss, MixedLoss)
        ):

            criterion = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device),
                label_smoothing=self.hparams.label_smoothing,
            )
            loss = criterion(logits, targets)
        else:
       
            loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        return loss, preds, targets

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
     
        loss, preds, targets = self.model_step(batch)
        batch_size = targets.size(0)

        # 更新指标
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)

        # 记录指标
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/f1",
            self.train_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """简化的验证步骤"""
        loss, preds, targets = self.model_step(batch)
        batch_size = targets.size(0)

        # 更新指标
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)

        # 记录指标
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "val/acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "val/f1",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """测试步骤"""
        # 1. 前向传播
        output = self.forward(batch)
        logits = output["logits"]
        targets = batch["polarity"]

        # 2. 计算损失
        if isinstance(self.criterion, MixedLoss):
            loss_result = self.criterion(logits, targets)
            if isinstance(loss_result, tuple):
                loss, _ = loss_result
            else:
                loss = loss_result
        else:
            loss = self.criterion(logits, targets)

        # 3. 获取预测结果
        preds = torch.argmax(logits, dim=-1)
        batch_size = targets.size(0)

        # 4. 更新测试指标
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.test_confusion_matrix(preds, targets)
        self.test_f1_per_class(preds, targets)
        self.test_acc_per_class(preds, targets)

        # 5. 存储预测结果用于详细分析
        self.test_predictions.extend(preds.cpu().numpy().tolist())
        self.test_targets.extend(targets.cpu().numpy().tolist())

        # 6. 实时显示分析（对于每个batch）
        self._show_batch_analysis(preds.cpu().numpy(), targets.cpu().numpy(), batch_idx)

        # 7. 记录指标
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "test/acc",
            self.test_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "test/f1",
            self.test_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def on_validation_epoch_end(self) -> None:
        """验证周期结束时的处理"""
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()

        self.val_acc_best(acc)
        self.val_f1_best(f1)

        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )
        self.log(
            "val/f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True
        )

    def _save_confusion_matrix(self, confusion_matrix: np.ndarray):
        """
        保存混淆矩阵数据和可视化图表,并记录到wandb

        Args:
            confusion_matrix: 混淆矩阵数组 shape: (num_classes, num_classes)
        """
        # 获取数据集名称
        dataset_name = self.hparams.get("dataset_name", "unknown").lower()

        # 创建保存目录
        save_dir = Path("logs/confusion_matrices") / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # 生成时间戳
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 保存原始混淆矩阵数据 (numpy格式)
        np_path = save_dir / f"confusion_matrix_{dataset_name}_{timestamp}.npy"
        np.save(np_path, confusion_matrix)
        print(f"\n💾 混淆矩阵已保存到: {np_path}")

        # 2. 保存为JSON格式(便于查看)
        cm_dict = {
            "dataset": dataset_name,
            "confusion_matrix": confusion_matrix.tolist(),
            "class_names": self.class_names,
            "total_samples": int(confusion_matrix.sum()),
            "correct_samples": int(np.diag(confusion_matrix).sum()),
            "accuracy": float(np.diag(confusion_matrix).sum() / confusion_matrix.sum()),
        }

        # 计算每个类别的指标
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            true_total = confusion_matrix[i, :].sum()
            pred_total = confusion_matrix[:, i].sum()
            true_positive = confusion_matrix[i, i]

            precision = true_positive / pred_total if pred_total > 0 else 0
            recall = true_positive / true_total if true_total > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            per_class_metrics[class_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "support": int(true_total),
            }

        cm_dict["per_class_metrics"] = per_class_metrics

        json_path = save_dir / f"confusion_matrix_{dataset_name}_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(cm_dict, f, indent=2, ensure_ascii=False)
        print(f"💾 混淆矩阵JSON已保存到: {json_path}")

        # 3. 生成可视化图表
        img_paths = self._plot_confusion_matrix(
            confusion_matrix, save_dir, timestamp, dataset_name
        )

        # 4. 记录到 WandB
        self._log_to_wandb(confusion_matrix, cm_dict, img_paths, dataset_name)

    def _plot_confusion_matrix(
        self, cm: np.ndarray, save_dir: Path, timestamp: str, dataset_name: str
    ):
        """
        绘制混淆矩阵热力图

        Args:
            cm: 混淆矩阵
            save_dir: 保存目录
            timestamp: 时间戳
            dataset_name: 数据集名称

        Returns:
            dict: 包含所有图片路径的字典
        """
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. 绘制原始计数混淆矩阵
        sns.heatmap(
            cm,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[0],
            cbar_kws={"label": "Count"},
        )
        axes[0].set_title(
            f"Confusion Matrix - {dataset_name.upper()} (Count)",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].set_ylabel("True Label", fontsize=12)
        axes[0].set_xlabel("Predicted Label", fontsize=12)

        # 2. 绘制归一化混淆矩阵(按行归一化,显示每个真实类别的预测分布)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # 处理除以0的情况

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2%",
            cmap="YlOrRd",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[1],
            cbar_kws={"label": "Percentage"},
        )
        axes[1].set_title(
            f"Confusion Matrix - {dataset_name.upper()} (Normalized)",
            fontsize=14,
            fontweight="bold",
        )
        axes[1].set_ylabel("True Label", fontsize=12)
        axes[1].set_xlabel("Predicted Label", fontsize=12)

        plt.tight_layout()

        # 保存图表
        cm_img_path = save_dir / f"confusion_matrix_{dataset_name}_{timestamp}.png"
        plt.savefig(cm_img_path, dpi=300, bbox_inches="tight")
        print(f"📊 混淆矩阵可视化已保存到: {cm_img_path}")
        plt.close()

        # 3. 绘制每个类别的性能条形图
        metrics_img_path = self._plot_per_class_metrics(
            cm, save_dir, timestamp, dataset_name
        )

        return {"confusion_matrix": cm_img_path, "per_class_metrics": metrics_img_path}

    def _plot_per_class_metrics(
        self, cm: np.ndarray, save_dir: Path, timestamp: str, dataset_name: str
    ):
        """
        绘制每个类别的Precision, Recall, F1-Score条形图

        Args:
            cm: 混淆矩阵
            save_dir: 保存目录
            timestamp: 时间戳
            dataset_name: 数据集名称

        Returns:
            Path: 保存的图片路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics_data = []
        for i, class_name in enumerate(self.class_names):
            true_total = cm[i, :].sum()
            pred_total = cm[:, i].sum()
            true_positive = cm[i, i]

            precision = true_positive / pred_total if pred_total > 0 else 0
            recall = true_positive / true_total if true_total > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            metrics_data.append(
                {
                    "class": class_name,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                }
            )

        # 准备数据
        classes = [d["class"] for d in metrics_data]
        precisions = [d["precision"] for d in metrics_data]
        recalls = [d["recall"] for d in metrics_data]
        f1_scores = [d["f1_score"] for d in metrics_data]

        # 设置柱状图位置
        x = np.arange(len(classes))
        width = 0.25

        # 绘制柱状图
        ax.bar(x - width, precisions, width, label="Precision", color="#3498db")
        ax.bar(x, recalls, width, label="Recall", color="#2ecc71")
        ax.bar(x + width, f1_scores, width, label="F1-Score", color="#e74c3c")

        # 设置图表
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(
            f"Per-Class Metrics - {dataset_name.upper()}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=0)
        ax.legend(loc="lower right")
        ax.set_ylim([0, 1.1])
        ax.grid(axis="y", alpha=0.3)

        # 在柱子上添加数值标签
        for i, (p, r, f) in enumerate(zip(precisions, recalls, f1_scores)):
            ax.text(
                i - width, p + 0.02, f"{p:.2f}", ha="center", va="bottom", fontsize=8
            )
            ax.text(i, r + 0.02, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(
                i + width, f + 0.02, f"{f:.2f}", ha="center", va="bottom", fontsize=8
            )

        plt.tight_layout()

        # 保存图表
        img_path = save_dir / f"per_class_metrics_{dataset_name}_{timestamp}.png"
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        plt.close()

        return img_path

    def _log_to_wandb(
        self, cm: np.ndarray, cm_dict: dict, img_paths: dict, dataset_name: str
    ):
        """
        将混淆矩阵结果记录到WandB

        Args:
            cm: 混淆矩阵数组
            cm_dict: 混淆矩阵字典(包含各类指标)
            img_paths: 图片路径字典
            dataset_name: 数据集名称
        """
        # 检查是否有可用的logger
        if not hasattr(self, "logger") or self.logger is None:
            print("⚠️ 未检测到logger,跳过WandB记录")
            return

        # 检查是否是WandB logger
        try:
            import wandb

            # 如果使用的是WandbLogger
            if hasattr(self.logger, "experiment"):
                # 1. 记录混淆矩阵热力图
                wandb_cm = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=self.test_targets,
                    preds=self.test_predictions,
                    class_names=self.class_names,
                    title=f"Confusion Matrix - {dataset_name.upper()}",
                )
                self.logger.experiment.log(
                    {f"test/{dataset_name}/confusion_matrix_wandb": wandb_cm}
                )

                # 2. 记录本地生成的图片
                if (
                    img_paths.get("confusion_matrix")
                    and img_paths["confusion_matrix"].exists()
                ):
                    self.logger.experiment.log(
                        {
                            f"test/{dataset_name}/confusion_matrix_image": wandb.Image(
                                str(img_paths["confusion_matrix"]),
                                caption=f"Confusion Matrix - {dataset_name.upper()}",
                            )
                        }
                    )

                if (
                    img_paths.get("per_class_metrics")
                    and img_paths["per_class_metrics"].exists()
                ):
                    self.logger.experiment.log(
                        {
                            f"test/{dataset_name}/per_class_metrics": wandb.Image(
                                str(img_paths["per_class_metrics"]),
                                caption=f"Per-Class Metrics - {dataset_name.upper()}",
                            )
                        }
                    )

                # 3. 记录各类别详细指标
                per_class_metrics = cm_dict.get("per_class_metrics", {})
                for class_name, metrics in per_class_metrics.items():
                    self.logger.experiment.log(
                        {
                            f"test/{dataset_name}/{class_name}/precision": metrics[
                                "precision"
                            ],
                            f"test/{dataset_name}/{class_name}/recall": metrics[
                                "recall"
                            ],
                            f"test/{dataset_name}/{class_name}/f1_score": metrics[
                                "f1_score"
                            ],
                            f"test/{dataset_name}/{class_name}/support": metrics[
                                "support"
                            ],
                        }
                    )

                # 4. 记录整体指标
                self.logger.experiment.log(
                    {
                        f"test/{dataset_name}/total_samples": cm_dict["total_samples"],
                        f"test/{dataset_name}/correct_samples": cm_dict[
                            "correct_samples"
                        ],
                        f"test/{dataset_name}/accuracy": cm_dict["accuracy"],
                    }
                )

                # 5. 创建汇总表格
                table_data = []
                for class_name in self.class_names:
                    metrics = per_class_metrics.get(class_name, {})
                    table_data.append(
                        [
                            class_name,
                            metrics.get("precision", 0),
                            metrics.get("recall", 0),
                            metrics.get("f1_score", 0),
                            metrics.get("support", 0),
                        ]
                    )

                metrics_table = wandb.Table(
                    columns=["Class", "Precision", "Recall", "F1-Score", "Support"],
                    data=table_data,
                )
                self.logger.experiment.log(
                    {f"test/{dataset_name}/metrics_table": metrics_table}
                )

                print(f"✅ 混淆矩阵已记录到 WandB (数据集: {dataset_name.upper()})")

        except ImportError:
            print("⚠️ wandb未安装,跳过WandB记录")
        except Exception as e:
            print(f"⚠️ WandB记录失败: {e}")

    def _show_batch_analysis(self, preds, targets, batch_idx):
        """实时显示批次分析 - 简化版"""
        if batch_idx % 50 == 0:  # 每50个batch显示一次
            correct = (preds == targets).sum()
            total = len(preds)
            accuracy = correct / total if total > 0 else 0
            print(f"Batch {batch_idx}: {correct}/{total} correct ({accuracy:.3f})")

    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器"""
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
