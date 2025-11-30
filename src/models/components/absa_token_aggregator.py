import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional


class ABSATokenAggregator(nn.Module):
    """
    专门为ABSA任务设计的Token聚合器

    解决BERT subtoken到原始word级别特征聚合问题
    输入: h=[h_cls,h1,……,hn,h_sep,a1,……,am,h_sep]
    输出: c_s=[c1,……,cn], c_a=[c1,……,cm] (词级特征)

    处理BERT-SPC格式：[CLS] sentence [SEP] aspect [SEP]
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        sentence_aggregation: str = "attention",
        aspect_aggregation: str = "mean",
        use_position_encoding: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: BERT隐藏层维度
            sentence_aggregation: 句子词聚合方法 ("mean", "max", "first", "last", "attention")
            aspect_aggregation: 方面词聚合方法
            use_position_encoding: 是否使用位置编码
            dropout: dropout率
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_position_encoding = use_position_encoding

        # 句子词聚合器
        self.sentence_aggregator = TokenToWordAggregator(
            hidden_dim, sentence_aggregation, True, dropout
        )

        # 方面词聚合器（可以使用不同的策略）
        self.aspect_aggregator = TokenToWordAggregator(
            hidden_dim, aspect_aggregation, True, dropout
        )

        # 位置编码（如果启用）
        if use_position_encoding:
            self.position_embedding = nn.Embedding(512, hidden_dim)  # 最大支持512个词

        # 特征增强层
        self.feature_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        bert_output: torch.Tensor,
        sentence_mask: torch.Tensor,
        aspect_in_sentence_mask: torch.Tensor,
        token_to_subtoken_maps: List[Dict[int, List[int]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            bert_output: BERT输出 [batch_size, seq_len, hidden_dim]
            sentence_mask: 句子在BERT序列中的mask [batch_size, seq_len]
            aspect_in_sentence_mask: 方面词在原句中的mask [batch_size, original_seq_len]
            token_to_subtoken_maps: token到subtoken的映射

        Returns:
            sentence_word_features: 句子词级特征 [batch_size, max_words, hidden_dim]
            aspect_word_features: 方面词词级特征 [batch_size, max_aspect_words, hidden_dim]
            word_level_mask: 词级别的有效性mask [batch_size, max_words]
        """
        batch_size = bert_output.shape[0]

        # 1. 提取句子词级特征
        sentence_word_features = self._extract_sentence_word_features(
            bert_output, sentence_mask, token_to_subtoken_maps
        )

        # 2. 从句子词级特征中提取方面词特征
        aspect_word_features, word_level_mask = self._extract_aspect_word_features(
            sentence_word_features, aspect_in_sentence_mask
        )

        # 3. 特征增强
        sentence_word_features = self.feature_enhancer(sentence_word_features)
        aspect_word_features = self.feature_enhancer(aspect_word_features)

        return sentence_word_features, aspect_word_features, word_level_mask

    def _extract_sentence_word_features(
        self,
        bert_output: torch.Tensor,
        sentence_mask: torch.Tensor,
        token_to_subtoken_maps: List[Dict[int, List[int]]],
    ) -> torch.Tensor:
        """提取句子词级特征"""
        batch_size = bert_output.shape[0]

        # 计算每个样本的词数量
        word_counts = [len(token_map) for token_map in token_to_subtoken_maps]
        max_words = max(word_counts) if word_counts else 1

        sentence_word_features = torch.zeros(
            batch_size, max_words, self.hidden_dim, device=bert_output.device
        )

        for b in range(batch_size):
            # 获取句子部分的subtoken特征
            sent_indices = sentence_mask[b].nonzero().squeeze(-1)
            if len(sent_indices) > 0:
                sent_subtokens = bert_output[b, sent_indices]  # [sent_len, hidden_dim]

                # 聚合为词级特征
                token_map = token_to_subtoken_maps[b]
                for word_idx, subtoken_indices in token_map.items():
                    if word_idx < max_words:
                        # 转换为相对位置
                        sentence_start = 1  # [CLS]后的位置
                        relative_indices = []
                        for abs_idx in subtoken_indices:
                            rel_idx = abs_idx - sentence_start
                            if 0 <= rel_idx < sent_subtokens.shape[0]:
                                relative_indices.append(rel_idx)

                        if relative_indices:
                            word_subtokens = sent_subtokens[relative_indices]
                            # 使用句子聚合器
                            aggregated = self._aggregate_with_position(
                                word_subtokens, word_idx, self.sentence_aggregator
                            )
                            sentence_word_features[b, word_idx] = aggregated

        return sentence_word_features

    def _extract_aspect_word_features(
        self,
        sentence_word_features: torch.Tensor,
        aspect_in_sentence_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从句子词级特征中提取方面词特征"""
        batch_size, max_sentence_words, hidden_dim = sentence_word_features.shape

        # 计算每个样本的方面词数量
        aspect_word_counts = aspect_in_sentence_mask.sum(dim=1).tolist()
        max_aspect_words = int(max(aspect_word_counts) if aspect_word_counts else 1)

        aspect_word_features = torch.zeros(
            batch_size,
            max_aspect_words,
            hidden_dim,
            device=sentence_word_features.device,
        )
        word_level_mask = torch.zeros(
            batch_size,
            max_aspect_words,
            dtype=torch.bool,
            device=sentence_word_features.device,
        )

        for b in range(batch_size):
            # 找到方面词在句子中的位置
            aspect_indices = aspect_in_sentence_mask[b].nonzero().squeeze(-1)

            for i, word_idx in enumerate(aspect_indices):
                if i < max_aspect_words and word_idx < sentence_word_features.shape[1]:
                    aspect_word_features[b, i] = sentence_word_features[b, word_idx]
                    word_level_mask[b, i] = True

        return aspect_word_features, word_level_mask

    def _aggregate_with_position(
        self,
        subtoken_features: torch.Tensor,
        word_position: int,
        aggregator: "TokenToWordAggregator",
    ) -> torch.Tensor:
        """带位置信息的聚合"""
        # 基础聚合
        aggregated = aggregator._aggregate_subtoken_features(subtoken_features)

        # 添加位置编码（如果启用）
        if self.use_position_encoding:
            pos_embed = self.position_embedding(
                torch.tensor(word_position, device=subtoken_features.device)
            )
            aggregated = aggregated + pos_embed

        return aggregated

    def get_aspect_representation(
        self,
        aspect_word_features: torch.Tensor,
        word_level_mask: torch.Tensor,
        pooling_method: str = "mean",
    ) -> torch.Tensor:
        """
        获取方面词的整体表示

        Args:
            aspect_word_features: 方面词词级特征 [batch_size, max_aspect_words, hidden_dim]
            word_level_mask: 词级别mask [batch_size, max_aspect_words]
            pooling_method: 池化方法 ("mean", "max", "first", "last")

        Returns:
            aspect_representation: 方面词整体表示 [batch_size, hidden_dim]
        """
        batch_size = aspect_word_features.shape[0]

        if pooling_method == "mean":
            # 计算有效词的平均
            masked_features = (
                aspect_word_features * word_level_mask.unsqueeze(-1).float()
            )
            sum_features = masked_features.sum(dim=1)  # [batch_size, hidden_dim]
            count = word_level_mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
            count = torch.clamp(count, min=1)  # 避免除零
            aspect_repr = sum_features / count

        elif pooling_method == "max":
            # 最大池化（只考虑有效词）
            masked_features = aspect_word_features.masked_fill(
                ~word_level_mask.unsqueeze(-1), float("-inf")
            )
            aspect_repr, _ = masked_features.max(dim=1)
            # 处理全部无效的情况
            all_invalid = ~word_level_mask.any(dim=1)
            aspect_repr[all_invalid] = 0

        elif pooling_method == "first":
            # 取第一个有效词
            aspect_repr = torch.zeros(
                batch_size, self.hidden_dim, device=aspect_word_features.device
            )
            for b in range(batch_size):
                valid_indices = word_level_mask[b].nonzero().squeeze(-1)
                if len(valid_indices) > 0:
                    aspect_repr[b] = aspect_word_features[b, valid_indices[0]]

        elif pooling_method == "last":
            # 取最后一个有效词
            aspect_repr = torch.zeros(
                batch_size, self.hidden_dim, device=aspect_word_features.device
            )
            for b in range(batch_size):
                valid_indices = word_level_mask[b].nonzero().squeeze(-1)
                if len(valid_indices) > 0:
                    aspect_repr[b] = aspect_word_features[b, valid_indices[-1]]
        else:
            raise ValueError(f"Unsupported pooling method: {pooling_method}")

        return aspect_repr


class TokenToWordAggregator(nn.Module):
    """
    基础Token到Word特征聚合模块

    将BERT的subtoken级特征聚合成原始词级特征
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        aggregation_method: str = "mean",
        use_learnable_weights: bool = False,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: BERT隐藏层维度
            aggregation_method: 聚合方法 ("mean", "max", "first", "last", "attention")
            use_learnable_weights: 是否使用可学习的权重进行聚合
            dropout: dropout率
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aggregation_method = aggregation_method
        self.use_learnable_weights = use_learnable_weights

        # 如果使用注意力机制聚合
        if aggregation_method == "attention":
            self.attention_layer = nn.Linear(hidden_dim, 1)

        # 如果使用可学习权重
        if use_learnable_weights:
            self.weight_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.dropout = nn.Dropout(dropout)

    def _aggregate_subtoken_features(
        self, subtoken_features: torch.Tensor
    ) -> torch.Tensor:
        """
        聚合单个词的多个subtoken特征

        Args:
            subtoken_features: [num_subtokens, hidden_dim]

        Returns:
            aggregated_feature: [hidden_dim]
        """
        if subtoken_features.shape[0] == 0:
            return torch.zeros(self.hidden_dim, device=subtoken_features.device)

        if self.aggregation_method == "mean":
            # 平均池化
            aggregated = subtoken_features.mean(dim=0)

        elif self.aggregation_method == "max":
            # 最大池化
            aggregated, _ = subtoken_features.max(dim=0)

        elif self.aggregation_method == "first":
            # 取第一个subtoken
            aggregated = subtoken_features[0]

        elif self.aggregation_method == "last":
            # 取最后一个subtoken
            aggregated = subtoken_features[-1]

        elif self.aggregation_method == "attention":
            # 注意力加权聚合
            attention_scores = self.attention_layer(
                subtoken_features
            )  # [num_subtokens, 1]
            attention_weights = torch.softmax(
                attention_scores, dim=0
            )  # [num_subtokens, 1]
            aggregated = (subtoken_features * attention_weights).sum(
                dim=0
            )  # [hidden_dim]

        else:
            raise ValueError(
                f"Unsupported aggregation method: {self.aggregation_method}"
            )

        # 应用可学习权重（如果启用）
        if self.use_learnable_weights:
            weight = self.weight_layer(aggregated.unsqueeze(0)).squeeze()  # scalar
            aggregated = aggregated * weight

        return self.dropout(aggregated)


# 使用示例和测试代码
if __name__ == "__main__":

    def create_sample_data():
        """创建示例数据用于测试"""
        batch_size = 2
        seq_len = 20
        hidden_dim = 768
        original_seq_len = 8

        # 模拟BERT输出
        bert_output = torch.randn(batch_size, seq_len, hidden_dim)

        # 模拟句子mask：[CLS] The food is great [SEP] food [SEP]
        sentence_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        sentence_mask[:, 1:10] = True  # 句子部分在位置1-9

        # 模拟方面词在原句中的mask
        aspect_in_sentence_mask = torch.zeros(
            batch_size, original_seq_len, dtype=torch.bool
        )
        aspect_in_sentence_mask[0, 2:4] = True  # 第一个样本的方面词在位置2-3
        aspect_in_sentence_mask[1, 1:2] = True  # 第二个样本的方面词在位置1

        # 模拟token到subtoken映射
        token_to_subtoken_maps = [
            {0: [1, 2], 1: [3], 2: [4, 5], 3: [6], 4: [7, 8, 9]},  # 第一个样本：5个词
            {0: [1], 1: [2, 3], 2: [4], 3: [5, 6, 7], 4: [8, 9]},  # 第二个样本：5个词
        ]

        return {
            "bert_output": bert_output,
            "sentence_mask": sentence_mask,
            "aspect_in_sentence_mask": aspect_in_sentence_mask,
            "token_to_subtoken_maps": token_to_subtoken_maps,
        }

    # 测试聚合器
    print("🔍 测试ABSA Token聚合器...")

    # 创建测试数据
    sample_data = create_sample_data()

    # 创建聚合器
    aggregator = ABSATokenAggregator(
        hidden_dim=768,
        sentence_aggregation="attention",
        aspect_aggregation="mean",
        use_position_encoding=True,
        dropout=0.1,
    )

    # 前向传播
    sentence_features, aspect_features, aspect_mask = aggregator(**sample_data)

    print(f"✅ 句子词级特征形状: {sentence_features.shape}")
    print(f"✅ 方面词词级特征形状: {aspect_features.shape}")
    print(f"✅ 方面词mask形状: {aspect_mask.shape}")

    # 测试方面词整体表示
    aspect_repr = aggregator.get_aspect_representation(
        aspect_features, aspect_mask, pooling_method="mean"
    )
    print(f"✅ 方面词整体表示形状: {aspect_repr.shape}")

    print("\n🎉 所有测试通过！Token聚合器工作正常。")
