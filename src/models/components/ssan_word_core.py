import torch
import torch.nn as nn
from typing import Tuple
from .self_atten import SelfAttentionModule
from .cross_atten import CrossAttentionModule
from .dep_matrix import DependencyEmbeddingModule


class SSANWordLevelCore(nn.Module):
    """
    SSAN词级核心模块

    专门处理词级特征，构造基于词级的三个关键矩阵：
    1. 自注意力矩阵A：c_s_bi内部的自注意力
    2. 交叉注意力矩阵B：c_a与c_s_bi的交叉注意力
    3. 依存关系嵌入矩阵：基于词级的依存关系
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        dep_vocab_size: int = 100,
        dep_embed_dim: int = 30,
        num_heads: int = 8,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.self_attention = SelfAttentionModule(hidden_dim, num_heads)
        self.cross_attention = CrossAttentionModule(hidden_dim, num_heads)
        self.dep_embedding = DependencyEmbeddingModule(dep_vocab_size, dep_embed_dim)

    def forward(
        self,
        sentence_word_features: torch.Tensor,  # c_s_bi: BiLSTM增强后的句子词级特征
        aspect_word_features: torch.Tensor,  # c_a: 方面词词级特征
        aspect_word_mask: torch.Tensor,  # 方面词mask
        adj_matrix: torch.Tensor,  # 词级依存邻接矩阵
        dep_rel_matrix: torch.Tensor,  # 词级依存关系矩阵
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构造三个关键矩阵（词级版本）

        Args:
            sentence_word_features: [batch_size, max_words, hidden_dim] BiLSTM增强的句子词级特征
            aspect_word_features: [batch_size, max_aspect_words, hidden_dim] 方面词词级特征
            aspect_word_mask: [batch_size, max_aspect_words] 方面词有效性mask
            adj_matrix: [batch_size, max_words, max_words] 词级依存邻接矩阵
            dep_rel_matrix: [batch_size, max_words, max_words] 词级依存关系矩阵

        Returns:
            sentence_features: 句子词级特征 [batch_size, max_words, hidden_dim]
            attention_matrix_a: 自注意力矩阵 [batch_size, max_words, max_words]
            attention_matrix_b: 交叉注意力矩阵 [batch_size, max_words, max_words]
            dep_embed_matrix: 依存关系嵌入矩阵 [batch_size, max_words, max_words, embed_dim]
        """
        batch_size, max_words, hidden_dim = sentence_word_features.shape

        # 1. 创建句子词级mask
        sentence_word_mask = (
            sentence_word_features.abs().sum(dim=-1) > 1e-6
        )  # [batch_size, max_words]

        # 2. 自注意力矩阵A：c_s_bi做自注意力
        attention_matrix_a = self.self_attention(
            sentence_word_features, sentence_word_mask
        )  # [batch_size, max_words, max_words]

        # 3. 交叉注意力矩阵B：c_a重复作为query，c_s_bi作为key和value
        attention_matrix_b = self._compute_cross_attention(
            sentence_word_features,
            aspect_word_features,
            sentence_word_mask,
            aspect_word_mask,
        )  # [batch_size, max_words, max_words]

        # 4. 依存关系嵌入矩阵（基于词级）
        dep_embed_matrix = self.dep_embedding(adj_matrix, dep_rel_matrix)

        return (
            sentence_word_features,
            attention_matrix_a,
            attention_matrix_b,
            dep_embed_matrix,
        )

    def _compute_cross_attention(
        self,
        sentence_features: torch.Tensor,  # c_s_bi [batch_size, max_words, hidden_dim]
        aspect_features: torch.Tensor,  # c_a [batch_size, max_aspect_words, hidden_dim]
        sentence_mask: torch.Tensor,  # [batch_size, max_words]
        aspect_mask: torch.Tensor,  # [batch_size, max_aspect_words]
    ) -> torch.Tensor:
        """
        计算交叉注意力：c_a重复4次作为query，c_s_bi作为key和value

        正确逻辑：
        - Query: c_a重复max_words次 → [batch_size, max_words, hidden_dim]
        - Key & Value: c_s_bi → [batch_size, max_words, hidden_dim]
        - 输出：交叉注意力矩阵 [batch_size, max_words, max_words]
        """
        batch_size, max_words, hidden_dim = sentence_features.shape
        max_aspect_words = aspect_features.shape[1]

        # 1. 获取方面词的平均表示（处理多个方面词的情况）
        if aspect_mask.any():
            # 计算每个样本的有效方面词表示
            aspect_repr_list = []
            for b in range(batch_size):
                valid_aspects = aspect_features[
                    b, aspect_mask[b]
                ]  # [num_valid_aspects, hidden_dim]
                if valid_aspects.shape[0] > 0:
                    aspect_repr = valid_aspects.mean(dim=0)  # [hidden_dim]
                else:
                    aspect_repr = torch.zeros(hidden_dim, device=aspect_features.device)
                aspect_repr_list.append(aspect_repr)

            aspect_repr = torch.stack(aspect_repr_list)  # [batch_size, hidden_dim]
        else:
            aspect_repr = torch.zeros(
                batch_size, hidden_dim, device=aspect_features.device
            )

        # 2. 将方面词表示重复max_words次作为query
        aspect_query = aspect_repr.unsqueeze(1).expand(
            -1, max_words, -1
        )  # [batch_size, max_words, hidden_dim]

        # 3. 使用交叉注意力模块计算注意力矩阵
        # 这里我们需要创建合适的mask
        aspect_query_mask = sentence_mask  # query的mask与sentence相同

        attention_matrix_b = self.cross_attention.forward_with_explicit_qkv(
            query=aspect_query,  # [batch_size, max_words, hidden_dim]
            key_value=sentence_features,  # [batch_size, max_words, hidden_dim]
            query_mask=aspect_query_mask,  # [batch_size, max_words]
            key_mask=sentence_mask,  # [batch_size, max_words]
        )

        return attention_matrix_b  # [batch_size, max_words, max_words]


# 为了兼容性，保留原始的SSANCore类
class SSANCore(SSANWordLevelCore):
    """兼容性别名，使用词级处理"""

    def __init__(self, bert_hidden_dim: int = 768, **kwargs):
        super().__init__(hidden_dim=bert_hidden_dim, **kwargs)

    def extract_sentence_aspect_features(self, *args, **kwargs):
        """保持兼容性的方法，但现在建议直接使用词级特征"""
        raise NotImplementedError(
            "This method is deprecated. Please use word-level features directly "
            "with the ABSATokenAggregator module."
        )
