import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class CrossAttentionModule(nn.Module):
    """
    优化的交叉注意力模块，用于计算方面词对句子的注意力

    特性：
    - 支持多头交叉注意力
    - 高效的方面词聚合策略
    - 灵活的query-key-value配置
    - 改进的mask处理机制
    - 支持多种聚合方式
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        aspect_aggregation: str = "mean",  # mean, max, attention
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.aspect_aggregation = aspect_aggregation

        assert (
            hidden_dim % num_heads == 0
        ), f"hidden_dim({hidden_dim}) must be divisible by num_heads({num_heads})"

        # 线性投影层
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # 正则化组件
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        # 方面词注意力聚合（当使用attention模式时）
        if aspect_aggregation == "attention":
            self.aspect_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

        self.scale = self.head_dim**-0.5
        self._init_weights()

    def _init_weights(self):
        """初始化模型参数"""
        for module in [self.query, self.key, self.value, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        sentence_mask: Optional[torch.Tensor] = None,
        aspect_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算交叉注意力矩阵

        Args:
            h1: 句子隐藏状态 [batch_size, seq_len, hidden_dim]
            h2: 方面词隐藏状态 [batch_size, aspect_len, hidden_dim]
            sentence_mask: 句子mask [batch_size, seq_len]
            aspect_mask: 方面词mask [batch_size, aspect_len]

        Returns:
            attention_matrix: 交叉注意力权重矩阵 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, hidden_dim = h1.shape

        # 1. 聚合方面词表示
        h2_aggregated = self._aggregate_aspect_features(
            h2, aspect_mask
        )  # [batch, hidden_dim]

        # 2. 复制为句子长度
        h3 = h2_aggregated.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # [batch, seq_len, hidden_dim]

        # 3. 计算交叉注意力
        attention_matrix = self._compute_cross_attention(h3, h1, sentence_mask)

        return attention_matrix

    def forward_with_explicit_qkv(
        self,
        query: torch.Tensor,  # [batch_size, seq_len, hidden_dim]
        key_value: torch.Tensor,  # [batch_size, seq_len, hidden_dim]
        query_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        key_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        return_enhanced_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        使用显式的query、key、value计算交叉注意力

        专门用于词级SSAN：c_a重复作为query，c_s_bi作为key和value

        Args:
            query: 查询向量 [batch_size, seq_len, hidden_dim] (重复的方面词特征)
            key_value: 键值向量 [batch_size, seq_len, hidden_dim] (句子词级特征)
            query_mask: 查询mask [batch_size, seq_len]
            key_mask: 键值mask [batch_size, seq_len]
            return_enhanced_features: 是否返回增强的特征

        Returns:
            attention_matrix: 交叉注意力矩阵 [batch_size, seq_len, seq_len]
            enhanced_features: 如果return_enhanced_features=True，返回注意力增强的特征
        """
        batch_size, seq_len, hidden_dim = query.shape

        # 残差连接
        residual = key_value

        # 多头注意力投影
        Q = self._reshape_for_multihead(
            self.query(query)
        )  # [batch, heads, seq_len, head_dim]
        K = self._reshape_for_multihead(
            self.key(key_value)
        )  # [batch, heads, seq_len, head_dim]
        V = self._reshape_for_multihead(
            self.value(key_value)
        )  # [batch, heads, seq_len, head_dim]

        # 计算注意力分数
        attention_scores = (
            torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        )  # [batch, heads, seq_len, seq_len]

        # 应用mask
        if key_mask is not None:
            attention_scores = self._apply_cross_mask(
                attention_scores, query_mask, key_mask
            )

        # 计算注意力权重
        attention_weights = F.softmax(
            attention_scores, dim=-1
        )  # [batch, heads, seq_len, seq_len]
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        attended_values = torch.matmul(
            attention_weights, V
        )  # [batch, heads, seq_len, head_dim]

        # 重组多头输出
        attended_values = self._reshape_from_multihead(
            attended_values
        )  # [batch, seq_len, hidden_dim]

        # 输出投影
        enhanced_features = self.out_proj(attended_values)

        # 残差连接和层归一化
        enhanced_features = self.layer_norm(enhanced_features + residual)

        # 返回注意力权重矩阵（平均所有头）
        attention_matrix = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]

        if return_enhanced_features:
            return attention_matrix, enhanced_features
        else:
            return attention_matrix

    def _aggregate_aspect_features(
        self, h2: torch.Tensor, aspect_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """聚合方面词特征"""
        if aspect_mask is None:
            # 没有mask，直接聚合
            if self.aspect_aggregation == "mean":
                return h2.mean(dim=1)
            elif self.aspect_aggregation == "max":
                return h2.max(dim=1)[0]
            else:
                raise ValueError(
                    f"No mask provided but aggregation mode '{self.aspect_aggregation}' requires mask"
                )

        batch_size, aspect_len, hidden_dim = h2.shape

        if self.aspect_aggregation == "mean":
            # 加权平均聚合
            aspect_lengths = aspect_mask.sum(dim=1, keepdim=True).float()  # [batch, 1]
            aspect_lengths = torch.clamp(aspect_lengths, min=1.0)  # 避免除零

            h2_masked = (
                h2 * aspect_mask.unsqueeze(-1).float()
            )  # [batch, aspect_len, hidden_dim]
            h2_aggregated = h2_masked.sum(dim=1) / aspect_lengths  # [batch, hidden_dim]

        elif self.aspect_aggregation == "max":
            # 最大池化聚合
            h2_masked = h2.masked_fill(~aspect_mask.unsqueeze(-1), float("-inf"))
            h2_aggregated = h2_masked.max(dim=1)[0]  # [batch, hidden_dim]

        elif self.aspect_aggregation == "attention":
            # 注意力加权聚合
            attention_scores = self.aspect_attention(h2).squeeze(
                -1
            )  # [batch, aspect_len]
            attention_scores = attention_scores.masked_fill(~aspect_mask, float("-inf"))
            attention_weights = F.softmax(
                attention_scores, dim=1
            )  # [batch, aspect_len]

            h2_aggregated = torch.sum(
                h2 * attention_weights.unsqueeze(-1), dim=1
            )  # [batch, hidden_dim]

        else:
            raise ValueError(
                f"Unknown aspect aggregation mode: {self.aspect_aggregation}"
            )

        return h2_aggregated

    def _compute_cross_attention(
        self,
        query_features: torch.Tensor,
        key_value_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算交叉注意力"""
        batch_size, seq_len, hidden_dim = query_features.shape

        # 多头注意力投影
        Q = self._reshape_for_multihead(
            self.query(query_features)
        )  # [batch, heads, seq_len, head_dim]
        K = self._reshape_for_multihead(
            self.key(key_value_features)
        )  # [batch, heads, seq_len, head_dim]
        V = self._reshape_for_multihead(
            self.value(key_value_features)
        )  # [batch, heads, seq_len, head_dim]

        # 计算注意力分数
        attention_scores = (
            torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        )  # [batch, heads, seq_len, seq_len]

        # 应用mask
        if mask is not None:
            attention_scores = self._apply_mask(attention_scores, mask)

        # 计算注意力权重
        attention_weights = F.softmax(
            attention_scores, dim=-1
        )  # [batch, heads, seq_len, seq_len]

        # 平均多个头的注意力权重
        attention_matrix = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]

        return attention_matrix

    def _reshape_for_multihead(self, x: torch.Tensor) -> torch.Tensor:
        """将输入重塑为多头格式"""
        batch_size, seq_len, hidden_dim = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def _reshape_from_multihead(self, x: torch.Tensor) -> torch.Tensor:
        """将多头输出重塑回原始格式"""
        batch_size, num_heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

    def _apply_mask(
        self, attention_scores: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """应用mask到注意力分数"""
        batch_size, num_heads, seq_len, _ = attention_scores.shape

        # 扩展mask维度
        expanded_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        expanded_mask = expanded_mask.expand(batch_size, num_heads, seq_len, seq_len)

        return attention_scores.masked_fill(~expanded_mask, float("-inf"))

    def _apply_cross_mask(
        self,
        attention_scores: torch.Tensor,
        query_mask: Optional[torch.Tensor],
        key_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """应用交叉注意力的mask"""
        batch_size, num_heads, seq_len, _ = attention_scores.shape

        # Key mask：防止注意到无效的key位置
        if key_mask is not None:
            key_mask_expanded = key_mask.unsqueeze(1).unsqueeze(
                2
            )  # [batch, 1, 1, seq_len]
            key_mask_expanded = key_mask_expanded.expand(
                batch_size, num_heads, seq_len, seq_len
            )
            attention_scores = attention_scores.masked_fill(
                ~key_mask_expanded, float("-inf")
            )

        # Query mask：如果query位置无效，则整行置零（在softmax后处理）
        if query_mask is not None:
            query_mask_expanded = query_mask.unsqueeze(1).unsqueeze(
                3
            )  # [batch, 1, seq_len, 1]
            query_mask_expanded = query_mask_expanded.expand(
                batch_size, num_heads, seq_len, seq_len
            )
            attention_scores = attention_scores.masked_fill(
                ~query_mask_expanded, float("-inf")
            )

        return attention_scores

    def get_attention_info(self) -> dict:
        """返回注意力模块的配置信息"""
        return {
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "aspect_aggregation": self.aspect_aggregation,
            "has_dropout": not isinstance(self.dropout, nn.Identity),
            "has_layer_norm": not isinstance(self.layer_norm, nn.Identity),
        }
