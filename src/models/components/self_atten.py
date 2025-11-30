import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SelfAttentionModule(nn.Module):
    """
    优化的自注意力模块，用于计算句子的自注意力矩阵

    特性：
    - 支持多头注意力机制
    - 高效的mask处理
    - 可选的dropout和层归一化
    - 支持不同的注意力计算模式
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        attention_mode: str = "scaled_dot_product",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attention_mode = attention_mode

        assert (
            hidden_dim % num_heads == 0
        ), f"hidden_dim({hidden_dim}) must be divisible by num_heads({num_heads})"

        # 线性投影层
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # 正则化和dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        # 缩放因子
        self.scale = self.head_dim**-0.5

        # 初始化参数
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
        sentence_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        计算自注意力

        Args:
            h1: 句子隐藏状态 [batch_size, seq_len, hidden_dim]
            sentence_mask: 句子mask [batch_size, seq_len], True表示有效位置
            return_attention_weights: 是否返回注意力权重

        Returns:
            如果return_attention_weights=False:
                attention_matrix: 自注意力权重矩阵 [batch_size, seq_len, seq_len]
            如果return_attention_weights=True:
                (enhanced_features, attention_matrix): 增强特征和注意力权重
        """
        batch_size, seq_len, hidden_dim = h1.shape

        # 残差连接
        residual = h1

        # 多头注意力投影
        Q = self._reshape_for_multihead(
            self.query(h1)
        )  # [batch, heads, seq_len, head_dim]
        K = self._reshape_for_multihead(
            self.key(h1)
        )  # [batch, heads, seq_len, head_dim]
        V = self._reshape_for_multihead(
            self.value(h1)
        )  # [batch, heads, seq_len, head_dim]

        # 计算注意力分数
        attention_scores = self._compute_attention_scores(
            Q, K
        )  # [batch, heads, seq_len, seq_len]

        # 应用mask
        if sentence_mask is not None:
            attention_scores = self._apply_mask(attention_scores, sentence_mask)

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
        output = self.out_proj(attended_values)

        # 残差连接和层归一化
        output = self.layer_norm(output + residual)

        # 平均多头注意力权重用于返回
        attention_matrix = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]

        if return_attention_weights:
            return output, attention_matrix
        else:
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

    def _compute_attention_scores(
        self, Q: torch.Tensor, K: torch.Tensor
    ) -> torch.Tensor:
        """计算注意力分数"""
        if self.attention_mode == "scaled_dot_product":
            return torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        elif self.attention_mode == "additive":
            # 加性注意力机制（可选实现）
            raise NotImplementedError("Additive attention not implemented yet")
        else:
            raise ValueError(f"Unknown attention mode: {self.attention_mode}")

    def _apply_mask(
        self, attention_scores: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """应用mask到注意力分数"""
        batch_size, num_heads, seq_len, _ = attention_scores.shape

        # 扩展mask维度: [batch, seq_len] -> [batch, heads, seq_len, seq_len]
        expanded_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        expanded_mask = expanded_mask.expand(batch_size, num_heads, seq_len, seq_len)

        # 对行进行mask（query positions）
        row_mask = expanded_mask
        # 对列进行mask（key positions）
        col_mask = (
            mask.unsqueeze(1)
            .unsqueeze(3)
            .expand(batch_size, num_heads, seq_len, seq_len)
        )

        # 同时应用行列mask
        final_mask = row_mask & col_mask

        return attention_scores.masked_fill(~final_mask, float("-inf"))

    def get_attention_info(self) -> dict:
        """返回注意力模块的配置信息"""
        return {
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "attention_mode": self.attention_mode,
            "has_dropout": not isinstance(self.dropout, nn.Identity),
            "has_layer_norm": not isinstance(self.layer_norm, nn.Identity),
        }
