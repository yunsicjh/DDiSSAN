import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DependencyEmbeddingModule(nn.Module):
    """依存关系嵌入模块"""

    def __init__(self, dep_vocab_size: int, embed_dim: int):
        super().__init__()
        self.dep_embedding = nn.Embedding(dep_vocab_size, embed_dim)

    def forward(
        self, adj_matrix: torch.Tensor, dep_rel_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        构建依存关系嵌入矩阵
        Args:
            adj_matrix: 邻接矩阵 [batch_size, seq_len, seq_len]
            dep_rel_matrix: 依存关系标签矩阵 [batch_size, seq_len, seq_len]
        Returns:
            dep_embed_matrix: 依存关系嵌入矩阵 [batch_size, seq_len, seq_len, embed_dim]
        """
        # 获取依存关系嵌入
        dep_embeddings = self.dep_embedding(
            dep_rel_matrix
        )  # [batch, seq_len, seq_len, embed_dim]

        # 只保留有边的位置的嵌入
        adj_mask = adj_matrix.unsqueeze(-1)  # [batch, seq_len, seq_len, 1]
        dep_embed_matrix = (
            dep_embeddings * adj_mask
        )  # [batch, seq_len, seq_len, embed_dim]

        return dep_embed_matrix
