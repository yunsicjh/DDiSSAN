"""
语义增强模块：实现多层交叉注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .diffcross_atten import MultiHeadDifferentialAttention


class MultiLayerCrossAttention(nn.Module):
    """
    多层交叉注意力模块：用于语义特征增强
    实现公式：
    H_sem^{l+1} = LayerNorm(H_sem^l + MSA(W_Q^l H_sem^l, W_K^l [H_sem^l || G^l], W_V^l [H_sem^l || G^l]))
    H_sem^{l+1} = LayerNorm(H_sem^{l+1} + FFN(H_sem^{l+1}))
    """

    def __init__(self, d_model=768, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads

        # 多层注意力
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    d_model, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )

        # 层归一化
        self.layer_norms1 = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )
        self.layer_norms2 = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )

        # 前馈网络
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, H_sem, G):
        """
        Args:
            H_sem: 语义特征 [B, seq_len, d_model]
            G: 全局特征 [B, d_model]
        Returns:
            增强后的语义特征 [B, seq_len, d_model]
        """
        B, seq_len, d_model = H_sem.shape

        # 扩展G到序列长度维度
        G_expanded = G.unsqueeze(1).expand(B, seq_len, d_model)  # [B, seq_len, d_model]

        current_H_sem = H_sem

        for i in range(self.num_layers):
            # 构建Key和Value：[H_sem || G]
            kv_input = torch.cat(
                [current_H_sem, G_expanded], dim=-1
            )  # [B, seq_len, 2*d_model]

            # 投影到正确维度
            key_proj = nn.Linear(2 * d_model, d_model).to(H_sem.device)
            value_proj = nn.Linear(2 * d_model, d_model).to(H_sem.device)

            key = key_proj(kv_input)  # [B, seq_len, d_model]
            value = value_proj(kv_input)  # [B, seq_len, d_model]

            # 多头注意力
            attn_output, _ = self.attention_layers[i](
                query=current_H_sem, key=key, value=value
            )

            # 残差连接和层归一化
            current_H_sem = self.layer_norms1[i](current_H_sem + attn_output)

            # FFN
            ffn_output = self.ffn_layers[i](current_H_sem)
            current_H_sem = self.layer_norms2[i](current_H_sem + ffn_output)

        return current_H_sem


class GlobalFeatureEnhancer(nn.Module):
    """
    全局特征增强模块：用于增强全局特征G
    实现公式：
    G^{l+1} = LayerNorm(G^l + MSA(W_Q G^l, W_K [H_sem^l || G^l], W_V [H_sem^l || G^l]))
    G^{l+1} = LayerNorm(G^{l+1} + FFN(G^{l+1}))
    """

    def __init__(self, d_model=768, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads

        # 多层注意力（用于全局特征）
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    d_model, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )

        # 层归一化
        self.layer_norms1 = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )
        self.layer_norms2 = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )

        # 前馈网络
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, G, H_sem):
        """
        Args:
            G: 全局特征 [B, d_model]
            H_sem: 语义特征 [B, seq_len, d_model]
        Returns:
            增强后的全局特征 [B, d_model]
        """
        B, seq_len, d_model = H_sem.shape

        # 为G添加序列维度以适配注意力机制
        current_G = G.unsqueeze(1)  # [B, 1, d_model]

        for i in range(self.num_layers):
            # 扩展G到序列长度维度用于拼接
            G_expanded = current_G.expand(B, seq_len, d_model)  # [B, seq_len, d_model]

            # 构建Key和Value：[H_sem || G]
            kv_input = torch.cat([H_sem, G_expanded], dim=-1)  # [B, seq_len, 2*d_model]

            # 投影到正确维度
            key_proj = nn.Linear(2 * d_model, d_model).to(G.device)
            value_proj = nn.Linear(2 * d_model, d_model).to(G.device)

            key = key_proj(kv_input)  # [B, seq_len, d_model]
            value = value_proj(kv_input)  # [B, seq_len, d_model]

            # 多头注意力
            attn_output, _ = self.attention_layers[i](
                query=current_G, key=key, value=value
            )

            # 残差连接和层归一化
            current_G = self.layer_norms1[i](current_G + attn_output)

            # FFN
            ffn_output = self.ffn_layers[i](current_G)
            current_G = self.layer_norms2[i](current_G + ffn_output)

        # 移除序列维度返回
        return current_G.squeeze(1)  # [B, d_model]


class DifferentialCrossAttentionLayer(nn.Module):
    """
    差分交叉注意力层：实现交叉注意力版本的差分注意力机制
    """

    def __init__(self, d_model=768, num_heads=8, dropout=0.1, lambda_init=0.8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        assert d_model % num_heads == 0

        # 投影矩阵 - 支持交叉注意力
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # 差分注意力的Lambda参数
        self.lambda_init = lambda_init
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))

        # 层归一化
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Args:
            query: 查询向量 [B, seq_len_q, d_model]
            key: 键向量 [B, seq_len_k, d_model]
            value: 值向量 [B, seq_len_v, d_model]
        Returns:
            增强后的查询特征 [B, seq_len_q, d_model]
        """
        B, seq_len_q, d_model = query.shape
        seq_len_k = key.shape[1]

        # 投影得到Q, K, V
        Q = (
            self.W_q(query)
            .view(B, seq_len_q, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        K = (
            self.W_k(key)
            .view(B, seq_len_k, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        V = (
            self.W_v(value)
            .view(B, seq_len_k, self.num_heads, self.d_head)
            .transpose(1, 2)
        )

        # 计算注意力分数
        scale = math.sqrt(self.d_head)
        scores = (
            torch.matmul(Q, K.transpose(-2, -1)) / scale
        )  # [B, num_heads, seq_len_q, seq_len_k]

        # 标准注意力
        attn_weights_1 = F.softmax(scores, dim=-1)

        # 差分注意力：第二个注意力头
        attn_weights_2 = F.softmax(scores - 1.0, dim=-1)  # 简化的差分机制

        # 差分组合
        lambda_val = torch.sigmoid(self.lambda_param)
        attn_weights = attn_weights_1 - lambda_val * attn_weights_2

        # 应用dropout
        attn_weights = self.dropout(attn_weights)

        # 计算输出
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, seq_len_q, d_head]
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, seq_len_q, d_model)
        )

        # 输出投影
        attn_output = self.W_o(attn_output)

        # 残差连接和层归一化
        query_enhanced = self.layer_norm1(query + attn_output)

        # FFN
        ffn_output = self.ffn(query_enhanced)
        query_final = self.layer_norm2(query_enhanced + ffn_output)

        return query_final


class MultiLayerDifferentialCrossAttention(nn.Module):
    """
    多层差分交叉注意力模块
    """

    def __init__(
        self, d_model=768, num_heads=8, num_layers=3, dropout=0.1, lambda_init=0.8
    ):
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                DifferentialCrossAttentionLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    lambda_init=lambda_init,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, query, key, value):
        """
        Args:
            query: 查询向量 [B, seq_len_q, d_model]
            key: 键向量 [B, seq_len_k, d_model]
            value: 值向量 [B, seq_len_v, d_model]
        Returns:
            增强后的查询特征 [B, seq_len_q, d_model]
        """
        current_query = query

        for layer in self.layers:
            current_query = layer(current_query, key, value)

        return current_query


class PreFusionModule(nn.Module):
    """
    预融合模块：将H_syn_c与G_out融合得到H_syn_out
    实现公式：H_syn_out = LayerNorm(H_syn_c^L + σ(W_f [H_syn_c^L || G_out^L]))
    """

    def __init__(self, d_model=768):
        super().__init__()
        self.d_model = d_model

        # 融合权重
        self.fusion_proj = nn.Linear(2 * d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, H_syn_c, G_out):
        """
        Args:
            H_syn_c: 语法特征 [B, seq_len, d_model]
            G_out: 全局特征 [B, d_model]
        Returns:
            融合后的语法特征 [B, seq_len, d_model]
        """
        B, seq_len, d_model = H_syn_c.shape

        # 扩展G_out到序列长度
        G_expanded = G_out.unsqueeze(1).expand(B, seq_len, d_model)

        # 拼接特征
        concat_features = torch.cat(
            [H_syn_c, G_expanded], dim=-1
        )  # [B, seq_len, 2*d_model]

        # 融合投影
        fusion_output = torch.sigmoid(
            self.fusion_proj(concat_features)
        )  # [B, seq_len, d_model]

        # 残差连接和层归一化
        H_syn_out = self.layer_norm(H_syn_c + fusion_output)

        return H_syn_out


class FinalDifferentialAttention(nn.Module):
    """
    最终差分注意力模块：对拼接后的H_out进行3层差分注意力处理
    """

    def __init__(
        self, d_model=256, num_heads=8, num_layers=3, dropout=0.1, lambda_init=0.8
    ):
        super().__init__()
        self.num_layers = num_layers

        # 多层差分自注意力
        self.layers = nn.ModuleList(
            [
                DifferentialCrossAttentionLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    lambda_init=lambda_init,
                )
                for _ in range(num_layers)
            ]
        )

        # 残差连接的层归一化
        self.residual_norm = nn.LayerNorm(d_model)

    def forward(self, H_out):
        """
        Args:
            H_out: 拼接后的特征 [B, seq_len, d_model]
        Returns:
            H_out_f: 增强后的特征 [B, seq_len, d_model]
            H_out_residual: 残差连接后的特征 [B, seq_len, d_model]
        """
        current_features = H_out

        # 多层差分自注意力（自注意力：Q=K=V）
        for layer in self.layers:
            current_features = layer(
                current_features, current_features, current_features
            )

        H_out_f = current_features

        # 残差连接
        H_out_residual = self.residual_norm(H_out + H_out_f)

        return H_out_f, H_out_residual


class ProgressiveDimensionReduction(nn.Module):
    """
    渐进降维模块：将高维特征渐进降维到合适的分类维度
    """

    def __init__(
        self, input_dim=256, hidden_dims=[128, 64], output_dim=32, dropout=0.2
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        # 最终输出层
        layers.append(nn.Linear(current_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, seq_len, input_dim]
        Returns:
            降维后的特征 [B, seq_len, output_dim]
        """
        return self.layers(x)
