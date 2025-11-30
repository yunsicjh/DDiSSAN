"""
跨模态注意力机制模块
实现语义特征和语法特征之间的交叉注意力交互

根据新prompt.txt的设计：
1. 语义增强的交叉注意力
2. 全局增强的交叉注意力
3. 跨模态交互注意力
4. 预融合模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadCrossAttention(nn.Module):
    """
    多头交叉注意力机制
    实现Query, Key, Value来自不同源的注意力计算

    支持的模式：
    1. 语义增强: Q=H_sem, K=V=[H_sem || G]
    2. 全局增强: Q=G, K=V=[H_sem || G]
    3. 跨模态交互: Q=H_sem, K=V=H_syn 或 Q=H_syn, K=V=H_sem
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Args:
            d_model: 模型维度 (通常是128)
            num_heads: 注意力头数
            dropout: Dropout概率
            bias: 是否使用偏置
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_prob = dropout

        # Q, K, V投影层
        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(d_model, d_model, bias=bias)
        self.W_V = nn.Linear(d_model, d_model, bias=bias)

        # 输出投影
        self.W_O = nn.Linear(d_model, d_model, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 缩放因子
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: 注意力掩码 [batch_size, seq_len_q, seq_len_k]

        Returns:
            output: [batch_size, seq_len_q, d_model]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]

        # 投影到Q, K, V
        Q = self.W_Q(query)  # [batch_size, seq_len_q, d_model]
        K = self.W_K(key)  # [batch_size, seq_len_k, d_model]
        V = self.W_V(value)  # [batch_size, seq_len_v, d_model]

        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(
            1, 2
        )  # [batch_size, num_heads, seq_len_q, d_k]
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(
            1, 2
        )  # [batch_size, num_heads, seq_len_k, d_k]
        V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(
            1, 2
        )  # [batch_size, num_heads, seq_len_v, d_k]

        # 计算注意力
        attention_output = self.scaled_dot_product_attention(
            Q, K, V, mask
        )  # [batch_size, num_heads, seq_len_q, d_k]

        # 合并多头
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, self.d_model)
        )  # [batch_size, seq_len_q, d_model]

        # 输出投影
        output = self.W_O(attention_output)

        return output

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        缩放点积注意力
        """
        # 计算注意力分数
        scores = (
            torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        )  # [batch_size, num_heads, seq_len_q, seq_len_k]

        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用权重到Value
        output = torch.matmul(
            attention_weights, V
        )  # [batch_size, num_heads, seq_len_q, d_k]

        return output


class FeedForwardNetwork(nn.Module):
    """
    前馈网络模块 (FFN)
    实现 FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        """
        Args:
            d_model: 模型维度
            d_ff: 前馈网络隐藏维度 (默认为4*d_model)
            dropout: Dropout概率
            activation: 激活函数类型
        """
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class CrossAttentionLayer(nn.Module):
    """
    完整的交叉注意力层
    包含多头交叉注意力 + 残差连接 + 层归一化 + FFN

    实现公式：
    ĤH^{l+1} = LayerNorm(H^l + MSA(W_Q^l H^l, W_K^l [H^l || G^l], W_V^l [H^l || G^l]))
    H̃^{l+1} = LayerNorm(Ĥ^{l+1} + FFN(Ĥ^{l+1}))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏维度
            dropout: Dropout概率
        """
        super().__init__()

        self.d_model = d_model

        # 多头交叉注意力
        self.cross_attention = MultiHeadCrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 前馈网络
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            query: Query输入 [batch_size, seq_len_q, d_model]
            key: Key输入 [batch_size, seq_len_k, d_model]
            value: Value输入 [batch_size, seq_len_v, d_model]
            mask: 注意力掩码

        Returns:
            output: [batch_size, seq_len_q, d_model]
        """
        # 第一个子层：多头交叉注意力 + 残差连接 + 层归一化
        residual = query
        attention_output = self.cross_attention(query, key, value, mask)
        attention_output = self.dropout(attention_output)
        query = self.norm1(residual + attention_output)  # 公式(9)的前半部分

        # 第二个子层：前馈网络 + 残差连接 + 层归一化
        residual = query
        ffn_output = self.ffn(query)
        ffn_output = self.dropout(ffn_output)
        output = self.norm2(residual + ffn_output)  # 公式(10)

        return output


class GlobalSemanticEnhancer(nn.Module):
    """
    全局语义增强模块

    实现语义特征和全局特征的交互增强：
    1. 语义增强: H_sem → H_sem_1
    2. 全局增强: G → G_out
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: 模型维度 (128)
            num_heads: 注意力头数
            num_layers: Transformer层数
            dropout: Dropout概率
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # 语义增强层
        self.semantic_layers = nn.ModuleList(
            [
                CrossAttentionLayer(d_model, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # 全局增强层
        self.global_layers = nn.ModuleList(
            [
                CrossAttentionLayer(d_model, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        H_sem: torch.Tensor,
        G: torch.Tensor,
        semantic_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            H_sem: 语义特征 [batch_size, seq_len, d_model]
            G: 全局特征 [batch_size, 1, d_model]
            semantic_mask: 语义特征掩码

        Returns:
            H_sem_1: 增强后的语义特征 [batch_size, seq_len, d_model]
            G_out: 增强后的全局特征 [batch_size, 1, d_model]
        """
        # 准备拼接的Key和Value: [H_sem || G]
        # H_sem: [batch_size, seq_len, d_model]
        # G: [batch_size, 1, d_model]
        G_expanded = G.expand(-1, H_sem.shape[1], -1)  # 扩展G到seq_len长度
        HG_concat = torch.cat(
            [H_sem, G_expanded], dim=1
        )  # [batch_size, seq_len*2, d_model]

        # 多层处理
        for i in range(self.num_layers):
            # 语义增强: Q=H_sem, K=V=[H_sem || G] (公式9-10)
            H_sem = self.semantic_layers[i](
                query=H_sem,
                key=HG_concat,
                value=HG_concat,
                mask=None,  # 可以根据需要添加掩码
            )

            # 全局增强: Q=G, K=V=[H_sem || G] (公式11-12)
            G = self.global_layers[i](
                query=G,
                key=HG_concat,
                value=HG_concat,
                mask=None,
            )

            # 更新拼接特征用于下一层
            G_expanded = G.expand(-1, H_sem.shape[1], -1)
            HG_concat = torch.cat([H_sem, G_expanded], dim=1)

        return H_sem, G  # H_sem_1, G_out


class CrossModalInteraction(nn.Module):
    """
    跨模态交互模块

    实现语义特征H_sem和语法特征H_syn之间的交叉注意力：
    1. H_sem_1 作为Q，H_syn作为K,V → H_sem_out
    2. H_syn作为Q，H_sem作为K,V → H_syn_c
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout概率
        """
        super().__init__()

        # 语义→语法的交叉注意力
        self.sem_to_syn_attention = CrossAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 语法→语义的交叉注意力
        self.syn_to_sem_attention = CrossAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        H_sem: torch.Tensor,
        H_syn: torch.Tensor,
        sem_mask: Optional[torch.Tensor] = None,
        syn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            H_sem: 语义特征 [batch_size, seq_len, d_model]
            H_syn: 语法特征 [batch_size, seq_len, d_model]
            sem_mask: 语义特征掩码
            syn_mask: 语法特征掩码

        Returns:
            H_sem_out: 语义增强后的特征 [batch_size, seq_len, d_model]
            H_syn_c: 语法增强后的特征 [batch_size, seq_len, d_model]
        """
        # H_sem作为Q，H_syn作为K,V
        H_sem_out = self.sem_to_syn_attention(
            query=H_sem,
            key=H_syn,
            value=H_syn,
            mask=syn_mask,
        )

        # H_syn作为Q，H_sem作为K,V
        H_syn_c = self.syn_to_sem_attention(
            query=H_syn,
            key=H_sem,
            value=H_sem,
            mask=sem_mask,
        )

        return H_sem_out, H_syn_c


class PreFusionModule(nn.Module):
    """
    预融合模块

    将H_syn_c与全局特征G_out进行融合，得到H_syn_out
    公式: H_syn_out = LayerNorm(H_syn_c + σ(W_f [H_syn_c || G_out]))
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: 模型维度
            dropout: Dropout概率
        """
        super().__init__()

        self.d_model = d_model

        # 融合投影层: 输入是拼接的特征 [H_syn_c || G_out]
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        # 激活函数
        self.activation = nn.Sigmoid()  # σ(·)

        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        H_syn_c: torch.Tensor,
        G_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            H_syn_c: 语法特征 [batch_size, seq_len, d_model]
            G_out: 全局特征 [batch_size, 1, d_model]

        Returns:
            H_syn_out: 融合后的语法特征 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = H_syn_c.shape

        # 扩展全局特征到序列长度
        G_expanded = G_out.expand(-1, seq_len, -1)  # [batch_size, seq_len, d_model]

        # 拼接特征 [H_syn_c || G_out]
        concat_features = torch.cat(
            [H_syn_c, G_expanded], dim=-1
        )  # [batch_size, seq_len, d_model*2]

        # 融合投影 + 激活
        fusion_output = self.activation(
            self.fusion_proj(concat_features)
        )  # [batch_size, seq_len, d_model]
        fusion_output = self.dropout(fusion_output)

        # 残差连接 + 层归一化
        H_syn_out = self.layer_norm(H_syn_c + fusion_output)

        return H_syn_out


# 测试代码
if __name__ == "__main__":
    print("Testing Cross-Modal Attention Modules...")

    batch_size = 2
    seq_len = 20
    d_model = 128
    num_heads = 8

    # 创建测试数据
    H_sem = torch.randn(batch_size, seq_len, d_model)
    H_syn = torch.randn(batch_size, seq_len, d_model)
    G = torch.randn(batch_size, 1, d_model)

    print(f"Input shapes:")
    print(f"H_sem: {H_sem.shape}")
    print(f"H_syn: {H_syn.shape}")
    print(f"G: {G.shape}")

    # 1. 测试全局语义增强
    print(f"\n1. Testing GlobalSemanticEnhancer:")
    global_enhancer = GlobalSemanticEnhancer(d_model, num_heads)
    H_sem_1, G_out = global_enhancer(H_sem, G)
    print(f"H_sem_1: {H_sem_1.shape}")
    print(f"G_out: {G_out.shape}")

    # 2. 测试跨模态交互
    print(f"\n2. Testing CrossModalInteraction:")
    cross_modal = CrossModalInteraction(d_model, num_heads)
    H_sem_out, H_syn_c = cross_modal(H_sem_1, H_syn)
    print(f"H_sem_out: {H_sem_out.shape}")
    print(f"H_syn_c: {H_syn_c.shape}")

    # 3. 测试预融合模块
    print(f"\n3. Testing PreFusionModule:")
    pre_fusion = PreFusionModule(d_model)
    H_syn_out = pre_fusion(H_syn_c, G_out)
    print(f"H_syn_out: {H_syn_out.shape}")

    # 4. 最终特征拼接
    print(f"\n4. Final Feature Concatenation:")
    H_final = torch.cat([H_sem_out, H_syn_out], dim=-1)
    print(f"H_final: {H_final.shape}")  # [batch_size, seq_len, d_model*2]

    print(f"\n✅ All tests passed!")
