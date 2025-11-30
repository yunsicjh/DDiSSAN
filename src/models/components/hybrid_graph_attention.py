import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops, degree, softmax
    from torch_geometric.typing import Adj, OptTensor, PairTensor

    HAS_TORCH_GEOMETRIC = True
except ImportError:
    print(
        "Warning: torch_geometric not installed. Please install it using: pip install torch_geometric"
    )

    # Define dummy classes for development
    class MessagePassing:
        def __init__(self, **kwargs):
            pass

    Adj = Union[torch.Tensor, None]
    OptTensor = Optional[torch.Tensor]
    PairTensor = Tuple[torch.Tensor, torch.Tensor]
    HAS_TORCH_GEOMETRIC = False

try:
    # 相对导入（在包内使用时）
    from .diffgraph_atten import DifferentialGraphAttention, RMSNorm
except ImportError:
    # 绝对导入（直接运行此文件时）
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from diffgraph_atten import DifferentialGraphAttention, RMSNorm


class RelationActivationAttention(MessagePassing):
    """
    关系激活注意力机制
    实现Transformer风格的Query-Key-Value注意力，但将关系信息融入Key和Value中

    公式:
    t_ij = (W_Q h_i)^T (W_K h_j + W_Kr r_ij) / √d_k
    b_ij = softmax(t_ij)
    h_actv = Σ b_ij (W_V h_j + W_Vr r_ij)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        """
        Args:
            in_channels (int): 输入特征维度
            out_channels (int): 输出特征维度
            edge_dim (int, optional): 边特征维度
            heads (int): 多头注意力数量
            concat (bool): 是否拼接多头输出
            dropout (float): Dropout概率
            add_self_loops (bool): 是否添加自环
            bias (bool): 是否使用偏置
        """
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim

        # Query, Key, Value变换矩阵
        self.W_Q = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.W_K = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.W_V = nn.Linear(in_channels, heads * out_channels, bias=False)

        # 关系嵌入的Key和Value变换矩阵
        if edge_dim is not None:
            self.W_Kr = nn.Linear(edge_dim, heads * out_channels, bias=False)
            self.W_Vr = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.W_Kr = None
            self.W_Vr = None

        # 缩放因子
        self.scale = 1.0 / math.sqrt(out_channels)

        # 输出投影
        if concat:
            self.out_proj = nn.Linear(
                heads * out_channels, heads * out_channels, bias=bias
            )
        else:
            self.out_proj = nn.Linear(out_channels, out_channels, bias=bias)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)

        if self.edge_dim is not None:
            nn.init.xavier_uniform_(self.W_Kr.weight)
            nn.init.xavier_uniform_(self.W_Vr.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播

        Args:
            x (Tensor): 节点特征矩阵 [N, in_channels]
            edge_index (LongTensor): 边索引 [2, M]
            edge_attr (Tensor, optional): 边特征矩阵 [M, edge_dim]

        Returns:
            Tensor: 输出节点嵌入 [N, heads * out_channels] 或 [N, out_channels]
        """
        H, C = self.heads, self.out_channels
        N = x.size(0)

        if self.add_self_loops:
            if isinstance(edge_index, torch.Tensor):
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value="mean", num_nodes=N
                )

        # 计算Query, Key, Value
        Q = self.W_Q(x).view(-1, H, C)  # [N, H, C]
        K = self.W_K(x).view(-1, H, C)  # [N, H, C]
        V = self.W_V(x).view(-1, H, C)  # [N, H, C]

        # 消息传播
        out = self.propagate(
            edge_index,
            Q=Q,
            K=K,
            V=V,
            edge_attr=edge_attr,
            size=(N, N),
        )

        # 输出处理
        if self.concat:
            out = out.view(-1, H * C)  # [N, H*C]
        else:
            out = out.mean(dim=1)  # [N, C]

        out = self.out_proj(out)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, Q_i, K_j, V_j, edge_attr, index, ptr, size_i):
        """
        计算消息，实现关系激活注意力

        Args:
            Q_i: 目标节点的Query [E, H, C]
            K_j: 源节点的Key [E, H, C]
            V_j: 源节点的Value [E, H, C]
            edge_attr: 边属性 [E, edge_dim]
            index: 边索引
            ptr: 指针
            size_i: 目标节点数量

        Returns:
            Tensor: 消息 [E, H, C]
        """
        # 计算关系增强的Key和Value
        K_enhanced = K_j  # [E, H, C]
        V_enhanced = V_j  # [E, H, C]

        if edge_attr is not None and self.W_Kr is not None:
            # 关系嵌入变换
            edge_K = self.W_Kr(edge_attr).view(
                -1, self.heads, self.out_channels
            )  # [E, H, C]
            edge_V = self.W_Vr(edge_attr).view(
                -1, self.heads, self.out_channels
            )  # [E, H, C]

            # 融入关系信息
            K_enhanced = K_j + edge_K  # [E, H, C]
            V_enhanced = V_j + edge_V  # [E, H, C]

        # 计算注意力分数 t_ij = Q_i^T * K_enhanced / √d_k
        attn_scores = torch.sum(Q_i * K_enhanced, dim=-1) * self.scale  # [E, H]

        # Softmax归一化
        attn_weights = softmax(attn_scores, index, ptr, size_i)  # [E, H]
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 应用注意力权重到增强的Value
        out = V_enhanced * attn_weights.unsqueeze(-1)  # [E, H, C]

        return out


class HybridGraphAttention(nn.Module):
    """
    混合图注意力网络 (Hybrid GAT)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        lambda_init: float = 0.8,
        add_self_loops: bool = True,
        bias: bool = True,
        use_layer_norm: bool = True,
        **kwargs,
    ):
        """
        Args:
            in_channels (int): 输入特征维度
            out_channels (int): 输出特征维度
            edge_dim (int, optional): 边特征维度
            heads (int): 多头注意力数量
            concat (bool): 是否拼接多头输出
            dropout (float): Dropout概率
            lambda_init (float): 差分注意力的lambda初始值
            add_self_loops (bool): 是否添加自环
            bias (bool): 是否使用偏置
            use_layer_norm (bool): 是否使用层归一化
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.use_layer_norm = use_layer_norm

        # 策略1: 关系聚合 - 使用差分图注意力
        self.relation_aggregation = DifferentialGraphAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            heads=heads,
            concat=concat,
            dropout=dropout,
            lambda_init=lambda_init,
            add_self_loops=add_self_loops,
            bias=False,  # 最终输出层会处理bias
            **kwargs,
        )

        # 策略2: 关系激活 - 使用QKV注意力
        self.relation_activation = RelationActivationAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            heads=heads,
            concat=concat,
            dropout=dropout,
            add_self_loops=add_self_loops,
            bias=False,  # 最终输出层会处理bias
            **kwargs,
        )

        # 确定拼接后的维度
        if concat:
            single_output_dim = heads * out_channels
        else:
            single_output_dim = out_channels

        hybrid_dim = single_output_dim * 2  # 两种策略拼接

        # 输入投影层（用于残差连接） - 确保维度匹配
        self.input_proj = nn.Linear(in_channels, single_output_dim)

        # 输出投影层
        self.output_proj = nn.Linear(hybrid_dim, single_output_dim, bias=bias)

        # 层归一化
        if use_layer_norm:
            self.layer_norm = RMSNorm(single_output_dim)
        else:
            self.layer_norm = None

    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播

        Args:
            x (Tensor): 节点特征矩阵 [N, in_channels]
            edge_index (LongTensor): 边索引 [2, M]
            edge_attr (Tensor, optional): 边特征矩阵 [M, edge_dim]

        Returns:
            Tensor: 输出节点嵌入
        """
        # 保存原始输入用于残差连接
        residual = x

        # 策略1: 关系聚合
        h_aggr = self.relation_aggregation(
            x, edge_index, edge_attr
        )  # [N, single_output_dim]

        # 策略2: 关系激活
        h_actv = self.relation_activation(
            x, edge_index, edge_attr
        )  # [N, single_output_dim]

        # 拼接两种策略的输出 (公式7)
        h_hybrid = torch.cat([h_aggr, h_actv], dim=-1)  # [N, hybrid_dim]

        # 输出投影
        h_out = self.output_proj(h_hybrid)  # [N, single_output_dim]

        # 残差连接 (公式8) - 现在维度总是匹配的
        residual = self.input_proj(residual)
        h_out = h_out + residual

        # 层归一化
        if self.layer_norm is not None:
            h_out = self.layer_norm(h_out)

        return h_out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads}, "
            f"edge_dim={getattr(self.relation_aggregation, 'edge_dim', None)})"
        )


class HybridGraphTransformerLayer(nn.Module):
    """
    完整的混合图Transformer层
    包含混合图注意力和前馈网络
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        lambda_init: float = 0.8,
        ff_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        **kwargs,
    ):
        """
        Args:
            in_channels (int): 输入特征维度
            out_channels (int): 输出特征维度
            edge_dim (int, optional): 边特征维度
            heads (int): 多头注意力数量
            lambda_init (float): 差分注意力的lambda初始值
            ff_hidden_dim (int, optional): 前馈网络隐藏维度
            dropout (float): Dropout概率
            use_layer_norm (bool): 是否使用层归一化
        """
        super().__init__()

        # 第一层归一化
        if use_layer_norm:
            self.norm1 = RMSNorm(in_channels)
        else:
            self.norm1 = None

        # 混合图注意力
        self.hybrid_attention = HybridGraphAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            heads=heads,
            lambda_init=lambda_init,
            dropout=dropout,
            use_layer_norm=False,  
            **kwargs,
        )

        # 确定注意力输出维度
        if kwargs.get("concat", True):
            attn_out_dim = heads * out_channels
        else:
            attn_out_dim = out_channels

        # 第二层归一化
        if use_layer_norm:
            self.norm2 = RMSNorm(attn_out_dim)
        else:
            self.norm2 = None

        # 前馈网络
        if ff_hidden_dim is None:
            ff_hidden_dim = attn_out_dim * 4

        self.ff = nn.Sequential(
            nn.Linear(attn_out_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, attn_out_dim),
            nn.Dropout(dropout),
        )

        # 输入投影层（如果需要）
        if in_channels != attn_out_dim:
            self.input_proj = nn.Linear(in_channels, attn_out_dim)
        else:
            self.input_proj = None

    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播

        Args:
            x (Tensor): 节点特征矩阵 [N, in_channels]
            edge_index (LongTensor): 边索引 [2, M]
            edge_attr (Tensor, optional): 边特征矩阵 [M, edge_dim]

        Returns:
            Tensor: 输出节点嵌入
        """
        # 保存原始输入
        residual = x

        # 第一个子层: 混合图注意力
        if self.norm1 is not None:
            x = self.norm1(x)

        attn_out = self.hybrid_attention(x, edge_index, edge_attr)

        # 处理维度不匹配
        if self.input_proj is not None:
            residual = self.input_proj(residual)

        # 第一个残差连接
        x = attn_out + residual

        # 第二个子层: 前馈网络
        residual = x
        if self.norm2 is not None:
            x = self.norm2(x)

        ff_out = self.ff(x)

        # 第二个残差连接
        x = ff_out + residual

        return x


# 测试代码
if __name__ == "__main__":
    try:
        import torch_geometric

        print("Testing Hybrid Graph Attention Network...")

        # 图参数
        num_nodes = 50
        num_edges = 100
        node_features = 64
        edge_features = 16
        output_features = 32
        heads = 4

        # 创建随机图数据
        x = torch.randn(num_nodes, node_features)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, edge_features)

        print(f"Input: nodes={num_nodes}, edges={num_edges}")
        print(f"Node features: {node_features}, Edge features: {edge_features}")

        # 测试关系激活注意力
        print("\n1. Testing RelationActivationAttention:")
        rel_actv = RelationActivationAttention(
            in_channels=node_features,
            out_channels=output_features,
            edge_dim=edge_features,
            heads=heads,
            concat=True,
        )

        output = rel_actv(x, edge_index, edge_attr)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected: ({num_nodes}, {heads * output_features})")

        # 测试混合图注意力
        print("\n2. Testing HybridGraphAttention:")
        hybrid_gat = HybridGraphAttention(
            in_channels=node_features,
            out_channels=output_features,
            edge_dim=edge_features,
            heads=heads,
            concat=True,
            lambda_init=0.8,
        )

        output = hybrid_gat(x, edge_index, edge_attr)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")

        # 测试完整的Transformer层
        print("\n3. Testing HybridGraphTransformerLayer:")
        hybrid_transformer = HybridGraphTransformerLayer(
            in_channels=node_features,
            out_channels=output_features,
            edge_dim=edge_features,
            heads=heads,
            lambda_init=0.8,
        )

        output = hybrid_transformer(x, edge_index, edge_attr)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")

        # 测试无边属性的情况
        print("\n4. Testing without edge attributes:")
        hybrid_gat_no_edge = HybridGraphAttention(
            in_channels=node_features,
            out_channels=output_features,
            heads=heads,
            concat=False,
        )

        output = hybrid_gat_no_edge(x, edge_index)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected: ({num_nodes}, {output_features})")

        print("\n✅ All tests passed successfully!")

    except ImportError as e:
        print(f"PyTorch Geometric not installed: {e}")
        print("To test this module, please install PyTorch Geometric:")
        print("pip install torch_geometric")
