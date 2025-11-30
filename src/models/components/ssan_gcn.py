import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """单层图卷积网络"""

    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, features, adj):
        """
        Args:
            features: 节点特征 [batch_size, seq_len, in_dim]
            adj: 邻接矩阵 [batch_size, seq_len, seq_len]
        """
        # A * H * W
        support = self.linear(features)  # H * W
        output = torch.bmm(adj, support)  # A * (H * W)
        return output


class GCN(nn.Module):
    """多层GCN模型 - 增强正则化版本"""

    def __init__(self, in_dim, hidden_dim, num_layers, dropout=0.2):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()  # 添加层归一化

        for i in range(num_layers):
            # 第一层: in_dim -> hidden_dim
            # 后续层: hidden_dim -> hidden_dim
            input_dim = in_dim if i == 0 else hidden_dim
            self.layers.append(GCNLayer(input_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_dropout = nn.Dropout(dropout * 0.5)  # 输入dropout，稍微弱一些

    def forward(self, features, adj):
        x = self.input_dropout(features)  # 在输入时就加dropout

        for i, (layer, layer_norm) in enumerate(zip(self.layers, self.layer_norms)):
            # 残差连接（如果维度匹配）
            if i > 0 and x.shape[-1] == features.shape[-1]:
                residual = x
            else:
                residual = None

            x = layer(x, adj)
            x = layer_norm(x)  # 层归一化
            x = self.activation(x)

            # 残差连接
            if residual is not None:
                x = x + residual * 0.3  # 弱化残差连接

            x = self.dropout(x)

        return x
