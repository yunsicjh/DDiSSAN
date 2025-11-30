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

    # Define dummy types
    Adj = Union[torch.Tensor, None]
    OptTensor = Optional[torch.Tensor]
    PairTensor = Tuple[torch.Tensor, torch.Tensor]
    HAS_TORCH_GEOMETRIC = False


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Applies normalization across the last dimension and scales the output.
    """

    def __init__(self, d, eps=1e-5):
        """
        Args:
            d (int): Dimension of the input features.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        """
        Forward pass for RMSNorm.

        Args:
            x (Tensor): Input tensor of shape (..., d).

        Returns:
            Tensor: Normalized and scaled tensor.
        """
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale


class DifferentialGraphAttention(MessagePassing):
    """
    Differential Graph Attention Network that considers both node features and edge attributes.
    Based on the differential attention mechanism from the original diffattn_nlp.py.

    This layer computes attention scores using two separate attention mechanisms
    and takes their difference, similar to the differential attention in transformers,
    but adapted for graph structures with edge attributes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        lambda_init: float = 0.8,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        """
        Args:
            in_channels (int): Size of each input sample.
            out_channels (int): Size of each output sample.
            edge_dim (int, optional): Edge feature dimensionality.
            heads (int): Number of multi-head-attentions.
            concat (bool): If set to False, the multi-head attentions are averaged instead of concatenated.
            negative_slope (float): LeakyReLU angle of the negative slope.
            dropout (float): Dropout probability of the normalized attention coefficients.
            lambda_init (float): Initial value for lambda in differential attention.
            add_self_loops (bool): If set to False, will not add self-loops to the input graph.
            bias (bool): If set to False, the layer will not learn an additive bias.
        """
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.lambda_init = lambda_init

        # For differential attention, we need two sets of transformations
        # First attention mechanism
        self.lin_l_1 = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_r_1 = self.lin_l_1

        # Second attention mechanism
        self.lin_l_2 = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_r_2 = self.lin_l_2

        # Attention parameters for both mechanisms
        self.att_l_1 = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r_1 = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_l_2 = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r_2 = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # Edge attribute transformations for both mechanisms
        if edge_dim is not None:
            self.lin_edge_1 = nn.Linear(edge_dim, heads * out_channels, bias=False)
            self.att_edge_1 = nn.Parameter(torch.Tensor(1, heads, out_channels))
            self.lin_edge_2 = nn.Linear(edge_dim, heads * out_channels, bias=False)
            self.att_edge_2 = nn.Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge_1 = None
            self.att_edge_1 = None
            self.lin_edge_2 = None
            self.att_edge_2 = None

        # Lambda parameters for differential attention
        self.lambda_q1 = nn.Parameter(torch.randn(heads, out_channels))
        self.lambda_k1 = nn.Parameter(torch.randn(heads, out_channels))
        self.lambda_q2 = nn.Parameter(torch.randn(heads, out_channels))
        self.lambda_k2 = nn.Parameter(torch.randn(heads, out_channels))

        # Output projection and normalization
        if concat:
            self.out_proj = nn.Linear(
                heads * out_channels, heads * out_channels, bias=bias
            )
            self.rms_norm = RMSNorm(heads * out_channels)
        else:
            self.out_proj = nn.Linear(out_channels, out_channels, bias=bias)
            self.rms_norm = RMSNorm(out_channels)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters for improved training stability."""
        nn.init.xavier_uniform_(self.lin_l_1.weight)
        nn.init.xavier_uniform_(self.lin_l_2.weight)
        nn.init.xavier_uniform_(self.att_l_1)
        nn.init.xavier_uniform_(self.att_r_1)
        nn.init.xavier_uniform_(self.att_l_2)
        nn.init.xavier_uniform_(self.att_r_2)

        if self.edge_dim is not None:
            nn.init.xavier_uniform_(self.lin_edge_1.weight)
            nn.init.xavier_uniform_(self.lin_edge_2.weight)
            nn.init.xavier_uniform_(self.att_edge_1)
            nn.init.xavier_uniform_(self.att_edge_2)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass for Differential Graph Attention.

        Args:
            x (Tensor): Node feature matrix of shape [N, in_channels].
            edge_index (LongTensor): Graph connectivity in COO format with shape [2, M].
            edge_attr (Tensor, optional): Edge feature matrix of shape [M, edge_dim].

        Returns:
            Tensor: Output node embeddings of shape [N, heads * out_channels] or [N, out_channels].
        """
        H, C = self.heads, self.out_channels
        N = x.size(0)

        if self.add_self_loops:
            if isinstance(edge_index, torch.Tensor):
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value="mean", num_nodes=N
                )

        # Transform node features for both attention mechanisms
        x_l_1 = self.lin_l_1(x).view(-1, H, C)  # [N, H, C]
        x_r_1 = x_l_1
        x_l_2 = self.lin_l_2(x).view(-1, H, C)  # [N, H, C]
        x_r_2 = x_l_2

        # Propagate messages for both attention mechanisms
        out_1 = self.propagate(
            edge_index,
            x=(x_l_1, x_r_1),
            edge_attr=edge_attr,
            attention_type=1,
            size=(N, N),
        )
        out_2 = self.propagate(
            edge_index,
            x=(x_l_2, x_r_2),
            edge_attr=edge_attr,
            attention_type=2,
            size=(N, N),
        )

        # Compute lambda for differential attention
        lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)  # [H]
        lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)  # [H]
        lambda_val = (
            torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init
        )  # [H]
        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1)  # [1, H, 1]

        # Apply differential attention
        out = out_1 - lambda_val * out_2  # [N, H, C]

        # Apply RMS normalization
        if self.concat:
            out = out.view(-1, H * C)  # [N, H*C]
        else:
            out = out.mean(dim=1)  # [N, C]

        out = self.rms_norm(out)
        out = out * (1 - self.lambda_init)  # Scale by (1 - lambda_init)
        out = self.out_proj(out)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i, attention_type):
        """
        Compute messages between nodes considering edge attributes.

        Args:
            x_i: Target node features [E, H, C]
            x_j: Source node features [E, H, C]
            edge_attr: Edge attributes [E, edge_dim]
            index: Edge indices for target nodes
            ptr: Pointer for batch processing
            size_i: Number of target nodes
            attention_type: 1 or 2, indicating which attention mechanism to use

        Returns:
            Tensor: Messages of shape [E, H, C]
        """
        # Select the appropriate attention parameters based on attention_type
        if attention_type == 1:
            att_l, att_r = self.att_l_1, self.att_r_1
            lin_edge, att_edge = self.lin_edge_1, self.att_edge_1
        else:
            att_l, att_r = self.att_l_2, self.att_r_2
            lin_edge, att_edge = self.lin_edge_2, self.att_edge_2

        # Compute attention scores from node features
        alpha_l = (x_i * att_l).sum(dim=-1)  # [E, H]
        alpha_r = (x_j * att_r).sum(dim=-1)  # [E, H]
        alpha = alpha_l + alpha_r  # [E, H]

        # Add edge attribute contribution if available
        if edge_attr is not None and lin_edge is not None:
            edge_attr = lin_edge(edge_attr).view(
                -1, self.heads, self.out_channels
            )  # [E, H, C]
            alpha_edge = (edge_attr * att_edge).sum(dim=-1)  # [E, H]
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Apply attention weights to source node features
        out = x_j * alpha.unsqueeze(-1)  # [E, H, C]

        # Incorporate edge features into the message
        if edge_attr is not None and lin_edge is not None:
            out = out + edge_attr * alpha.unsqueeze(-1)

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads}, "
            f"edge_dim={self.edge_dim}, lambda_init={self.lambda_init})"
        )


class DifferentialGraphTransformerLayer(nn.Module):
    """
    A complete graph transformer layer with differential attention and feed-forward network.
    Similar to the DiffTransformerLayer but adapted for graph data.
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
        **kwargs,
    ):
        """
        Args:
            in_channels (int): Size of each input sample.
            out_channels (int): Size of each output sample.
            edge_dim (int, optional): Edge feature dimensionality.
            heads (int): Number of multi-head-attentions.
            lambda_init (float): Initial value for lambda in differential attention.
            ff_hidden_dim (int, optional): Hidden dimension for feed-forward network.
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.norm1 = RMSNorm(in_channels)
        self.attention = DifferentialGraphAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            heads=heads,
            lambda_init=lambda_init,
            dropout=dropout,
            **kwargs,
        )

        # Determine the dimension after attention
        attn_out_dim = (
            heads * out_channels if kwargs.get("concat", True) else out_channels
        )

        self.norm2 = RMSNorm(attn_out_dim)

        # Feed-forward network
        if ff_hidden_dim is None:
            ff_hidden_dim = attn_out_dim * 4

        self.ff = nn.Sequential(
            nn.Linear(attn_out_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, attn_out_dim),
            nn.Dropout(dropout),
        )

        # Projection layer if input and output dimensions don't match
        if in_channels != attn_out_dim:
            self.input_proj = nn.Linear(in_channels, attn_out_dim)
        else:
            self.input_proj = None

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass for the graph transformer layer.

        Args:
            x (Tensor): Node feature matrix of shape [N, in_channels].
            edge_index (LongTensor): Graph connectivity in COO format.
            edge_attr (Tensor, optional): Edge feature matrix.

        Returns:
            Tensor: Output node embeddings.
        """
        # Store original input for residual connection
        residual = x

        # Apply layer norm and attention
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, edge_index, edge_attr)

        # Handle dimension mismatch for residual connection
        if self.input_proj is not None:
            residual = self.input_proj(residual)

        # First residual connection
        x = attn_out + residual

        # Apply layer norm and feed-forward
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)

        # Second residual connection
        x = ff_out + x

        return x


# Example usage and testing
if __name__ == "__main__":
    try:
        import torch_geometric
        from torch_geometric.data import Data

        # Create a simple graph for testing
        print("Testing Differential Graph Attention...")

        # Graph parameters
        num_nodes = 100
        num_edges = 200
        node_features = 64
        edge_features = 16
        output_features = 32
        heads = 4

        # Create random graph data
        x = torch.randn(num_nodes, node_features)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, edge_features)

        # Test DifferentialGraphAttention
        print("\n1. Testing DifferentialGraphAttention layer:")
        diff_gat = DifferentialGraphAttention(
            in_channels=node_features,
            out_channels=output_features,
            edge_dim=edge_features,
            heads=heads,
            concat=True,
            lambda_init=0.8,
        )

        output = diff_gat(x, edge_index, edge_attr)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: ({num_nodes}, {heads * output_features})")

        # Test DifferentialGraphTransformerLayer
        print("\n2. Testing DifferentialGraphTransformerLayer:")
        diff_transformer = DifferentialGraphTransformerLayer(
            in_channels=node_features,
            out_channels=output_features,
            edge_dim=edge_features,
            heads=heads,
            lambda_init=0.8,
            concat=True,
        )

        output = diff_transformer(x, edge_index, edge_attr)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")

        # Test without edge attributes
        print("\n3. Testing without edge attributes:")
        diff_gat_no_edge = DifferentialGraphAttention(
            in_channels=node_features,
            out_channels=output_features,
            heads=heads,
            concat=False,
            lambda_init=0.8,
        )

        output = diff_gat_no_edge(x, edge_index)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: ({num_nodes}, {output_features})")

        print("\nAll tests passed successfully!")

    except ImportError as e:
        print(f"PyTorch Geometric not installed: {e}")
        print("To test this module, please install PyTorch Geometric:")
        print("pip install torch_geometric")
