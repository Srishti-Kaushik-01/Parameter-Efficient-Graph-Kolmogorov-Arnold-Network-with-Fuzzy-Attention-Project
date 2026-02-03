"""
Parameter-Efficient HSI Classification with GKAN and Fuzzy Attention
Architecture: CNN Encoder -> Tokenization -> Spatial Graph -> GKAN -> Fuzzy Attention -> Global Pool -> MLP Classifier
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math



try:
    from models.registry import register_model
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    def register_model(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator




# ==================== MLP Classifier ====================



class MLPClassifier(nn.Module):
    """
    Standard MLP classifier with ReLU activations.
    """
    def __init__(self, in_features: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim


        # Layer 1: input -> hidden
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)


        # Layer 2: hidden -> output
        self.fc2 = nn.Linear(hidden_dim, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features]
        Returns:
            [B, num_classes]
        """
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)


        # Layer 2
        x = self.fc2(x)


        return x




# ==================== GKAN Layer ====================



class GKANLayer(nn.Module):
    """
    Graph Kolmogorov-Arnold Network layer.
    Combines spatial graph convolution with learnable KAN activation.
    """
    def __init__(self, in_features: int, out_features: int, num_neighbors: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_neighbors = num_neighbors


        # Graph convolution weights
        self.weight_self = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_neighbor = nn.Parameter(torch.FloatTensor(in_features, out_features))


        # KAN-style learnable activation
        self.kan_a = nn.Parameter(torch.randn(out_features, 3) * 0.1)
        self.kan_b = nn.Parameter(torch.randn(out_features, 2) * 0.1)


        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.bn = nn.BatchNorm1d(out_features)


        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_self)
        nn.init.xavier_uniform_(self.weight_neighbor)
        nn.init.zeros_(self.bias)


    def kan_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Learnable KAN activation using rational function.
        """
        x2 = x ** 2
        numerator = self.kan_a[:, 0] + self.kan_a[:, 1] * x + self.kan_a[:, 2] * x2
        denominator = 1.0 + torch.abs(self.kan_b[:, 0] * x + self.kan_b[:, 1] * x2)
        return numerator / (denominator + 1e-8)


    def build_knn_graph(self, x: torch.Tensor) -> torch.Tensor:
        """Build k-NN spatial adjacency matrix."""
        B, N, D = x.shape
        device = x.device


        # Compute pairwise distances
        xx = torch.sum(x ** 2, dim=2, keepdim=True)
        xy = torch.matmul(x, x.transpose(1, 2))
        dist = xx - 2 * xy + xx.transpose(1, 2)
        dist = torch.clamp(dist, min=0.0)


        # Get k nearest neighbors
        k = min(self.num_neighbors + 1, N)
        _, indices = torch.topk(dist, k, dim=2, largest=False)


        # Build adjacency matrix
        adj = torch.zeros(B, N, N, device=device)
        for b in range(B):
            for i in range(N):
                neighbors = indices[b, i, 1:]  # Exclude self
                adj[b, i, neighbors] = 1.0


        # Normalize: D^{-1/2} A D^{-1/2}
        degree = torch.sum(adj, dim=2) + 1e-6
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_mat = torch.diag_embed(degree_inv_sqrt)
        adj_norm = torch.matmul(torch.matmul(degree_mat, adj), degree_mat)


        # Add self-loops
        identity = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        adj_norm = adj_norm + identity


        return adj_norm


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, in_features] node features
        Returns:
            [B, N, out_features] output features
        """
        B, N, D = x.shape


        # Build spatial graph
        adj = self.build_knn_graph(x)  # [B, N, N]


        # Self transformation
        h_self = torch.matmul(x, self.weight_self)  # [B, N, out_features]


        # Neighbor aggregation
        h_neighbor = torch.matmul(x, self.weight_neighbor)  # [B, N, out_features]
        h_neighbor = torch.matmul(adj, h_neighbor)  # [B, N, out_features]


        # Combine
        h = h_self + h_neighbor + self.bias  # [B, N, out_features]


        # Apply KAN activation
        h = self.kan_activation(h)


        # Batch normalization
        h = h.transpose(1, 2)  # [B, out_features, N]
        h = self.bn(h)
        h = h.transpose(1, 2)  # [B, N, out_features]


        return h




# ==================== Fuzzy Attention ====================



class FuzzyAttention(nn.Module):
    """
    Fuzzy attention mechanism for spatial feature refinement.
    Uses learnable RBF kernel membership functions.
    """
    def __init__(self, dim: int, num_fuzzy: int = 9):
        super().__init__()
        self.dim = dim
        self.num_fuzzy = num_fuzzy


        # Learnable RBF parameters
        self.centers = nn.Parameter(torch.randn(dim, num_fuzzy))
        self.gamma = nn.Parameter(torch.ones(dim, num_fuzzy) * 0.5)


        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )


        self.bn = nn.BatchNorm1d(dim)


    def rbf_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF kernel membership values.
        K(x, c) = exp(-gamma * ||x - c||^2)
        
        """
        x_expanded = x.unsqueeze(-1)  # [B, N, D, 1]
        centers_expanded = self.centers.view(1, 1, self.dim, self.num_fuzzy)
        gamma_expanded = self.gamma.view(1, 1, self.dim, self.num_fuzzy)


        # RBF kernel
        squared_dist = (x_expanded - centers_expanded) ** 2
        rbf_values = torch.exp(-gamma_expanded.abs() * squared_dist)
        
        # Aggregate across fuzzy sets
        rbf_output = torch.logsumexp(torch.log(rbf_values + 1e-10), dim=-1)
        rbf_output = torch.exp(rbf_output)


        return rbf_output


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input features
        Returns:
            [B, N, D] attended features
        """
        B, N, D = x.shape


        # Apply RBF kernel membership
        x_rbf = self.rbf_kernel(x)


        # Compute attention weights
        attn_weights = self.attention(x_rbf)


        # Apply attention
        x_attended = x * attn_weights + x  # Residual connection


        # Batch normalization
        x_attended = x_attended.transpose(1, 2)  # [B, D, N]
        x_attended = self.bn(x_attended)
        x_attended = x_attended.transpose(1, 2)  # [B, N, D]


        return x_attended




# ==================== CNN Encoder ====================



class CNNEncoder(nn.Module):


    def __init__(self, in_channels: int, embed_dim: int = 64):
        super().__init__()


        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),


            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),


            # Layer 3
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, embed_dim, H, W]
        """
        return self.encoder(x)




# ==================== Main Model ====================



class FGKAN(nn.Module):
    """
    Parameter-Efficient HSI Classification with GKAN and Fuzzy Attention.


    Architecture:
    1. Input single modality
    2. CNN encoder
    3. Tokenization
    4. Spatial Graph Construction
    5. GKAN Layer
    6. Fuzzy Attention (with RBF kernel)
    7. Global pooling
    8. MLP classifier
    """
    def __init__(self,
                 in_channels: int = 200,
                 patch_size: int = 11,
                 num_classes: int = 16,
                 embed_dim: int = 64,
                 num_neighbors: int = 8,
                 num_fuzzy: int = 9,
                 dropout: float = 0.1):
        super().__init__()


        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes


        # 1. CNN Encoder
        self.cnn_encoder = CNNEncoder(in_channels, embed_dim)


        # 2. Tokenization (flatten spatial dimensions)
        # No explicit layer needed - handled in forward


        # 3. GKAN Layer with spatial graph construction
        self.gkan = GKANLayer(embed_dim, embed_dim, num_neighbors=num_neighbors)


        # 4. Fuzzy Attention with RBF kernel
        self.fuzzy_attention = FuzzyAttention(embed_dim, num_fuzzy=num_fuzzy)


        # 5. Dropout
        self.dropout = nn.Dropout(dropout)


        # 6. MLP Classifier
        self.classifier = MLPClassifier(embed_dim, num_classes, hidden_dim=embed_dim * 2)




    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert spatial feature maps to tokens.
        Args:
            x: [B, D, H, W]
        Returns:
            [B, N, D] where N = H*W
        """
        B, D, H, W = x.shape
        x = x.flatten(2)  # [B, D, H*W]
        x = x.transpose(1, 2)  # [B, H*W, D]
        return x


    def global_pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Global pooling over spatial tokens.
        Args:
            x: [B, N, D]
        Returns:
            [B, D]
        """
        # Use mean and max pooling
        x_mean = torch.mean(x, dim=1)  # [B, D]
        x_max, _ = torch.max(x, dim=1)  # [B, D]


        # Combine with learnable weights
        return 0.5 * x_mean + 0.5 * x_max


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.


        Args:
            x: [B, C, H, W] input HSI patch
        Returns:
            [B, num_classes] logits
        """
        # Handle 5D input (B, 1, C, H, W) -> (B, C, H, W)
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)


        # 1. CNN Encoder
        x = self.cnn_encoder(x)  # [B, embed_dim, H, W]


        # 2. Tokenization
        x = self.tokenize(x)  # [B, N, embed_dim]


        # 3. GKAN Layer (includes spatial graph construction)
        x = self.gkan(x)  # [B, N, embed_dim]


        # 4. Fuzzy Attention with RBF kernel
        x = self.fuzzy_attention(x)  # [B, N, embed_dim]


        # 5. Dropout
        x = self.dropout(x)


        # 6. Global Pooling
        x = self.global_pool(x)  # [B, embed_dim]


        # 7. MLP Classifier
        logits = self.classifier(x)  # [B, num_classes]


        return logits


    def get_param_count(self):
        """Return parameter count."""
        return sum(p.numel() for p in self.parameters())




# ==================== Model Factory ====================



@register_model('FGKAN', expects_4d=True, feature_dim=64, encoder_depth=1)
def FGKAN(pretrained: bool = False, **kwargs) -> FGKAN:
    """Constructs a parameter-efficient FGKAN model with MLP classifier."""
    
    # Remove registry metadata
    for key in ['expects_4d', 'dual_input', 'feature_dim', 'encoder_depth']:
        kwargs.pop(key, None)
    
    # Map standardized parameter names
    if 'bands' in kwargs:
        kwargs['in_channels'] = kwargs.pop('bands')
    
    defaults = dict(
        in_channels=200,
        patch_size=11,
        num_classes=16,
        embed_dim=64,
        num_neighbors=8,
        num_fuzzy=9,
        dropout=0.1
    )
    
    config = {**defaults, **kwargs}
    return FGKAN(**config)





# ==================== Testing ====================



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print("=" * 70)
    print("Testing Parameter-Efficient FGKAN with MLP Classifier (RBF Kernel)")
    print("=" * 70)


    test_configs = [
        (8, 200, 11, 16),
        (4, 48, 9, 20),
        (2, 145, 11, 14),
        (16, 103, 7, 9),
    ]


    for batch, channels, spatial, classes in test_configs:
        print(f"\nTest: B={batch}, C={channels}, S={spatial}, Classes={classes}")


        try:
            model = FGKAN(
                in_channels=channels,
                patch_size=spatial,
                num_classes=classes,
                embed_dim=64
            ).to(device)


            x = torch.randn(batch, channels, spatial, spatial).to(device)


            with torch.no_grad():
                logits = model(x)


            params = model.get_param_count()


            print(f"  ✓ Input: {x.shape}")
            print(f"  ✓ Output: {logits.shape}")
            print(f"  ✓ Params: {params:,}")
            print(f"  ✓ Param efficiency: {params / 1e6:.2f}M")


        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()


    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)

