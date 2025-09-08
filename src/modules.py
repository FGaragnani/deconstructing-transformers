import torch
import torch.nn as nn
from src.layers import MultiHeadAttentionLayer, FeedForwardLayer
from typing import Optional

class EncoderModule(nn.Module):
  def __init__(self, embed_size: int, head_num: int, hidden_ff_size: Optional[int], dropout: float = 0.0):
    super(EncoderModule, self).__init__()
    self.embed_size = embed_size
    self.head_num = head_num
    self.hidden_ff_size = hidden_ff_size
    self.dropout = nn.Dropout(dropout)

    self.mha = MultiHeadAttentionLayer(embed_size, head_num)
    self.norm1 = nn.LayerNorm(embed_size)
    self.ff = FeedForwardLayer(embed_size, hidden_ff_size)
    self.norm2 = nn.LayerNorm(embed_size)

  def forward(self, X: torch.Tensor):
    A = self.mha((X, X, X))               # multihead attention, concatenation, output projection
    X = self.norm1(X + self.dropout(A))   # add and LayerNorm
    F = self.ff(X)                        # feed-forward
    X = self.norm2(X + self.dropout(F))   # add and LayerNorm
    return X
  
class DecoderModule(nn.Module):
  def __init__(self, embed_size: int, head_num_1: int, head_num_2: int, hidden_ff_size: Optional[int], dropout: float = 0.0):
    super(DecoderModule, self).__init__()
    self.embed_size = embed_size
    self.head_num_1 = head_num_1
    self.head_num_2 = head_num_2
    self.dropout = nn.Dropout(dropout)

    self.mha_1 = MultiHeadAttentionLayer(embed_size, head_num_1, mask=True)
    self.norm1 = nn.LayerNorm(embed_size)
    self.mha_2 = MultiHeadAttentionLayer(embed_size, head_num_2, mask=False)
    self.norm2 = nn.LayerNorm(embed_size)
    self.ff = FeedForwardLayer(embed_size, hidden_ff_size)
    self.norm3 = nn.LayerNorm(embed_size)

  def forward(self, input: tuple[torch.Tensor, torch.Tensor]):
    X, Z = input
    A = self.mha_1((X, X, X))                 # multihead masked self-attention, concatenation, output projection
    X = self.norm1(X + self.dropout(A))       # add and LayerNorm
    A = self.mha_2((X, Z, Z))                 # multihead cross-attention, concatenation, output projection
    X = self.norm2(X + self.dropout(A))       # add and LayerNorm
    F = self.ff(X)                            # feed-forward
    X = self.norm3(X + self.dropout(F))       # add and LayerNorm
    return (X, Z)
  
class Output(nn.Module):
  def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
    super().__init__()
    hidden_dim = hidden_dim or embed_dim

    self.head = nn.Sequential(
      nn.Linear(embed_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, embed_dim),
    )

    self.scale_net = nn.Linear(embed_dim, embed_dim)
    self.bias_net = nn.Linear(embed_dim, embed_dim)

  def forward(self, X: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """
      Predict scale/bias from context.
    """
    head_in = X[:, -1, :]
    out = self.head(head_in)

    s = torch.sigmoid(self.scale_net(context))
    b = self.bias_net(context)
    return out * s + b                          # feedforward(Yj) * scale + bias