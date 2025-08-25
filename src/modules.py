import torch
import torch.nn as nn
from src.layers import MultiHeadAttentionLayer, FeedForwardLayer
from typing import Optional

class EncoderModule(nn.Module):
  def __init__(self, embed_size: int, head_num: int, hidden_ff_size: Optional[int]):
    super(EncoderModule, self).__init__()
    self.embed_size = embed_size
    self.head_num = head_num
    self.hidden_ff_size = hidden_ff_size

    self.mha = MultiHeadAttentionLayer(embed_size, head_num)
    self.norm1 = nn.LayerNorm(embed_size)
    self.ff = FeedForwardLayer(embed_size, hidden_ff_size)
    self.norm2 = nn.LayerNorm(embed_size)

  def forward(self, X: torch.Tensor):
    X = self.norm1(X + self.mha((X, X, X)))
    X = self.norm2(X + self.ff(X))
    return X
  
class DecoderModule(nn.Module):
  def __init__(self, embed_size: int, head_num_1: int, head_num_2: int, hidden_ff_size: Optional[int]):
    super(DecoderModule, self).__init__()
    self.embed_size = embed_size
    self.head_num_1 = head_num_1
    self.head_num_2 = head_num_2

    self.mha_1 = MultiHeadAttentionLayer(embed_size, head_num_1, mask=True)
    self.norm1 = nn.LayerNorm(embed_size)
    self.mha_2 = MultiHeadAttentionLayer(embed_size, head_num_2, mask=False)
    self.norm2 = nn.LayerNorm(embed_size)
    self.ff = FeedForwardLayer(embed_size, hidden_ff_size)
    self.norm3 = nn.LayerNorm(embed_size)

  def forward(self, input: tuple[torch.Tensor, torch.Tensor]):
    X, Z = input
    X = self.norm1(X + self.mha_1((X, X, X)))
    X = self.norm2(X + self.mha_2((X, Z, Z)))
    X = self.norm3(X + self.ff(X))
    return (X, Z)
  
class Output(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim

        self.head = nn.Sequential(
           nn.Linear(embed_dim, embed_dim),
           nn.LayerNorm(embed_dim)
        )

        """
        Old Version
        -----------
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.head(X[:, -1, :])
        return X