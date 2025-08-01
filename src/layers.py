import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional

"""
Multi-head attention layer
"""
class MultiHeadAttentionLayer(nn.Module):
  def __init__(self, embed_size: int, num_heads: int, mask: bool = False):
        super(MultiHeadAttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.mask = mask

        # Check if the input is valid
        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by num_heads"
        assert self.embed_size > 0, "Embedding size must be greater than 0"
        assert self.num_heads > 0, "The number of heads must be greater than 0"

        self.W_Q = nn.Linear(embed_size, embed_size)
        self.W_K = nn.Linear(embed_size, embed_size)
        self.W_V = nn.Linear(embed_size, embed_size)
        self.W_O = nn.Linear(embed_size, embed_size)

  def forward(self, Q_K_V: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        Q, K, V = Q_K_V
        batch_size, seq_length = Q.shape[0:2]

        # Transpose the input in every space
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        # Shape: (batch_size, seq_length, num_heads, embed_head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Apply the attention function
        A = torch.matmul(Q, K.transpose(-2, -1))
        A /= (self.head_dim ** 0.5)
        if self.mask:
          q_seq_len, k_seq_len = Q.size(-2), K.size(-2)
          mask = torch.tril(torch.ones(q_seq_len, k_seq_len)).to(A.device)
          A = A.masked_fill(mask == 0, float('-inf'))
        A = F.softmax(A, dim=-1)
        A = torch.matmul(A, V)

        # Concatenating heads
        A = A.transpose(1, 2).contiguous().reshape(batch_size, -1, self.embed_size)

        return self.W_O(A)


"""
  Feed-forward layer
"""
class FeedForwardLayer(nn.Module):
  def __init__(self, embed_size: int, inner_size: Optional[int]):
    super(FeedForwardLayer, self).__init__()
    self.embed_size = embed_size
    self.inner_size = inner_size if inner_size is not None else embed_size

    self.fc1 = nn.Linear(in_features=self.embed_size, out_features=self.inner_size, bias=True)
    self.fc2 = nn.Linear(in_features=self.inner_size, out_features=self.embed_size, bias=True)

  def forward(self, X: torch.Tensor):
    y = F.relu(self.fc1(X))
    y = self.fc2(y)
    return y


class PositionalEmbeddingLayer(nn.Module):
  """
    Implements positional embeddings for sequence data.

    This layer supports both fixed (sinusoidal) and learnable positional embeddings.

    Args:
        embed_size (int): The dimensionality of the embeddings.
        embeddings (str, optional): The type of positional embeddings to use.
            - "fixed" (default): Uses sinusoidal positional embeddings.
            - "learnable": Uses trainable positional embeddings.

    Raises:
        ValueError: If the embeddings argument is not "fixed" or "learnable".
  """
  def __init__(self, max_seq_length: int, embed_size: int, embeddings: str = "fixed"):
    super(PositionalEmbeddingLayer, self).__init__()
    self.embed_size = embed_size
    self.embeddings = embeddings
    self.max_seq_length = max_seq_length

    if self.embeddings == "learnable":
        self.position_embeddings = nn.Parameter(torch.randn(max_seq_length, embed_size))
    elif self.embeddings == "fixed":
        self.register_buffer("position_embeddings", self._get_sinusoidal_embeddings(max_seq_length, embed_size))
    else:
        raise ValueError(f"{self.embeddings} is not a valid embedding type. Must be either 'fixed' or 'learnable'.")

  def _get_sinusoidal_embeddings(self, max_seq_len: int, embed_size: int):
    position = torch.arange(max_seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))

    pos_embeddings = torch.zeros(max_seq_len, embed_size)
    pos_embeddings[:, 0::2] = torch.sin(position * div_term)
    pos_embeddings[:, 1::2] = torch.cos(position * div_term)

    return pos_embeddings

  def forward(self, X: torch.Tensor):
    seq_len = X.shape[1]
    return X + self.position_embeddings[:seq_len, :]