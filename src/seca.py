import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class ScalarExpansionContractiveAutoencoder(nn.Module):
  def __init__(self, embed_size: int, input_size: int = 1, lam: float = 1e-10):
    super(ScalarExpansionContractiveAutoencoder, self).__init__()
    self.embed_size = embed_size
    self.input_size = input_size
    self.lam = lam

    self.encoder = nn.Linear(in_features=input_size, out_features=embed_size, bias=False)
    self.decoder = nn.Linear(in_features=embed_size, out_features=input_size, bias=False)

  def encode(self, X: torch.Tensor):
    X = self.encoder(X)
    return X

  def decode(self, X: torch.Tensor):
    X = self.decoder(X)
    return X

  def forward(self, X: torch.Tensor):
    h = self.encode(X)
    y = self.decode(h)
    return h, y

  def freeze(self):
    for param in self.parameters():
      param.requires_grad = False

  def unfreeze(self):
    for param in self.parameters():
      param.requires_grad = True

  def start(self):
    W_I = self.encoder.weight.data.clone()  # shape: [embed_size, input_size]
    norm_sq = torch.norm(W_I, p=2) ** 2
    W_O = W_I / norm_sq
    self.decoder.weight.data.copy_(W_O.T)

  """
    Compare with: https://icml.cc/2011/papers/455_icmlpaper.pdf
  """
  def loss(self, X: torch.Tensor, y: torch.Tensor, h: torch.Tensor):
    W = self.encoder.weight / torch.norm(self.encoder.weight, p=2)

    # Compute the error
    total_loss = torch.sum(abs(X- y)) / X.size(0)

    dh = (h > 0).float() # Hadamard product
    w_sum = torch.sum(W**2, dim=1, keepdim=True)
    contractive_loss = torch.sum(dh**2 @ w_sum)
    total_loss = total_loss + self.lam * contractive_loss

    return total_loss