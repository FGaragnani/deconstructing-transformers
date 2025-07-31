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

"""
Training loop for the SECA model
"""
def train_SECA(model: ScalarExpansionContractiveAutoencoder, optimizer: optim.Optimizer, data_loader: DataLoader, 
               epochs: int = 50, verbose: bool = True):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.train()
  model.unfreeze()

  if verbose:
    print(f"---\tSECA Training - {epochs} Epochs\t---")

  for i in range(epochs):
    loss_value = 0

    for _, (X, _) in enumerate(data_loader):
        X = X.to(device)
        h, y = model(X)
        # loss = model.loss(X, y, h)
        loss = torch.mean(torch.abs(X - y))
        optimizer.zero_grad()
        loss.backward()
        loss_value += loss.item()
        optimizer.step()

    if verbose:
      print(f"Epoch {i + 1}/{epochs}, Loss: {loss_value}")

  model.freeze()

"""
Testing loop for the SECA model
"""
def test_SECA(model: ScalarExpansionContractiveAutoencoder, data_loader: DataLoader, verbose: bool = True):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.freeze()

  loss_value = 0

  if verbose:
    print(f"---\tSECA Testing\t---")

  for _, (X, _) in enumerate(data_loader):
    X = X.to(device)
    h, y = model(X)
    # loss = model.loss(X, y, h)
    loss = torch.mean(torch.abs(X - y))
    loss_value += loss.item()

  if verbose:
      print(f"Global Loss: {loss_value}")
      print(f"Average Loss: {loss_value/len(data_loader):.4f}")