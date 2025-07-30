import torch
import torch.nn as nn
import torch.optim as optim
import random

from src.seca import train_SECA, test_SECA
from src.model import TransformerLikeModel
from torch.utils.data import DataLoader

def train_model(model: TransformerLikeModel, epochs: int, train_data_loader: DataLoader, test_data_loader: DataLoader, verbose: bool = True,
                teacher_forcing_ratio: float = 1.0):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  model.train()
  train_SECA(model.seca, optim.Adam(model.seca.parameters(), lr=5e-5), train_data_loader, epochs * 3, verbose)
  test_SECA(model.seca, test_data_loader, verbose)

  optimizer = optim.Adam(model.parameters(), lr=1e-4)
  criterion = nn.MSELoss()

  for epoch in range(epochs):
    epoch_loss = 0

    for X_batch, y_batch in train_data_loader:
      X_batch, y_batch = X_batch.to(device), y_batch.to(device)
      batch_size, output_len, _ = y_batch.shape

      optimizer.zero_grad()
      Y = model.cls_token.expand((batch_size, 1, -1))
      total_loss = 0

      for step in range(model.output_len):
        output = model.single_forward((X_batch, Y))
        if random.random() < teacher_forcing_ratio:
          Y = torch.cat([Y, model.seca.encode(y_batch[:, step].unsqueeze(1))], dim=1)
        else:
          Y = torch.cat([Y, output], dim=1)
        y = model.seca.decode(output)
        loss = criterion(y, y_batch[:, step].unsqueeze(-1))
        total_loss += loss

      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()

    if (epoch + 1) % 5 == 0:
      teacher_forcing_ratio *= 0.9

    if verbose:
      print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_data_loader):.4f}")

  model.eval()
  test_loss = 0
  for X_test, y_test in test_data_loader:
    X_test, y_test = X_test.to(device), y_test.to(device)
    outputs = model(X_test)
    loss = criterion(outputs, y_test)
    test_loss += loss.item()

  if verbose:
      print(f"Test loss: {test_loss/len(test_data_loader):.4f}")