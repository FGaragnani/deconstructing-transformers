import torch
import torch.nn as nn
import torch.optim as optim
import random

from src.seca import train_SECA, test_SECA
from src.model import TransformerLikeModel, EncoderOnlyModel
from torch.utils.data import DataLoader

from typing import Tuple

def train_transformer_model(model: TransformerLikeModel, epochs: int, train_data_loader: DataLoader, test_data_loader: DataLoader, verbose: bool = True,
                teacher_forcing_ratio: float = 1.0, pretrain_seca: bool = True) -> Tuple[float, float]:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  model.train()
  if pretrain_seca:
    train_SECA(model.seca, optim.Adam(model.seca.parameters(), lr=5e-5), train_data_loader, 500, verbose)
    test_SECA(model.seca, test_data_loader, verbose)
    model.seca.unfreeze()

  optimizer = optim.AdamW(model.parameters(), lr=1e-4)
  criterion = nn.MSELoss()

  ret_train_loss = 0.0
  ret_test_loss = 0.0

  for epoch in range(epochs):
    epoch_loss = 0

    for X_batch, y_batch in train_data_loader:
      X_batch, y_batch = X_batch.to(device), y_batch.to(device)
      batch_size, output_len, _ = y_batch.shape

      optimizer.zero_grad()
      Y = model.cls_token.expand((batch_size, 1, -1))
      total_loss = torch.tensor(0.0, device=device)

      for step in range(model.output_len):
        output = model.single_forward((X_batch, Y))
        if random.random() < teacher_forcing_ratio:
          Y = torch.cat([Y, model.seca.encode(y_batch[:, step].unsqueeze(1))], dim=1)
        else:
          Y = torch.cat([Y, output.unsqueeze(1)], dim=1)
        y = model.seca.decode(output)
        loss = criterion(y, y_batch[:, step])
        loss = torch.sqrt(loss)  # Use RMSE
        total_loss = total_loss + loss

      total_loss.backward()
      optimizer.step()

      epoch_loss += total_loss.item()

    if (epoch + 1) % 5 == 0:
      teacher_forcing_ratio *= 0.9

    ret_train_loss = epoch_loss / len(train_data_loader)

    if verbose:
      print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_data_loader):.4f}")

  model.eval()
  test_loss = 0
  
  if verbose:
    for X_test, y_test in test_data_loader:
      X_test, y_test = X_test.to(device), y_test.to(device)
      print(f"Input sequence: {X_test[0]}\nTarget sequence: {y_test[0]}\nPredicted sequence: {model(X_test)[0]}")
      break

  for X_test, y_test in test_data_loader:
    X_test, y_test = X_test.to(device), y_test.to(device)
    outputs = model(X_test)
    loss = criterion(outputs, y_test)
    loss = torch.sqrt(loss)  # Use RMSE
    test_loss += loss.item()

  ret_test_loss = test_loss / len(test_data_loader)

  if verbose:
      print(f"Test loss: {test_loss/len(test_data_loader):.4f}")

  return ret_train_loss, ret_test_loss

def train_encoder_model(model: EncoderOnlyModel, epochs: int, train_data_loader: DataLoader, test_data_loader: DataLoader, verbose: bool = True,
                teacher_forcing_ratio: float = 1.0) -> Tuple[float, float]:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  model.train()
  train_SECA(model.seca, optim.Adam(model.seca.parameters(), lr=5e-5), train_data_loader, epochs * 3, verbose)
  test_SECA(model.seca, test_data_loader, verbose)
  model.seca.unfreeze()

  optimizer = optim.Adam(model.parameters(), lr=1e-4)
  criterion = nn.MSELoss()

  ret_train_loss = 0.0
  ret_test_loss = 0.0

  for epoch in range(epochs):
    epoch_loss = 0

    for X_batch, y_batch in train_data_loader:
      X_batch, y_batch = X_batch.to(device), y_batch.to(device)
      batch_size, output_len, _ = y_batch.shape

      optimizer.zero_grad()
      total_loss = torch.tensor(0.0, device=device)

      for step in range(model.output_len):
        y = model.single_forward(X_batch)
        X_batch = X_batch[:, 1:]
        if random.random() < teacher_forcing_ratio:
          X_batch = torch.cat([X_batch, model.seca.encode(y_batch[:, step].unsqueeze(1))], dim=1)
        else:
          X_batch = torch.cat([X_batch, y.unsqueeze(1)], dim=1)
        y = model.seca.decode(y)
        loss = criterion(y, y_batch[:, step])
        loss = torch.sqrt(loss)  # Use RMSE
        total_loss = total_loss + loss

      total_loss.backward()
      optimizer.step()

      epoch_loss += total_loss.item()

    if (epoch + 1) % 5 == 0:
      teacher_forcing_ratio *= 0.9

    ret_train_loss = epoch_loss / len(train_data_loader)

    if verbose:
      print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_data_loader):.4f}")

  model.eval()
  test_loss = 0
  
  if verbose:
    for X_test, y_test in test_data_loader:
      X_test, y_test = X_test.to(device), y_test.to(device)
      predicted = model(X_test)
      print(f"Input sequence: {X_test[0]}\nTarget sequence: {y_test[0]}\nPredicted sequence: {predicted[0]}")
      break

  for X_test, y_test in test_data_loader:
    X_test, y_test = X_test.to(device), y_test.to(device)
    outputs = model(X_test)
    loss = criterion(outputs, y_test)
    loss = torch.sqrt(loss)  # Use RMSE
    test_loss += loss.item()

  ret_test_loss = test_loss / len(test_data_loader)

  if verbose:
      print(f"Test loss: {test_loss/len(test_data_loader):.4f}")

  return ret_train_loss, ret_test_loss