import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Transformer ----------
class TransformerTimeSeries(nn.Module):
   def __init__(self, d_model=64, nhead=2, num_layers=1):
      super().__init__()
      self.input_layer = nn.Linear(1, d_model)
      self.pos_encoder = PositionalEncoding(d_model)
      encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128)
      self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
      self.decoder = nn.Linear(d_model, 1)

   def forward(self, src):
      # src: (seq_len, batch, 1)
      src = self.input_layer(src)
      src = self.pos_encoder(src)
      output = self.transformer(src)
      return self.decoder(output)

# ---------- Positional Encoding ----------
class PositionalEncoding(nn.Module):
   def __init__(self, d_model, max_len=5000):
      super().__init__()
      pe = torch.zeros(max_len, d_model)
      position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
      pe[:, 0::2] = torch.sin(position * div_term)
      pe[:, 1::2] = torch.cos(position * div_term)
      pe = pe.unsqueeze(1)
      self.register_buffer('pe', pe)

   def forward(self, x):
      return x + self.pe[:x.size(0)]

def create_sequences(data, seq_len=30, horizon=12):
   xs, ys = [], []
   for i in range(len(data) - seq_len - horizon):
      x = data[i:i + seq_len]
      y = data[i + seq_len:i + seq_len + horizon]
      xs.append(x)
      ys.append(y)
   return np.array(xs), np.array(ys)

# ---------- Forecast (autoregressive) ----------
def forecast_autoregressive(model, history, steps=12):
   model.eval()
   history = history.copy().tolist()
   for _ in range(steps):
      seq = torch.tensor(history[-seq_len:], dtype=torch.float32).unsqueeze(-1).unsqueeze(1).to(device)
      with torch.no_grad():
         pred = model(seq)[-1].cpu().item()
      history.append(pred)
   return history[-steps:]

if __name__ == "__main__":
   df = pd.read_excel("M3C.xls", sheet_name='M3Month')
   arrSeries = ['N1652', 'N1546', 'N1894', 'N2047', 'N2255', 'N2492', 'N2594', 'N2658', 'N2737', 'N2758', 'N2817', 'N2823']
   n = 18  # num elementi da predire
   for i in range(len(arrSeries)):
      series = df.loc[df['Series'] == arrSeries[i], 6:].dropna(axis=1, how='all').values.flatten()
      # s = "N1652.csv" #"auto_italia.csv"
      # series = pd.read_csv(datafile).values.astype(np.float32)
      # series = series.flatten()
      series = (series - series.mean()) / series.std()
      n     = 18
      niter = 1000
      batch_size = 32

      seq_len, horizon = 12, n
      X, Y = create_sequences(series, seq_len, horizon)
      X = torch.from_numpy(X).float()  # (samples, seq_len)
      Y = torch.from_numpy(Y).float()  # (samples, horizon)

      # ---------- Training ----------
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model = TransformerTimeSeries().to(device)
      criterion = nn.MSELoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

      for epoch in range(niter):
         epoch_loss = 0.0
         for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].unsqueeze(-1).transpose(0, 1).to(device)  # (seq_len, batch, 1)
            yb = Y[i:i + batch_size].to(device)  # (batch, horizon)

            optimizer.zero_grad()
            enc_out = model(xb)  # (seq_len, batch, 1)
            last_token = enc_out[-1]  # (batch, 1)

            # project to horizon
            pred = last_token.repeat(1, horizon)  # (batch, horizon)
            loss = criterion(pred, yb)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
         if(epoch%10==0):
            print(f"Epoch {epoch + 1}, Loss {epoch_loss:.4f}, rmse {np.sqrt(epoch_loss):.3f}")

      last_seq = series[-seq_len:]
      forecast = forecast_autoregressive(model, last_seq, steps=horizon)

      fig = plt.figure(figsize=(12,6))
      plt.plot(series)
      plt.plot(np.arange(len(series)-n,len(series)),forecast)
      plt.show()

      print("12-step forecast:", forecast)
