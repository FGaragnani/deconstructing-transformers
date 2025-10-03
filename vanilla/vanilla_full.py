import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import random
from datetime import datetime
import joblib

# 3. Modello Transformer
class TransformerAutoregressive(nn.Module):
   def __init__(self, num_features, embed_size, num_heads, num_layers, ff_hidden_dim, dropout=0.1):
      super(TransformerAutoregressive, self).__init__()
      self.embed_size = embed_size
      self.num_features = num_features
      self.input_linear = nn.Linear(num_features, embed_size)
      self.target_linear = nn.Linear(num_features, embed_size)
      self.pos_encoder = PositionalEncoding(embed_size, dropout)
      self.transformer = nn.Transformer(
         d_model=embed_size, nhead=num_heads,
         num_encoder_layers=num_layers, num_decoder_layers=num_layers,
         dim_feedforward=ff_hidden_dim, dropout=dropout, batch_first=False
      )
      self.fc_out = nn.Linear(embed_size, num_features)

   def _generate_square_subsequent_mask(self, sz, device):
      mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
      mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
      return mask

   def forward(self, src, tgt):
      device = src.device
      tgt_seq_len = tgt.size(0)
      tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device)
      src_emb = self.input_linear(src) * math.sqrt(self.embed_size)
      src_pos = self.pos_encoder(src_emb)
      tgt_emb = self.target_linear(tgt) * math.sqrt(self.embed_size)
      tgt_pos = self.pos_encoder(tgt_emb)
      output = self.transformer(src_pos, tgt_pos, src_mask=None, tgt_mask=tgt_mask)
      return self.fc_out(output)

   def predict(self, src, horizon, device):
      self.eval()
      with torch.no_grad():
         src_emb = self.input_linear(src) * math.sqrt(self.embed_size)
         src_pos = self.pos_encoder(src_emb)
         memory = self.transformer.encoder(src_pos)
         decoder_input = torch.zeros(1, 1, self.num_features).to(device)
         predictions = []
         for _ in range(horizon):
            tgt_seq_len = decoder_input.size(0)
            tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device)
            tgt_emb = self.target_linear(decoder_input) * math.sqrt(self.embed_size)
            tgt_pos = self.pos_encoder(tgt_emb)
            decoder_output = self.transformer.decoder(tgt_pos, memory, tgt_mask=tgt_mask)
            last_step_output = decoder_output[-1, :, :]
            prediction = self.fc_out(last_step_output).unsqueeze(0)
            predictions.append(prediction)
            decoder_input = torch.cat([decoder_input, prediction], dim=0)
         final_predictions = torch.cat(predictions, dim=0)
         return final_predictions.permute(1, 0, 2)

# 0. Classe PositionalEncoding
class PositionalEncoding(nn.Module):
   def __init__(self, d_model, dropout=0.1, max_len=5000):
      super(PositionalEncoding, self).__init__()
      self.dropout = nn.Dropout(p=dropout)
      pe = torch.zeros(max_len, d_model)
      position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
      pe[:, 0::2] = torch.sin(position * div_term)
      pe[:, 1::2] = torch.cos(position * div_term)
      pe = pe.unsqueeze(0).transpose(0, 1)
      self.register_buffer('pe', pe)

   def forward(self, x):
      x = x + self.pe[:x.size(0), :]
      return self.dropout(x)

class WeightedPeakLoss(nn.Module):
   def __init__(self, peak_penalty=3.0, threshold=0.75):
      super(WeightedPeakLoss, self).__init__()
      self.peak_penalty = peak_penalty
      self.threshold = threshold
      self.mse_loss = nn.MSELoss(reduction='none')

   def forward(self, y_pred, y_true):
      loss = self.mse_loss(y_pred, y_true)
      weights = torch.ones_like(y_true)
      peak_indices = y_true > self.threshold
      weights[peak_indices] = self.peak_penalty
      weighted_loss = loss * weights
      return torch.mean(weighted_loss)

def set_seed(seed_value=42):
   random.seed(seed_value)
   np.random.seed(seed_value)
   torch.manual_seed(seed_value)
   if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed_value)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

# 2. Creazione Sequenze
def create_one_shot_sequences(data, input_length, horizon):
   X, y = [], []
   for i in range(len(data) - input_length - horizon + 1):
      input_seq = data[i:(i + input_length), :]
      output_seq = data[(i + input_length):(i + input_length + horizon), :]
      X.append(input_seq)
      y.append(output_seq)
   return np.array(X), np.array(y)

# 4. Funzione di Training
def train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs, patience,
          fold_dir):  # NUOVO: Aggiunto fold_dir
   best_val_loss = float('inf')
   epochs_no_improve = 0
   best_model_path = os.path.join(fold_dir,
                                  'best_transformer_model.pth')  # NUOVO: Percorso del modello specifico per il fold
   history = {'train_loss': [], 'val_loss': []}

   for epoch in range(num_epochs):
      model.train()
      total_train_loss = 0
      for src_batch, y_batch in train_loader:
         src_batch, y_batch = src_batch.to(device), y_batch.to(device)
         src_tensor     = src_batch.permute(1, 0, 2)
         y_tensor       = y_batch.permute(1, 0, 2)
         batch_size     = src_batch.size(0)
         num_features   = src_batch.size(2)
         start_token    = torch.zeros((1, batch_size, num_features), device=device)
         decoder_input  = torch.cat([start_token, y_tensor[:-1, :, :]], dim=0)
         optimizer.zero_grad()
         output         = model(src_tensor, decoder_input)
         loss           = criterion(output, y_tensor)
         loss.backward()
         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
         optimizer.step()
         total_train_loss += loss.item()

      avg_train_loss = total_train_loss / len(train_loader)
      history['train_loss'].append(avg_train_loss)

      model.eval()
      total_val_loss = 0
      with torch.no_grad():
         for src_batch, y_batch in val_loader:
            src_batch, y_batch = src_batch.to(device), y_batch.to(device)
            src_tensor = src_batch.permute(1, 0, 2)
            y_tensor = y_batch.permute(1, 0, 2)
            batch_size = src_batch.size(0)
            start_token = torch.zeros((1, batch_size, num_features), device=device)
            decoder_input = torch.cat([start_token, y_tensor[:-1, :, :]], dim=0)
            output = model(src_tensor, decoder_input)
            val_loss = criterion(output, y_tensor)
            total_val_loss += val_loss.item()

      avg_val_loss = total_val_loss / len(val_loader)
      history['val_loss'].append(avg_val_loss)
      scheduler.step(avg_val_loss)

      if (epoch + 1) % 10 == 0:
         print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

      if avg_val_loss < best_val_loss:
         best_val_loss = avg_val_loss
         epochs_no_improve = 0
         torch.save(model.state_dict(), best_model_path)  # Salva il modello migliore
      else:
         epochs_no_improve += 1
      if epochs_no_improve >= patience:
         print(f"--- Early stopping at epoch {epoch + 1} ---")
         break

   print(f"Loading best model from '{best_model_path}' with Val Loss: {best_val_loss:.6f}")
   model.load_state_dict(torch.load(best_model_path))
   return model, history

if __name__ == "__main__":
   # --- Gestione Cartelle e File ---
   # Crea una cartella principale per questo run, basata sul timestamp corrente
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   RESULTS_DIR = f"transformer_results_{timestamp}"
   os.makedirs(RESULTS_DIR, exist_ok=True)
   print(f"I risultati, i modelli e i grafici verranno salvati in: '{RESULTS_DIR}'")

   # 1. Caricamento Dati e Feature Engineering
   try:
      data_df = pd.read_csv("testDati.csv", header=None)
   except FileNotFoundError:
      print("ERRORE: File 'testDati.csv' non trovato.")
      exit()

   TARGET_COL_INDEX_ORIGINAL = 8
   data_log = np.log1p(data_df)
   start_date = pd.to_datetime("1990-01-01")
   dates = pd.date_range(start_date, periods=len(data_df), freq='MS')
   data_log['year'] = dates.year
   data_log['month_sin'] = np.sin(2 * np.pi * dates.month / 12.0)
   data_log['month_cos'] = np.cos(2 * np.pi * dates.month / 12.0)
   target_col_name = data_log.columns[TARGET_COL_INDEX_ORIGINAL]
   data_log[f'target_lag_1'] = data_log[target_col_name].shift(1)
   data_log[f'target_lag_12'] = data_log[target_col_name].shift(12)
   data_log[f'target_lag_24'] = data_log[target_col_name].shift(24)
   data_log[f'rolling_mean_12'] = data_log[target_col_name].rolling(window=12).mean()
   data_log[f'rolling_std_12'] = data_log[target_col_name].rolling(window=12).std()
   peak_feature_threshold = data_log[target_col_name].quantile(0.85)
   data_log['is_peak'] = (data_log[target_col_name] > peak_feature_threshold).astype(int)
   data_log['is_peak_lag_1'] = data_log['is_peak'].shift(1)
   data_log = data_log.dropna().reset_index(drop=True)

   selected_columns = [
      TARGET_COL_INDEX_ORIGINAL, 11, 6, 1, 4, 10, 5, 7,
      data_log.columns.get_loc('target_lag_1'),
      data_log.columns.get_loc('target_lag_12'),
      data_log.columns.get_loc('target_lag_24'),
      data_log.columns.get_loc('rolling_mean_12'),
      data_log.columns.get_loc('rolling_std_12'),
      data_log.columns.get_loc('year'),
      data_log.columns.get_loc('month_sin'),
      data_log.columns.get_loc('month_cos'),
      data_log.columns.get_loc('is_peak_lag_1')
   ]
   values = data_log.iloc[:, selected_columns].values.astype('float32')
   TARGET_COL_INDEX = 0
   N_FEATURES = values.shape[1]
   print(f"Numero di feature totali: {N_FEATURES}")

   scalers = {}
   scaled_columns = []
   for i in range(values.shape[1]):
      scaler = StandardScaler()
      col_scaled = scaler.fit_transform(values[:, i].reshape(-1, 1))
      scalers[i] = scaler
      scaled_columns.append(col_scaled)
   scaled_values = np.hstack(scaled_columns)

   # Salva gli scaler per poterli riutilizzare
   scaler_path = os.path.join(RESULTS_DIR, 'scalers.gz')
   joblib.dump(scalers, scaler_path)
   print(f"Scalers salvati in: {scaler_path}")

   # 5. Cross-Validation
   set_seed(995)
   embed_size     = 64
   num_heads      = 4
   num_layers     = 2
   ff_hidden_dim  = 256
   dropout        = 0.11
   INPUT_LENGTH   = 24
   N_SPLITS       = 5
   BATCH_SIZE     = 64
   LEARNING_RATE  = 0.0003
   NUM_EPOCHS     = 800
   WEIGHT_DECAY   = 1.1564807738223053e-05
   PATIENCE       = 70
   PEAK_PENALTY_FACTOR = 10.0
   PEAK_QUANTILE  = 0.60
   HORIZON        = 12

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")

   tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=HORIZON)

   all_y_test_rescaled, all_y_pred_rescaled = [], []
   mse_scores_rescaled, mae_scores_rescaled = [], []
   fold_counter = 0

   for train_index, test_index in tscv.split(scaled_values):
      fold_counter += 1
      print(f"\n--- FOLD {fold_counter}/{N_SPLITS} ---")

      # Crea una sottocartella per il fold corrente
      fold_dir = os.path.join(RESULTS_DIR, f"fold_{fold_counter}")
      os.makedirs(fold_dir, exist_ok=True)

      train_data = scaled_values[train_index]
      test_data  = scaled_values[test_index]
      print(f"Lunghezza Training: {len(train_data)}, Lunghezza Test: {len(test_data)}")

      X_train_full, y_train_full = create_one_shot_sequences(train_data, INPUT_LENGTH, HORIZON)

      if len(X_train_full) <= HORIZON:
         print(f"Fold saltato: dati di training insufficienti ({len(X_train_full)} sequenze).")
         continue

      val_size = HORIZON
      X_train, y_train = X_train_full[:-val_size], y_train_full[:-val_size]
      X_val, y_val = X_train_full[-val_size:], y_train_full[-val_size:]
      print(f"Sequenze di Training: {len(X_train)}, Sequenze di Validazione: {len(X_val)}")

      train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
      train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
      val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
      val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

      target_train_data = train_data[:, TARGET_COL_INDEX]
      peak_threshold_value = np.quantile(target_train_data, PEAK_QUANTILE)
      criterion = WeightedPeakLoss(peak_penalty=PEAK_PENALTY_FACTOR, threshold=peak_threshold_value).to(device)

      # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
      # >>>>>>>>>>>>>>>>>>>> QUI IL TRANSFORMER <<<<<<<<<<<<<<<<<<<<<<<<<
      model     = TransformerAutoregressive(N_FEATURES, embed_size, num_heads, num_layers, ff_hidden_dim, dropout).to(device)
      optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
      scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=15)

      model, history = train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, NUM_EPOCHS,
                             PATIENCE, fold_dir)
      print(f"Modello del Fold {fold_counter} salvato in '{fold_dir}'")

      # Salva la history del training
      pd.DataFrame(history).to_csv(os.path.join(fold_dir, 'training_history.csv'), index=False)

      model.eval()
      context = train_data[-INPUT_LENGTH:]
      context_tensor = torch.FloatTensor(context).unsqueeze(1).to(device)
      prediction_scaled_tensor = model.predict(context_tensor, HORIZON, device)
      prediction_scaled = prediction_scaled_tensor.squeeze(0).cpu().numpy()

      y_pred_fold_scaled = prediction_scaled[:, TARGET_COL_INDEX]
      y_test_fold_scaled = test_data[:, TARGET_COL_INDEX]
      y_pred_log = scalers[TARGET_COL_INDEX].inverse_transform(y_pred_fold_scaled.reshape(-1, 1))
      y_test_log = scalers[TARGET_COL_INDEX].inverse_transform(y_test_fold_scaled.reshape(-1, 1))
      y_pred_final = np.expm1(y_pred_log).flatten()
      y_test_final = np.expm1(y_test_log).flatten()

      # Salva le predizioni del fold
      pd.DataFrame({
         'y_test_final': y_test_final,
         'y_pred_final': y_pred_final
      }).to_csv(os.path.join(fold_dir, 'predictions.csv'), index=False)

      all_y_pred_rescaled.extend(y_pred_final)
      all_y_test_rescaled.extend(y_test_final)

      mse_fold = mean_squared_error(y_test_final, y_pred_final)
      mae_fold = mean_absolute_error(y_test_final, y_pred_final)
      mse_scores_rescaled.append(mse_fold)
      mae_scores_rescaled.append(mae_fold)
      print(f"  RMSE Fold {fold_counter} (original scale): {np.sqrt(mse_fold):.4f}")
      print(f"  MAE Fold {fold_counter} (original scale): {mae_fold:.4f}")
      print(f"  MSE Fold {fold_counter} (original scale): {mse_fold:.4f}")

   # 6. Risultati Finali e Plot
   y_true_combined = np.array(all_y_test_rescaled)
   y_pred_combined = np.array(all_y_pred_rescaled)
   valid_indices = np.isfinite(y_true_combined) & np.isfinite(y_pred_combined)
   y_true_combined = y_true_combined[valid_indices]
   y_pred_combined = y_pred_combined[valid_indices]

   print("\n--- Metriche Globali (calcolate su tutti i dati di test combinati) ---")
   overall_rmse = np.sqrt(mean_squared_error(y_true_combined, y_pred_combined))
   overall_mae = mean_absolute_error(y_true_combined, y_pred_combined)
   print(f"  RMSE Globale: {overall_rmse:.4f}")
   print(f"  MAE Globale: {overall_mae:.4f}")

   print("\n--- Metriche Medie della Cross-Validation (media dei punteggi di ogni fold) ---")
   mean_mse = np.mean(mse_scores_rescaled)
   mean_rmse = np.sqrt(mean_mse)
   mean_mae = np.mean(mae_scores_rescaled)
   print(f"  MSE Medio: {mean_mse:.4f}")
   print(f"  RMSE Medio: {mean_rmse:.4f}")
   print(f"  MAE Medio: {mean_mae:.4f}")

   # Salva un riepilogo delle metriche
   with open(os.path.join(RESULTS_DIR, 'summary_metrics.txt'), 'w') as f:
      f.write("--- Risultati Finali della Cross-Validation ---\n")
      f.write(f"RMSE Globale: {overall_rmse:.4f}\n")
      f.write(f"MAE Globale: {overall_mae:.4f}\n")
      f.write(f"MSE Medio (dai fold): {mean_mse:.4f}\n")
      f.write(f"RMSE Medio (dai fold): {mean_rmse:.4f}\n")
      f.write(f"MAE Medio (dai fold): {mean_mae:.4f}\n")

   print("\nVisualizzazione e salvataggio dei risultati di tutti i fold...")
   plt.figure(figsize=(18, 9))
   prediction_indices = np.arange(len(y_true_combined))
   plt.plot(prediction_indices, y_true_combined, label='Valori Reali (Test)', color='blue', marker='o', linestyle='-',
            markersize=4)
   plt.plot(prediction_indices, y_pred_combined, label='Valori Predetti (Test)', color='red', marker='x', linestyle='--',
            markersize=4)
   for i in range(N_SPLITS - 1):
      vline_pos = (i + 1) * HORIZON - 0.5
      plt.axvline(x=vline_pos, color='gray', linestyle=':', label=f'Separazione Fold' if i == 0 else "")
   plt.title(f"Previsioni vs Reali sui Dati di Test - Tutti i {N_SPLITS} Fold")
   plt.xlabel("Time Step (nei set di test combinati)")
   plt.ylabel("Valore Originale della Feature Target")
   plt.legend()
   plt.grid(True, linestyle='--', alpha=0.6)
   plt.tight_layout()
   plt.savefig(os.path.join(RESULTS_DIR, 'combined_predictions_plot.png'))
   plt.show()

   # 7. ANALISI DEGLI ERRORI
   print("\n--- Analisi e Salvataggio degli Errori ---")
   if len(y_true_combined) > 0 and len(y_pred_combined) > 0:
      residuals = y_true_combined - y_pred_combined

      plt.figure(figsize=(18, 6))
      plt.plot(residuals, label='Residui (Reale - Predetto)', color='purple', alpha=0.9)
      plt.axhline(0, color='red', linestyle='--', label='Errore Zero')
      plt.title('Andamento dei Residui nel Tempo (su tutti i fold di test)')
      plt.xlabel('Time Step (nei set di test combinati)')
      plt.ylabel('Errore di Previsione')
      plt.legend()
      plt.grid(True, linestyle='--', alpha=0.5)
      plt.savefig(os.path.join(RESULTS_DIR, 'residuals_plot.png'))
      plt.show()

      plt.figure(figsize=(8, 8))
      plt.scatter(y_true_combined, y_pred_combined, alpha=0.6, label='Previsioni')
      perfect_line = np.linspace(min(y_true_combined.min(), y_pred_combined.min()),
                                 max(y_true_combined.max(), y_pred_combined.max()), 100)
      plt.plot(perfect_line, perfect_line, color='red', linestyle='--', label='Predizione Perfetta (y=x)')
      plt.title('Valori Reali vs. Valori Predetti')
      plt.xlabel('Valori Reali')
      plt.ylabel('Valori Predetti')
      plt.legend()
      plt.grid(True, linestyle='--', alpha=0.5)
      plt.axis('equal')
      plt.gca().set_aspect('equal', adjustable='box')
      plt.savefig(os.path.join(RESULTS_DIR, 'scatter_plot_real_vs_pred.png'))
      plt.show()

      epsilon = 1e-9
      y_true_safe = np.where(y_true_combined == 0, epsilon, y_true_combined)
      relative_errors = np.abs(residuals) / y_true_safe
      mean_relative_error = np.mean(relative_errors) * 100
      print(f"  Errore relativo medio: {mean_relative_error:.2f}%")
      with open(os.path.join(RESULTS_DIR, 'summary_metrics.txt'), 'a') as f:
         f.write(f"Errore relativo medio: {mean_relative_error:.2f}%\n")
   else:
      print("Nessun risultato di test da analizzare.")

   # 8. ANDAMENTO DELLA LOSS
   print("\n--- Andamento della Loss (ultimo fold) ---")
   if 'history' in locals() and len(history['train_loss']) > 0:
      plt.figure(figsize=(12, 6))
      plt.plot(history['train_loss'], label='Training Loss', color='teal')
      plt.plot(history['val_loss'], label='Validation Loss', color='orange')
      plt.title(f'Andamento della Loss durante il Training - Fold {fold_counter}')
      plt.xlabel('Epoca')
      plt.ylabel('Loss (Weighted MSE)')
      plt.legend()
      plt.grid(True, linestyle='--', alpha=0.6)
      plt.savefig(os.path.join(RESULTS_DIR, f'loss_plot_fold_{fold_counter}.png'))
      plt.show()
   else:
      print("Nessuna cronologia della loss da visualizzare.")

   print(f"\n✅ Esecuzione completata. Tutti i risultati sono stati salvati in '{RESULTS_DIR}'.")
