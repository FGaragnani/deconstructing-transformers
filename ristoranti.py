import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.model import TransformerLikeModel
from src.train import train_transformer_model

import matplotlib.pyplot as plt

SEQUENCE_LENGTH = 7
PREDICTION_LENGTH = 7
EMBED_SIZE = 4
ENCODER_SIZE = 1
DECODER_SIZE = 1
NUM_HEADS = 2
BATCH_SIZE = 2
EPOCHS = 500
DROPOUT = 0.001
TRAIN_PERCENTAGE = 0.75
SEED = 42
DELTA = False
EARLY_STOPPING = False

class RistorantiDataset(Dataset):
    def __init__(self, series, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.series = series.astype(np.float32)
        self.samples = []
        for i in range(len(series) - seq_len - pred_len + 1):
            x = self.series[i:i+seq_len]
            y = self.series[i+seq_len:i+seq_len+pred_len]
            self.samples.append((x, y))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x).unsqueeze(-1), torch.tensor(y).unsqueeze(-1)

def main():

    df = pd.read_csv('ristorantiGTrend.csv')
    series = df.iloc[:, 1].values.astype(np.float32)

    series = series[:35]
    print("Timesteps: ", len(series))
    min_val = np.min(series)
    max_val = np.max(series)
    series = (series - min_val) / (max_val - min_val)

    dataset = RistorantiDataset(series, SEQUENCE_LENGTH, PREDICTION_LENGTH)

    torch.manual_seed(SEED)

    split_idx = int(TRAIN_PERCENTAGE * len(dataset))
    train_set = torch.utils.data.Subset(dataset, range(split_idx))
    test_set = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
    print(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    """
    model = TransformerLikeModel(
        embed_size=EMBED_SIZE,
        encoder_size=ENCODER_SIZE,
        decoder_size=DECODER_SIZE,
        output_len=PREDICTION_LENGTH,
        num_head_enc=NUM_HEADS,
        num_head_dec_1=NUM_HEADS,
        num_head_dec_2=NUM_HEADS,
        dropout=DROPOUT,
        max_seq_length=SEQUENCE_LENGTH
    )
    train_transformer_model(
        model,
        epochs=EPOCHS,
        train_data_loader=train_loader,
        test_data_loader=test_loader,
        verbose=True,
        teacher_forcing_ratio=0.90,
        pretrain_seca=True,
        check_losses=EARLY_STOPPING,
        delta=DELTA,
        early_stopping=EARLY_STOPPING
    )
    """
    model = TransformerLikeModel.load_model("ristoranti_model.pth", TransformerLikeModel,
        embed_size=EMBED_SIZE,
        encoder_size=ENCODER_SIZE,
        decoder_size=DECODER_SIZE,
        output_len=PREDICTION_LENGTH,
        num_head_enc=NUM_HEADS,
        num_head_dec_1=NUM_HEADS,
        num_head_dec_2=NUM_HEADS,
        dropout=DROPOUT,
        max_seq_length=SEQUENCE_LENGTH)

    model.eval()
    with torch.no_grad():
        predictions = []
        for i in range(0, len(series) - SEQUENCE_LENGTH - PREDICTION_LENGTH + 1, PREDICTION_LENGTH):
            input_seq = torch.tensor(series[i:i + SEQUENCE_LENGTH].reshape(-1, 1), dtype=torch.float32).unsqueeze(0)
            pred = model.forward(input_seq).squeeze().cpu().numpy()
            # Ensure pred is a 1D array even when prediction length == 1
            if np.ndim(pred) == 0:
                pred = np.array([pred])
            else:
                pred = np.asarray(pred).reshape(-1)
            predictions.append(pred)

        prediction_series = np.full(len(series), np.nan)
        if DELTA:
            for idx, pred_window in enumerate(predictions):
                # Ensure pred_window is 1D
                pred_window = np.atleast_1d(pred_window).reshape(-1)
                start = SEQUENCE_LENGTH + idx * PREDICTION_LENGTH
                last_val = series[start - 1]
                abs_pred = []
                for delta in pred_window:
                    last_val = last_val + delta
                    abs_pred.append(last_val)
                end = start + len(abs_pred)
                prediction_series[start:end] = abs_pred
        else:
            if len(predictions) > 0:
                flat_preds = np.concatenate(predictions)
                prediction_series[SEQUENCE_LENGTH:SEQUENCE_LENGTH+len(flat_preds)] = flat_preds
        
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(14, 6))
        plt.plot(range(len(series)), series, 'b-', label='Standardized Input Series')
        test_start = len(series) - int((1 - TRAIN_PERCENTAGE) * len(series))
        plt.plot(range(test_start, len(series)), series[test_start:], 'g-', label='Test Series', linewidth=2)
        plt.plot(range(len(series)), prediction_series, 'r--', linewidth=3, label='Predicted Series')
        plt.title('Original vs Predicted Series')
        plt.xlabel('Time Step')
        plt.ylabel('Standardized Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    model.save_model("ristoranti_model.pth")

if __name__ == "__main__":
    main()