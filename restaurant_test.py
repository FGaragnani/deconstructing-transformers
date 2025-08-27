import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from typing import List, Tuple, Dict
from enum import Enum 
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import TransformerLikeModel
from src.train import train_transformer_model

class Normalization(Enum):
    RAW = "raw"
    MIN_MAX = "min_max"
    STANDARD = "standard"
    LOG = "log"
    LOG_MIN_MAX = "log_min_max"

    def initialize_values(self, max_val: float = 0, min_val: float = 0, std_val: float = 0, mean_val: float = 0):
        self._max_val: float = max_val
        self._min_val: float = min_val
        self._std_val: float = std_val
        self._mean_val: float = mean_val

    def normalize(self, series: np.ndarray) -> np.ndarray:
        if self == Normalization.RAW:
            return series
        elif self == Normalization.MIN_MAX:
            self.initialize_values(max_val=series.max(), min_val=series.min())
            return (series - self._min_val) / (self._max_val - self._min_val)
        elif self == Normalization.STANDARD:
            self.initialize_values(mean_val=series.mean(), std_val=series.std())
            return (series - self._mean_val) / self._std_val
        elif self == Normalization.LOG:
            return np.log(series + 1)
        elif self == Normalization.LOG_MIN_MAX:
            log_series = np.log(series + 1)
            self.initialize_values(max_val=log_series.max(), min_val=log_series.min())
            return (log_series - self._min_val) / (self._max_val - self._min_val)
        else:
            raise ValueError("Unknown normalization method")
        
    def convert(self, value: float) -> float:
        if self == Normalization.RAW:
            return value
        elif self == Normalization.MIN_MAX:
            return value * (self._max_val - self._min_val) + self._min_val
        elif self == Normalization.STANDARD:
            return value * self._std_val + self._mean_val
        elif self == Normalization.LOG:
            return np.exp(value) - 1
        elif self == Normalization.LOG_MIN_MAX:
            return (np.exp(value) - 1) * (self._max_val - self._min_val) + self._min_val
        else:
            raise ValueError("Unknown normalization method")

def create_restaurant_datasets(Xtrain_length: int = 12, Ytrain_length: int = 4, normalization: Normalization = Normalization.STANDARD) -> Tuple[TensorDataset, np.ndarray]:
    """
       Create training and testing datasets from the airline passenger data.

       :param Xtrain_length: Length of the input sequences
       :param Ytrain_length: Length of the prediction sequences
       :param normalization: Normalization method to apply

       :return: A tuple containing the training/testing datasets and the original series
    """
    timeseries = pd.read_csv("ristorantiGTrend.csv",usecols=[1]).values
    timeseries = normalization.normalize(timeseries)

    datasets = []
    for i in range(0, len(timeseries) - Xtrain_length - Ytrain_length + 1):
        seq    = timeseries[i:i + Xtrain_length].reshape(-1, 1)
        target = timeseries[i + Xtrain_length:i + Xtrain_length + Ytrain_length].reshape(-1, 1)
        datasets.append((torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)))

    datasets = TensorDataset(
        torch.stack([item[0] for item in datasets]),
        torch.stack([item[1] for item in datasets])
    )

    return datasets, timeseries

def main():
    NUM_HEADS     = 2
    EMBED_SIZE    = 4
    ENCODER_SIZE  = 1
    DECODER_SIZE  = 1
    XTRAIN_LENGTH = 7
    YTRAIN_LENGTH = 1
    SEASONALITY   = 7
    
    torch.manual_seed(995)

    datasets, original_series = create_restaurant_datasets(
        Xtrain_length = XTRAIN_LENGTH,
        Ytrain_length = YTRAIN_LENGTH,
        normalization = Normalization.MIN_MAX
    )
    
    fig, axes = plt.subplots(1, 1, figsize=(8,6))
    axes.plot(original_series, 'b-', linewidth=2)
    axes.set_title('Raw Data')
    axes.set_ylabel('vals')
    axes.grid(True, alpha=0.3)
    plt.show()

    loss_results = []
    
    train_size    = int(len(datasets)-SEASONALITY)
    train_dataset = torch.utils.data.Subset(datasets, range(0, train_size))
    test_dataset  = torch.utils.data.Subset(datasets, range(train_size, len(datasets)))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=True)

    torch.manual_seed(42)
    model = TransformerLikeModel(
        embed_size   = EMBED_SIZE,
        encoder_size = ENCODER_SIZE,
        decoder_size = DECODER_SIZE,
        output_len   = YTRAIN_LENGTH,
        num_head_enc = NUM_HEADS,
        num_head_dec_1=NUM_HEADS,
        num_head_dec_2=NUM_HEADS,
        positional_embedding_method="learnable"
    )
    
    model.eval()
    with torch.no_grad():
        X_sample, y_sample = next(iter(test_loader))
        initial_pred = model.forward(X_sample)
        initial_loss = torch.nn.MSELoss()(initial_pred, y_sample).item()
    
    nEpochs = 250
    train_loss, test_loss = train_transformer_model(
        model, 
        epochs = nEpochs,
        teacher_forcing_ratio = 0.85,
        verbose = True,
        train_data_loader = train_loader,
        test_data_loader  = test_loader
    )
    
    loss_results.append({
        'final_train_loss': train_loss,
        'final_test_loss': test_loss,
        'initial_loss': initial_loss
    })
    
    model.eval()

    with torch.no_grad():
        X_plot, y_plot = next(iter(test_loader))
        pred_plot = model.forward(X_plot)
        
        X_np = X_plot[0].squeeze().numpy()
        y_np = y_plot[0].squeeze().numpy()
        y_np = np.atleast_1d(y_np)
        pred_np = pred_plot[0].squeeze().numpy()
        pred_np = np.atleast_1d(pred_np)

        fig, axes = plt.subplots(1, 1, figsize=(9,6))
        axes.plot(range(len(X_np)), X_np, 'b-', label='Input', linewidth=2)
        axes.plot(
            range(len(X_np)-1, len(X_np) + len(y_np)),
            [X_np[-1]] + list(y_np),
            'g-', label='Actual', linewidth=2, marker='o'
        )
        axes.plot(
            range(len(X_np)-1, len(X_np) + len(pred_np)),
            [X_np[-1]] + list(pred_np),
            'r--', label='Predicted', linewidth=2, marker='^'
        )
        axes.set_title('Forecast in Testing')
        axes.legend()
        axes.grid(True, alpha=0.3)
        plt.show()
        
        fig, axes = plt.subplots(1, 1, figsize=(9,6))
        axes.plot(range(len(original_series)), original_series, 'b-', label='Original Series', linewidth=2)
        
        # Plot the test sequence in green
        train_size = int(0.8 * len(original_series))
        axes.plot(
            range(train_size, len(original_series)),
            original_series[train_size:],
            'g-', label='Test Series', linewidth=2
        )
        
        model.eval()
        with torch.no_grad():
            predictions = []
            for i in range(0, len(original_series) - XTRAIN_LENGTH, YTRAIN_LENGTH):
                input_seq = torch.tensor(original_series[i:i + XTRAIN_LENGTH].reshape(-1, 1), dtype=torch.float32).unsqueeze(0)
                pred = model.forward(input_seq).squeeze().numpy()
                pred = np.atleast_1d(pred)
                predictions.extend(pred)
        
        prediction_series = np.array(predictions)
        prediction_series = np.concatenate((np.full(XTRAIN_LENGTH, np.nan), prediction_series))
        
        axes.plot(range(len(prediction_series)), prediction_series, 'r--', label='Predicted Series', linewidth=2)
        axes.set_title('Original vs Predicted Series')
        axes.set_ylabel('vals')
        axes.legend()
        axes.grid(True, alpha=0.3)
        
    print(f"Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
