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

def get_airline_passenger_data() -> np.ndarray:
    """Load and preprocess the airline passenger dataset."""
    flights = sns.load_dataset('flights')      
    series = flights.pivot(index='year', columns='month', values='passengers')
    series = series.values.flatten()
    return series

def create_airline_datasets(sequence_length: int = 12, prediction_length: int = 4, normalization: Normalization = Normalization.STANDARD) -> Tuple[TensorDataset, np.ndarray]:
    """
        Create training and testing datasets from the airline passenger data.

        :param sequence_length: Length of the input sequences
        :param prediction_length: Length of the prediction sequences
        :param normalization: Normalization method to apply

        :return: A tuple containing the training/testing datasets and the original series
    """
    original_series = get_airline_passenger_data()
    original_series = normalization.normalize(original_series)

    datasets = []
    for i in range(0, len(original_series) - sequence_length - prediction_length + 1):
        seq = original_series[i:i + sequence_length].reshape(-1, 1)
        target = original_series[i + sequence_length:i + sequence_length + prediction_length].reshape(-1, 1)
        datasets.append((torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)))

    datasets = TensorDataset(
        torch.stack([item[0] for item in datasets]),
        torch.stack([item[1] for item in datasets])
    )

    return datasets, original_series

def main():
    NUM_HEADS = 2
    EMBED_SIZE = 8
    ENCODER_SIZE = 1
    DECODER_SIZE = 1
    SEQUENCE_LENGTH = 12
    PREDICTION_LENGTH = 4
    
    torch.manual_seed(42)

    datasets, original_series = create_airline_datasets(
        sequence_length=SEQUENCE_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        normalization=Normalization.MIN_MAX
    )
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    years = np.arange(1949, 1961, 1/12)[:len(original_series)]
    axes[0].plot(years, original_series, 'b-', linewidth=2)
    axes[0].set_title('Original Data (1949-1960)')
    axes[0].set_ylabel('Passengers')
    axes[0].grid(True, alpha=0.3)

    loss_results = []
    
    train_size = int(0.8 * len(datasets))
    train_dataset = torch.utils.data.Subset(datasets, range(0, train_size))
    test_dataset = torch.utils.data.Subset(datasets, range(train_size, len(datasets)))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    torch.manual_seed(42)
    model = TransformerLikeModel(
        embed_size=EMBED_SIZE,
        encoder_size=ENCODER_SIZE,
        decoder_size=DECODER_SIZE,
        output_len=PREDICTION_LENGTH,
        num_head_enc=NUM_HEADS,
        num_head_dec_1=NUM_HEADS,
        num_head_dec_2=NUM_HEADS,
        positional_embedding_method="learnable"
    )
    
    model.eval()
    with torch.no_grad():
        X_sample, y_sample = next(iter(test_loader))
        initial_pred = model.forward(X_sample)
        initial_loss = torch.nn.MSELoss()(initial_pred, y_sample).item()
    
    train_loss, test_loss = train_transformer_model(
        model, 
        epochs=200, 
        teacher_forcing_ratio=0.85, 
        verbose=True,
        train_data_loader=train_loader, 
        test_data_loader=test_loader
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
        pred_np = pred_plot[0].squeeze().numpy()
        
        axes[1].plot(range(len(X_np)), X_np, 'b-', label='Input', linewidth=2)
        axes[1].plot(
            range(len(X_np)-1, len(X_np) + len(y_np)),
            [X_np[-1]] + list(y_np),
            'g-', label='Actual', linewidth=2, marker='o'
        )
        axes[1].plot(
            range(len(X_np)-1, len(X_np) + len(pred_np)),
            [X_np[-1]] + list(pred_np),
            'r--', label='Predicted', linewidth=2, marker='^'
        )
        axes[1].set_title('Forecast in Testing')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(range(len(original_series)), original_series, 'b-', label='Original Series', linewidth=2)
        
        # Plot the test sequence in green
        train_size = int(0.8 * len(original_series))
        axes[2].plot(
            range(train_size, len(original_series)),
            original_series[train_size:],
            'g-', label='Test Series', linewidth=2
        )
        
        model.eval()
        with torch.no_grad():
            predictions = []
            for i in range(0, len(original_series) - SEQUENCE_LENGTH, PREDICTION_LENGTH):
                input_seq = torch.tensor(original_series[i:i + SEQUENCE_LENGTH].reshape(-1, 1), dtype=torch.float32).unsqueeze(0)
                pred = model.forward(input_seq).squeeze().numpy()
                predictions.extend(pred)
        
        prediction_series = np.array(predictions)
        prediction_series = np.concatenate((np.full(SEQUENCE_LENGTH, np.nan), prediction_series))
        
        axes[2].plot(range(len(prediction_series)), prediction_series, 'r--', label='Predicted Series', linewidth=2)
        axes[2].set_title('Original vs Predicted Series')
        axes[2].set_ylabel('Passengers')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
    print(f"Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
