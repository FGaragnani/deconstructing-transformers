from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataset.dataset import DatasetTimeSeries, SheetType, PreprocessingTimeSeries
from src.model import TransformerLikeModel
from src.train import train_transformer_model
from sklearn.ensemble import RandomForestRegressor
import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

OUTPUT_LEN = 18
INPUT_LEN = 24
EMBED_SIZE = 12
HEADS_NUM = 2
DROPOUT = 0.20
EPOCHS = 10
TRAIN_SIZE = 0.8
BATCH_SIZE = 32

def get_prediction(n: int) -> Tuple[List[float], List[float], List[float]]:
    import re

    with open("long_series.txt", "r") as f:
        data = f.read()

    # Split into blocks that start with 'Input sequence:'
    parts = data.split('Input sequence:')
    if n + 1 >= len(parts):
        raise IndexError(f"Requested prediction index {n} out of range (found {len(parts)-1} chunks)")

    chunk = parts[n + 1]

    def extract_after(name: str, src: str) -> List[float]:
        idx = src.find(name)
        if idx == -1:
            return []
        start = idx + len(name)
        end_candidates = []
        for marker in ('Input sequence:', 'Target sequence:', 'Predicted sequence:'):
            if marker == name:
                continue
            pos = src.find(marker, start)
            if pos != -1:
                end_candidates.append(pos)
        end = min(end_candidates) if end_candidates else len(src)
        section_text = src[start:end]
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", section_text)
        return [float(x) for x in nums]

    inputs = extract_after('', chunk)
    if not inputs:
        inputs = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", chunk)

    targets = extract_after('Target sequence:', chunk)
    predicted = extract_after('Predicted sequence:', chunk)

    return inputs, targets, predicted

def main():

    folder = 'dataset/long/'
    long_series = []
    for i, filename in enumerate(os.listdir(folder)):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath)
            second_col = df.iloc[:, 1].to_numpy()
            split_idx = int(len(second_col) * TRAIN_SIZE)
            train_dataset = DatasetTimeSeries(second_col[:split_idx], SheetType.OTHER, i, "long", output_len=OUTPUT_LEN, preprocessing=PreprocessingTimeSeries.MIN_MAX)
            test_dataset = DatasetTimeSeries(second_col[split_idx:], SheetType.OTHER, i, "long", output_len=OUTPUT_LEN, preprocessing=PreprocessingTimeSeries.MIN_MAX)
            long_series.append((filename, train_dataset, test_dataset))
            print(f"Loaded {filename} with {len(train_dataset)} training samples and {len(test_dataset)} testing samples.")
    
    """
    for filename, train, test in long_series:
        clf = RandomForestRegressor(n_estimators=250, random_state=42)
        X_np, y_np = train.np_datasets
        start_time = time.time()
        clf.fit(X_np, y_np)
        end_time = time.time()
        tim = end_time - start_time
        y_p_train = clf.predict(X_np)
        train_rmse = np.sqrt(np.mean((y_p_train - y_np) ** 2))    # RMSE
        X_np, y_np = test.np_datasets
        y_p_test = clf.predict(X_np)
        test_rmse = np.sqrt(np.mean((y_p_test - y_np) ** 2))    # RMSE
        print(f"ID: {train.id} - Time for training Forest: {tim :.2f} seconds")
        print(f"Dataset Name: {filename}, Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    """

    """
    for filename, train, test in long_series:
        train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

        model = TransformerLikeModel(
            embed_size=EMBED_SIZE,
            output_len=OUTPUT_LEN,
            max_seq_length=INPUT_LEN,
            num_head_enc=HEADS_NUM,
            num_head_dec_1=HEADS_NUM,
            num_head_dec_2=HEADS_NUM,
            dropout=DROPOUT,
            encoder_size=2,
            decoder_size=2
        )

        start_time = time.time()
        train_loss, test_loss = train_transformer_model(
            model, EPOCHS,
            train_data_loader=train_loader,
            test_data_loader=test_loader,
            verbose=True
        )
        end_time = time.time()
        tim = end_time - start_time

        print(f"Dataset Name: {filename}, Train Loss: {train_loss**0.5}, Test Loss: {test_loss**0.5}, Time for training Transformer: {tim/EPOCHS :.2f} seconds")
    """

    if len(long_series) > 0:
        plt.rcParams.update({
            'font.size': 16,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
        })
        rows, cols = 2, 2
        fig, axs = plt.subplots(rows, cols, figsize=(12, 8))
        axs_flat = axs.flatten()
        for idx, item in enumerate(long_series):
            filename, train_ds, test_ds = item
            clf = RandomForestRegressor(n_estimators=250, random_state=42)
            X_np, y_np = train_ds.np_datasets
            clf.fit(X_np, y_np)
            inp, target, pred = get_prediction(idx)
            inp, target, pred = inp[:-1], target[:-2], pred[:-2]
            pred_rf = clf.predict([inp])
            pred_rf = pred_rf.reshape(-1)
            filename = filename.split('.')[0]
            ax = axs_flat[idx]
            ax.plot(inp, label='series', color='blue')
            ax.plot(range(len(inp), len(inp) + len(target)), target, label='target', color='green')
            ax.plot(range(len(inp), len(inp) + len(pred)), pred, label='transformer', color='red')
            ax.plot(range(len(inp), len(inp) + len(pred_rf)), pred_rf, label='random forest', color='orange')
            ax.set_title(filename)
            ax.legend()
            ax.set_ylim(0.0, 1.0)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()