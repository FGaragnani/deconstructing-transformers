import torch
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
from tqdm import tqdm

OUTPUT_LEN = 18
INPUT_LEN = 24
EMBED_SIZE = 12
HEADS_NUM = 2
DROPOUT = 0.20
EPOCHS = 10
TRAIN_SIZE = 0.8
BATCH_SIZE = 32

def main():

    torch.manual_seed(42)
    np.random.seed(42)

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
            'font.size': 17,
            'axes.titlesize': 19,
            'axes.labelsize': 17,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'legend.fontsize': 15,
        })
        rows, cols = 2, 1
        fig, axs = plt.subplots(rows, cols, figsize=(12, 8))
        axs_flat = axs.flatten()
        for idx, item in enumerate(long_series):

            if idx not in [0, 1]:
                continue
            filename, train_ds, test_ds = item
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

            whole_series = np.concatenate([train_ds.original_series, test_ds.original_series])
            target = whole_series[-OUTPUT_LEN:]
            inp = whole_series[:-OUTPUT_LEN]

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

            train_transformer_model(
                model, EPOCHS + 10 * (idx),
                train_data_loader=train_loader,
                test_data_loader=test_loader,
                verbose=True, learning_rate=2e-3
            )

            # Transformer
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            transformer_input = inp[-INPUT_LEN:].reshape(1, INPUT_LEN, 1)
            transformer_input_tensor = torch.tensor(transformer_input, dtype=torch.float32).to(device)
            with torch.no_grad():
                pred_transformer = model(transformer_input_tensor)
                pred_transformer = pred_transformer.cpu().numpy().reshape(-1)

            """
            model = torch.nn.Transformer(d_model=1, nhead=1)
            criterion = torch.nn.MSELoss()
            for epoch in tqdm(range(EPOCHS + 10 * (idx)), desc=f"Training Transformer for {filename}"):
                model.train()
                for batch in train_loader:
                    src, tgt = batch
                    src = src.permute(1, 0, 2)  # (S, N, E)
                    tgt_input = tgt[:, :-1].permute(1, 0, 2)  # (T, N, E)
                    tgt_output = tgt[:, 1:].permute(1, 0, 2)  # (T, N, E)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    optimizer.zero_grad()
                    output = model(src, tgt_input)
                    loss = criterion(output, tgt_output)
                    loss.backward()
                    optimizer.step()
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            transformer_input = inp[-INPUT_LEN:].reshape(1, INPUT_LEN, 1)
            transformer_input_tensor = torch.tensor(transformer_input, dtype=torch.float32).to(device)
            with torch.no_grad():
                pred_transformer = model(transformer_input_tensor)
                pred_transformer = pred_transformer.cpu().numpy().reshape(-1)
                test_loss = criterion(torch.tensor(pred_transformer), torch.tensor(target)).item()
            print(f"Dataset Name: {filename}, Test Loss: {test_loss**0.5}")
            """

            # Random Forest
            clf = RandomForestRegressor(n_estimators=250, random_state=42)
            X_np, y_np = train_ds.np_datasets
            clf.fit(X_np, y_np)
            classifier_input = inp[-INPUT_LEN:].reshape(1, -1)
            pred_rf = clf.predict(classifier_input)
            pred_rf = pred_rf.reshape(-1)
            filename = filename.split('.')[0]
            ax = axs_flat[idx % 2]
            ax.plot(inp, label='series', color='blue')
            ax.plot(range(len(inp), len(inp) + len(target)), target, label='target', color='green')
            ax.plot(range(len(inp), len(inp) + len(pred_transformer)), pred_transformer, label='transformer', color='red')
            ax.plot(range(len(inp), len(inp) + len(pred_rf)), pred_rf, label='random forest', color='orange')
            ax.set_xlabel('Time Step', fontsize=20)
            ax.set_ylabel('Standardized Value', fontsize=20)
            ax.set_title(filename, fontsize=23)
            ax.legend()
            ax.set_ylim(0.0, 1.0)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()