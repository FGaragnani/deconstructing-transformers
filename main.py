import torch
from src.seca import ScalarExpansionContractiveAutoencoder
from src.train import train_transformer_model
from src.model import TransformerLikeModel
from dataset.dataset import parse_whole_dataset_from_xls, SheetType, PreprocessingTimeSeries, DatasetTimeSeries

from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple, Dict
import numpy as np
import pickle

class Result:

    def __init__(self, num_models: int):
        self.num_models = num_models
        self.train_loss = [float('inf') for _ in range(num_models)]
        self.test_loss = [float('inf') for _ in range(num_models)]
        self.predictions: List[List[float]] = [[] for _ in range(num_models)]

    def __getitem__(self, idx: int) -> Tuple[float, float]:
        return self.train_loss[idx], self.test_loss[idx]

    def __setitem__(self, idx: int, value: Tuple[float, float]):
        self.train_loss[idx], self.test_loss[idx] = value

    def get_predictions(self, idx: int) -> List[float]:
        return self.predictions[idx]
    
    def set_predictions(self, idx: int, preds: List[float]):
        self.predictions[idx] = preds

def main():
    torch.manual_seed(42)

    OUTPUT_LEN = 18
    EMBED_SIZE = 10
    NUM_HEADS = 2
    ENCODER_SIZE = 1
    DECODER_SIZE = 1
    BATCH_SIZE = 32

    datasets: List[Tuple[DatasetTimeSeries, DatasetTimeSeries]] = parse_whole_dataset_from_xls("M3C.xls", SheetType.MONTHLY, output_len=OUTPUT_LEN, preprocessing=PreprocessingTimeSeries.MIN_MAX)
    results: List[Result] = []

    for idx, (train_dataset, test_dataset) in enumerate(datasets):
        results[idx] = Result(num_models=2)

        print(f"Training on dataset: {train_dataset.category} (ID: {train_dataset.id})")
        print(f"Number of training samples: {len(train_dataset)}, Number of testing samples: {len(test_dataset)}")
    
        model: TransformerLikeModel = TransformerLikeModel(
            embed_size=EMBED_SIZE,
            encoder_size=ENCODER_SIZE,
            decoder_size=DECODER_SIZE,
            output_len=OUTPUT_LEN,
            num_head_enc=NUM_HEADS,
            num_head_dec_1=NUM_HEADS,
            num_head_dec_2=NUM_HEADS
        )
        train_loader, test_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        train_loss, test_loss = train_transformer_model(
            model=model, 
            epochs=400, 
            train_data_loader=train_loader, 
            test_data_loader=test_loader, 
            verbose=True, 
            pretrain_seca=True
        )
        results[idx][0] = (train_loss ** 0.5, test_loss ** 0.5)     # RMSE

        all_preds = []
        for X_batch, _ in test_loader:
            preds = model(X_batch)
            all_preds.append(preds.detach().cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        results[idx].set_predictions(0, all_preds.flatten().tolist())

        clf = RandomForestRegressor(n_estimators=100, random_state=42)
        X_np, y_np = train_dataset.np_datasets
        print("Train X shape:", train_dataset.np_datasets[0].shape)
        print("Train y shape:", train_dataset.np_datasets[1].shape)
        print("Test X shape:", test_dataset.np_datasets[0].shape)
        print("Test y shape:", test_dataset.np_datasets[1].shape)
        clf.fit(X_np, y_np)
        y_p = clf.predict(X_np)
        train_rmse = np.sqrt(np.mean((y_p - y_np) ** 2))    # RMSE

        X_np, y_np = test_dataset.np_datasets
        y_p = clf.predict(X_np)
        test_rmse = np.sqrt(np.mean((y_p - y_np) ** 2))    # RMSE
        results[idx][1] = (train_rmse, test_rmse)
        results[idx].set_predictions(1, y_p.flatten().tolist())

        print(f"Transformer - Train RMSE: {results[idx][0][0]:.4f}, Test RMSE: {results[idx][0][1]:.4f}")
        print(f"Random Forest - Train RMSE: {results[idx][1][0]:.4f}, Test RMSE: {results[idx][1][1]:.4f}")

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()