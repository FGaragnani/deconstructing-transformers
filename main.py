import torch
from src.seca import ScalarExpansionContractiveAutoencoder
from src.train import train_transformer_model
from src.model import TransformerLikeModel
from dataset.dataset import parse_whole_dataset_from_xls, SheetType, PreprocessingTimeSeries, DatasetTimeSeries

from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple, Dict
import numpy as np

class Result:

    def __init__(self, num_models: int):
        self.num_models = num_models
        self.train_loss = [float('inf') for _ in range(num_models)]
        self.test_loss = [float('inf') for _ in range(num_models)]

    def __getitem__(self, idx: int) -> Tuple[float, float]:
        return self.train_loss[idx], self.test_loss[idx]

    def __setitem__(self, idx: int, value: Tuple[float, float]):
        self.train_loss[idx], self.test_loss[idx] = value

def main():
    torch.manual_seed(42)

    OUTPUT_LEN = 1
    EMBED_SIZE = 48
    NUM_HEADS = 4
    ENCODER_SIZE = 1
    DECODER_SIZE = 1

    datasets: List[Tuple[DatasetTimeSeries, DatasetTimeSeries]] = parse_whole_dataset_from_xls("M3C.xls", SheetType.MONTHLY, output_len=OUTPUT_LEN, preprocessing=PreprocessingTimeSeries.MIN_MAX)
    results: Dict[int, Result] = {}

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
        train_loader, test_loader = DataLoader(train_dataset, batch_size=1, shuffle=True), DataLoader(test_dataset, batch_size=1, shuffle=False)
        train_loss, test_loss = train_transformer_model(model=model, epochs=100, train_data_loader=train_loader, test_data_loader=test_loader, verbose=False, pretrain_seca=True)
        results[idx][0] = (train_loss ** 0.5, test_loss ** 0.5)     # RMSE

        clf = RandomForestRegressor(n_estimators=100, random_state=42)
        X_np, y_np = train_dataset.np_datasets
        y_np = y_np.ravel()
        clf.fit(X_np, y_np)
        y_p = clf.predict(X_np)
        train_rmse = np.sqrt(np.mean((y_p - y_np) ** 2))

        X_np, y_np = test_dataset.np_datasets
        y_np = y_np.ravel()
        y_p = clf.predict(X_np)
        test_rmse = np.sqrt(np.mean((y_p - y_np) ** 2))
        results[idx][1] = (train_rmse, test_rmse)

        print(f"Transformer - Train RMSE: {results[idx][0][0]:.4f}, Test RMSE: {results[idx][0][1]:.4f}")
        print(f"Random Forest - Train RMSE: {results[idx][1][0]:.4f}, Test RMSE: {results[idx][1][1]:.4f}")

if __name__ == "__main__":
    main()