import torch
from src.train import train_model
from src.model import TransformerLikeModel
from dataset.dataset import parse_whole_dataset_from_xls, SheetType, PreprocessingTimeSeries, DatasetTimeSeries

from torch.utils.data import DataLoader
from typing import List, Tuple

OUTPUT_LEN = 4

def main():
    torch.manual_seed(42)

    datasets: List[Tuple[DatasetTimeSeries, DatasetTimeSeries]] = parse_whole_dataset_from_xls("M3C.xls", SheetType.QUARTERLY, output_len=OUTPUT_LEN, preprocessing=PreprocessingTimeSeries.NORMALIZE)
    model: TransformerLikeModel = TransformerLikeModel(embed_size=24, encoder_size=1, decoder_size=0, output_len=OUTPUT_LEN, num_head_enc=2)

    for idx, (train_dataset, test_dataset) in enumerate(datasets):
        print(f"Training on dataset: {train_dataset.category} (ID: {train_dataset.id})")
        print(f"Number of training samples: {len(train_dataset)}, Number of testing samples: {len(test_dataset)}")
    
        train_loader, test_loader = DataLoader(train_dataset, batch_size=1, shuffle=True), DataLoader(test_dataset, batch_size=1, shuffle=False)
        train_model(model=model, epochs=25, train_data_loader=train_loader, test_data_loader=test_loader, verbose=False)

if __name__ == "__main__":
    main()