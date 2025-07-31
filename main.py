import torch
from src.seca import ScalarExpansionContractiveAutoencoder
from src.train import train_transformer_model
from src.model import TransformerLikeModel
from dataset.dataset import parse_whole_dataset_from_xls, SheetType, PreprocessingTimeSeries, DatasetTimeSeries

from torch.utils.data import DataLoader
from typing import List, Tuple

OUTPUT_LEN = 4

def main():
    torch.manual_seed(42)

    datasets: List[Tuple[DatasetTimeSeries, DatasetTimeSeries]] = parse_whole_dataset_from_xls("M3C.xls", SheetType.QUARTERLY, output_len=OUTPUT_LEN, preprocessing=PreprocessingTimeSeries.NORMALIZE)

    best_dataset: Tuple[DatasetTimeSeries, DatasetTimeSeries] = datasets[0]
    best_train_loss, best_test_loss = float('inf'), float('inf')

    seca = ScalarExpansionContractiveAutoencoder(embed_size=8, input_size=1) # Preload SECA so it avoids pretraining in each iteration

    for idx, (train_dataset, test_dataset) in enumerate(datasets):
        print(f"Training on dataset: {train_dataset.category} (ID: {train_dataset.id})")
        print(f"Number of training samples: {len(train_dataset)}, Number of testing samples: {len(test_dataset)}")
    
        model: TransformerLikeModel = TransformerLikeModel(embed_size=8, encoder_size=1, decoder_size=1, output_len=OUTPUT_LEN, num_head_enc=2, positional_embedding_method="learnable", seca=seca)
        train_loader, test_loader = DataLoader(train_dataset, batch_size=1, shuffle=True), DataLoader(test_dataset, batch_size=1, shuffle=False)
        train_loss, test_loss = train_transformer_model(model=model, epochs=30, train_data_loader=train_loader, test_data_loader=test_loader, verbose=False, pretrain_seca=(idx == 0))
        if test_loss < best_test_loss:
            best_train_loss = train_loss
            best_test_loss = test_loss
            best_dataset = (train_dataset, test_dataset)
            print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}\n")

    print(f"Best dataset: {best_dataset[0].category} (ID: {best_dataset[0].id})")
    print(f"Best Train Loss: {best_train_loss:.4f}, Best Test Loss: {best_test_loss:.4f}")

if __name__ == "__main__":
    main()