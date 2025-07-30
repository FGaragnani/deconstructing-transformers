import torch
from src.train import train_model
from src.model import TransformerLikeModel
from dataset.dataset import parse_dataset_from_xls, SheetType, PreprocessingTimeSeries

from torch.utils.data import DataLoader

OUTPUT_LEN = 4

def main():
    torch.manual_seed(42)

    model: TransformerLikeModel = TransformerLikeModel(embed_size=24, encoder_size=1, decoder_size=0, output_len=OUTPUT_LEN, num_head_enc=2)
    train_dataset, test_dataset = parse_dataset_from_xls("M3C.xls", SheetType.YEARLY, row=227, output_len=OUTPUT_LEN, preprocessing=PreprocessingTimeSeries.NORMALIZE)
    train_loader, test_loader = DataLoader(train_dataset, batch_size=1, shuffle=True), DataLoader(test_dataset, batch_size=1, shuffle=False)
    train_model(model=model, epochs=25, train_data_loader=train_loader, test_data_loader=test_loader)

if __name__ == "__main__":
    main()