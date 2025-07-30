from src.train import train_model
from src.model import TransformerLikeModel
from dataset.dataset import parse_dataset_from_xls, SheetType, PreprocessingTimeSeries

from torch.utils.data import DataLoader

def main():
    train_dataset, test_dataset = parse_dataset_from_xls("M3C.xls", SheetType.YEARLY, row=227, output_len=6, preprocessing=PreprocessingTimeSeries.NONE)
    train_loader, test_loader = DataLoader(train_dataset, batch_size=1, shuffle=True), DataLoader(test_dataset, batch_size=1, shuffle=False)
    model: TransformerLikeModel = TransformerLikeModel(embed_size=32, encoder_size=2, decoder_size=2, output_len=6)
    train_model(model=model, epochs=20, train_data_loader=train_loader, test_data_loader=test_loader)

if __name__ == "__main__":
    main()