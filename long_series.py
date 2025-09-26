from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataset.dataset import DatasetTimeSeries, SheetType, PreprocessingTimeSeries
from src.model import TransformerLikeModel
from src.train import train_transformer_model
import os
import pandas as pd

OUTPUT_LEN = 18
INPUT_LEN = 24
EMBED_SIZE = 8
HEADS_NUM = 2
DROPOUT = 0.1
EPOCHS = 100
TRAIN_SIZE = 0.8
BATCH_SIZE = 64

def main():

    folder = 'dataset/long/'
    long_series = []
    for i, filename in enumerate(os.listdir(folder)):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath)
            second_col = df.iloc[:, [1]].to_numpy()
            second_col = second_col[:, 0]
            train_dataset = DatasetTimeSeries(second_col[:int(len(second_col) * TRAIN_SIZE)], SheetType.OTHER, i, "long", output_len=OUTPUT_LEN, preprocessing=PreprocessingTimeSeries.MIN_MAX)
            test_dataset = DatasetTimeSeries(second_col[int(len(second_col) * TRAIN_SIZE):], SheetType.OTHER, i, "long", output_len=OUTPUT_LEN, preprocessing=PreprocessingTimeSeries.MIN_MAX)
            long_series.append((filename, train_dataset, test_dataset))

    model = TransformerLikeModel(
        embed_size=EMBED_SIZE,
        output_len=OUTPUT_LEN,
        max_seq_length=INPUT_LEN,
        num_head_enc=HEADS_NUM,
        num_head_dec_1=HEADS_NUM,
        num_head_dec_2=HEADS_NUM,
        dropout=DROPOUT,
    )

    for filename, train, test in long_series:
        train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

        train_loss, test_loss = train_transformer_model(
            model, EPOCHS,
            train_data_loader=train_loader,
            test_data_loader=test_loader,
            verbose=True
        )

        print(f"Dataset Name: {filename}, Train Loss: {train_loss**0.5}, Test Loss: {test_loss**0.5}")

if __name__ == "__main__":
    main()