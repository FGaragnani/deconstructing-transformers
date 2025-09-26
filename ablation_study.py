import torch
import pandas as pd
from typing import List, Tuple
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset.dataset import parse_whole_dataset_from_xls, SheetType, PreprocessingTimeSeries, DatasetTimeSeries
from src.model import TransformerLikeModel
from src.train import train_transformer_model

def main():
    torch.manual_seed(42)

    OUTPUT_LEN = 18
    INPUT_LEN = 24
    EMBED_SIZE = 36
    NUM_HEADS = 4
    ENCODER_SIZE = 1
    DECODER_SIZE = 1
    BATCH_SIZE = 16
    EPOCHS = 100
    DROPOUT = 0.05

    df = pd.read_csv('results/res_monthly.csv')
    indices = [
        1652, 1546, 1894, 2047, 2255, 2492, 2594, 2658, 2737, 2758, 2817, 2823
    ]

    datasets: List[Tuple[DatasetTimeSeries, DatasetTimeSeries]] = parse_whole_dataset_from_xls("M3C.xls", SheetType.MONTHLY, input_len=INPUT_LEN, output_len=OUTPUT_LEN, preprocessing=PreprocessingTimeSeries.MIN_MAX)
    datasets = [dataset for dataset in datasets if dataset[0].id in indices]

    models = [
        TransformerLikeModel(
            embed_size=EMBED_SIZE,
            output_len=OUTPUT_LEN,
            max_seq_length=INPUT_LEN,
            num_head_enc=NUM_HEADS,
            num_head_dec_1=NUM_HEADS,
            num_head_dec_2=NUM_HEADS,
            encoder_size=ENCODER_SIZE,
            decoder_size=DECODER_SIZE,
            dropout=DROPOUT
        ),                                  # Normal model
        TransformerLikeModel(
            embed_size=EMBED_SIZE,
            output_len=OUTPUT_LEN,
            max_seq_length=INPUT_LEN,
            num_head_enc=1,
            num_head_dec_1=NUM_HEADS,
            num_head_dec_2=NUM_HEADS,
            encoder_size=ENCODER_SIZE,
            decoder_size=DECODER_SIZE,
            dropout=DROPOUT
        ),                                   # Single Head Encoder
        TransformerLikeModel(
            embed_size=EMBED_SIZE,
            output_len=OUTPUT_LEN,
            max_seq_length=INPUT_LEN,
            num_head_enc=NUM_HEADS,
            num_head_dec_1=NUM_HEADS,
            num_head_dec_2=NUM_HEADS,
            encoder_size=ENCODER_SIZE,
            decoder_size=DECODER_SIZE,
            dropout=DROPOUT,
            hidden_ff_size_enc=0
        ),                                   # No FF in Encoder
        TransformerLikeModel(
            embed_size=EMBED_SIZE,
            output_len=OUTPUT_LEN,
            max_seq_length=INPUT_LEN,
            num_head_enc=NUM_HEADS,
            num_head_dec_1=NUM_HEADS,
            num_head_dec_2=NUM_HEADS,
            encoder_size=ENCODER_SIZE,
            decoder_size=DECODER_SIZE,
            dropout=DROPOUT,
            enc_use_addnorm=[False, True]
        ),                                   # No first Add&Norm Encoder
        TransformerLikeModel(
            embed_size=EMBED_SIZE,
            output_len=OUTPUT_LEN,
            max_seq_length=INPUT_LEN,
            num_head_enc=NUM_HEADS,
            num_head_dec_1=NUM_HEADS,
            num_head_dec_2=NUM_HEADS,
            encoder_size=ENCODER_SIZE,
            decoder_size=DECODER_SIZE,
            dropout=DROPOUT,
            enc_use_addnorm=[True, False]
        ),                                    # No second Add&Norm Encoder
        TransformerLikeModel(
            embed_size=1,
            output_len=OUTPUT_LEN,
            max_seq_length=INPUT_LEN,
            num_head_enc=1,
            num_head_dec_1=1,
            num_head_dec_2=1,
            encoder_size=ENCODER_SIZE,
            decoder_size=DECODER_SIZE,
            dropout=DROPOUT,
            seca=nn.Identity() # type: ignore
        ),                                  # No SECA
        TransformerLikeModel(
            embed_size=EMBED_SIZE,
            output_len=OUTPUT_LEN,
            max_seq_length=INPUT_LEN,
            num_head_enc=NUM_HEADS,
            num_head_dec_1=NUM_HEADS,
            num_head_dec_2=NUM_HEADS,
            encoder_size=ENCODER_SIZE,
            decoder_size=DECODER_SIZE,
            dropout=DROPOUT,
            use_pe=False
        ),                              # No Positional Encoding
        TransformerLikeModel(
            embed_size=EMBED_SIZE,
            output_len=OUTPUT_LEN,
            max_seq_length=INPUT_LEN,
            num_head_enc=NUM_HEADS,
            num_head_dec_1=NUM_HEADS,
            num_head_dec_2=NUM_HEADS,
            encoder_size=ENCODER_SIZE,
            decoder_size=DECODER_SIZE,
            dropout=DROPOUT,
            use_out=False
        )                                   # No Output Layer
    ]

    desc = ["Standard", "Single Head Encoder", "No FF Encoder", "No first Add&Norm", "No second Add&Norm", "No SECA", "No Positional Encoding", "No Output Layer"]

    for i, (train_dataset, test_dataset) in enumerate(datasets):
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(str(indices[i]) + " & ", end="")

        for model in models:

            train_loss, test_loss = train_transformer_model(
                model=model, 
                epochs=EPOCHS, 
                train_data_loader=train_loader, 
                test_data_loader=test_loader, 
                verbose=False,
            )

            print(f"{test_loss ** 0.5:.4f} & ", end="")
        print(" \\\\")

    
if __name__ == "__main__":
    main()