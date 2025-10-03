import torch
import pandas as pd
from src.seca import ScalarExpansionContractiveAutoencoder
from src.train import train_transformer_model
from src.model import TransformerLikeModel
from dataset.dataset import parse_whole_dataset_from_xls, SheetType, PreprocessingTimeSeries, DatasetTimeSeries

from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple, Dict
import numpy as np
import pickle
import random
import time

configurations = [
    {
        "ENCODER_SIZE": 1,
        "DECODER_SIZE": 1,
        "EMBED_SIZE": 8,
    },
    {
        "ENCODER_SIZE": 2,
        "DECODER_SIZE": 2,
        "EMBED_SIZE": 8,
    },
    {
        "ENCODER_SIZE": 3,
        "DECODER_SIZE": 3,
        "EMBED_SIZE": 8,
    },
    {
        "ENCODER_SIZE": 4,
        "DECODER_SIZE": 4,
        "EMBED_SIZE": 8,
    },
]

def main():
    torch.manual_seed(42)

    OUTPUT_LEN = 18
    INPUT_LEN = 24
    NUM_HEADS = 4
    BATCH_SIZE = 64
    EPOCHS = 10
    DROPOUT = 0.05

    df = pd.read_csv('results/res_monthly.csv')
    # indices = df.id.tolist()
    indices = [2047]
    tim = 0

    datasets: List[Tuple[DatasetTimeSeries, DatasetTimeSeries]] = parse_whole_dataset_from_xls("M3C.xls", SheetType.MONTHLY, input_len=INPUT_LEN, output_len=OUTPUT_LEN, preprocessing=PreprocessingTimeSeries.MIN_MAX)
    datasets = [dataset for dataset in datasets if dataset[0].id in indices]

    for train_dataset, test_dataset in datasets:

        print(f"Training on dataset: {train_dataset.category} (ID: {train_dataset.id})")
        # print(f"Number of training samples: {len(train_dataset)}, Number of testing samples: {len(test_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        for configuration in configurations:
        
            model: TransformerLikeModel = TransformerLikeModel(
                embed_size=configuration["EMBED_SIZE"],
                encoder_size=configuration["ENCODER_SIZE"],
                decoder_size=configuration["DECODER_SIZE"],
                output_len=OUTPUT_LEN,
                num_head_enc=NUM_HEADS,
                num_head_dec_1=NUM_HEADS,
                num_head_dec_2=NUM_HEADS,
                dropout=DROPOUT,
                max_seq_length=INPUT_LEN
            )
            print(f"""
    Configuration - Encoder Size: {configuration["ENCODER_SIZE"]}, Decoder Size: {configuration["DECODER_SIZE"]}, Embedding Size: {configuration["EMBED_SIZE"]}
""")
            print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


            start_time = time.time()

            train_loss, test_loss = train_transformer_model(
                model=model, 
                epochs=EPOCHS, 
                train_data_loader=train_loader, 
                test_data_loader=test_loader, 
                verbose=False, 
                pretrain_seca=True
            )

            end_time = time.time()
            tim = end_time - start_time
            print(f"ID: {train_dataset.id} - Time for training Transformer: {tim / EPOCHS:.2f} seconds")
            print(f"Train Loss: {train_loss**0.5}, Test Loss: {test_loss**0.5}")     # RMSE

            # Measure inference time
            inference_times = []
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    start_inference = time.time()
                    _ = model(batch[0])
                    end_inference = time.time()
                    inference_times.append(end_inference - start_inference)

            avg_inference_time = sum(inference_times) / len(inference_times)
            print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")

if __name__ == "__main__":
    main()