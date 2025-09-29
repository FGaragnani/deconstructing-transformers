import optuna
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from ristoranti import RistorantiDataset
from src.model import TransformerLikeModel
from src.train import train_transformer_model

PREDICTION_LENGTH = 7
ENCODER_SIZE = 1
DECODER_SIZE = 1


# Hyperparameter search-space definitions
NUM_HEADS_CHOICES = [1, 2, 4]
SEQUENCE_LENGTHS = [i for i in range(5, 15)]
BATCH_SIZES = [4, 8, 16, 32]
EMBED_SIZES = [8, 12, 16]
EPOCHS = 25
DROPOUT_RANGE = (0.0, 0.30)
LR_RANGE = (1e-3, 1e-2)
TRAIN_PERCENTAGE = 0.75
SEED = 42

def objective(trial: optuna.trial.Trial):

    # Hyperparameters
    embed_size = trial.suggest_categorical("embed_size", EMBED_SIZES)
    sequence_length = trial.suggest_categorical("sequence_length", SEQUENCE_LENGTHS)
    num_heads = trial.suggest_categorical("num_head_enc", NUM_HEADS_CHOICES)
    dropout = trial.suggest_float("dropout", DROPOUT_RANGE[0], DROPOUT_RANGE[1])
    learning_rate = trial.suggest_float("lr", LR_RANGE[0], LR_RANGE[1])
    batch_size = trial.suggest_categorical("batch_size", BATCH_SIZES)

    # Dataset
    df = pd.read_csv('ristorantiGTrend.csv')
    series = df.iloc[:, 1].values.astype(np.float32)
    series = series[:35]
    min_val = np.min(series)
    max_val = np.max(series)
    series = (series - min_val) / (max_val - min_val)
    dataset = RistorantiDataset(series, sequence_length, PREDICTION_LENGTH)

    torch.manual_seed(SEED)
    output_len = PREDICTION_LENGTH
    epochs = EPOCHS 

    split_idx = int(TRAIN_PERCENTAGE * len(dataset))
    train_set = torch.utils.data.Subset(dataset, range(split_idx))
    test_set = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = TransformerLikeModel(
        embed_size=embed_size,
        encoder_size=ENCODER_SIZE,
        decoder_size=DECODER_SIZE,
        input_size=1,
        num_head_enc=num_heads,
        num_head_dec_1=num_heads,
        num_head_dec_2=num_heads,
        dropout=dropout,
        output_len=output_len,
        positional_embedding_method="learnable",
        max_seq_length=120
    )

    train_loss, val_loss = train_transformer_model(
        model,
        epochs=epochs,
        train_data_loader=train_loader,
        test_data_loader=test_loader,
        verbose=False,
        teacher_forcing_ratio=0.9,
        check_losses=False,
        early_stopping=False,
        learning_rate=learning_rate
    )

    return val_loss ** 0.5

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name="transformer_timeseries")
    study.optimize(objective, n_trials=50, n_jobs=1)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")