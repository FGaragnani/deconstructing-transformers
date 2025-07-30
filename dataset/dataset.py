import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, Dataset

from enum import Enum
from typing import Optional, Tuple

class SheetType(Enum):
    YEARLY = 0
    QUARTERLY = 1

    def to_recurrence(self) -> int:
        if self == SheetType.YEARLY:
            return 12
        elif self == SheetType.QUARTERLY:
            return 4
        else:
            raise ValueError("Invalid SheetType")


class PreprocessingTimeSeries(Enum):
    NONE = "none"
    NORMALIZE = "normalize"
    STANDARDIZE = "standardize"

    def apply(self, series: pd.Series) -> pd.Series:
        if self == PreprocessingTimeSeries.NONE:
            return series
        elif self == PreprocessingTimeSeries.NORMALIZE:
            return (series - series.min()) / (series.max() - series.min())
        elif self == PreprocessingTimeSeries.STANDARDIZE:
            return (series - series.mean()) / series.std()
        else:
            raise ValueError("Invalid PreprocessingTimeSeries")


class DatasetTimeSeries(Dataset):
    def __init__(self, series: np.ndarray, sheet_type: SheetType, id: int, category: str, output_len: int, preprocessing: PreprocessingTimeSeries = PreprocessingTimeSeries.NONE):
        self.sheet_type: SheetType = sheet_type
        self.id: int = id
        self.category: str = category

        timesteps: int = sheet_type.to_recurrence()
        self.tensor_dataset = self.create_tensor_dataset(series, timesteps=timesteps, output_len=output_len, preprocessing=preprocessing)

    @staticmethod
    def create_tensor_dataset(series: np.ndarray, timesteps: int, output_len: int, preprocessing: PreprocessingTimeSeries = PreprocessingTimeSeries.NONE) -> TensorDataset:
        series = preprocessing.apply(pd.Series(series)).to_numpy()
        X, y = [], []

        for i in range(len(series) - timesteps - output_len + 1):
            X.append(series[i:i + timesteps])
            y.append(series[i + timesteps:i + timesteps + output_len])

        return TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

    def __len__(self):
        return self.tensor_dataset.__len__()

    def __getitem__(self, idx):
        return self.tensor_dataset[idx]

"""
Parses a dataset from an Excel file.
Args:
    file_path (str): Path to the Excel file.
    sheet_type (SheetType): Type of the sheet (YEARLY or QUARTERLY).
    row (int): Row number to parse (1-based index).
    output_len (int): Length of the output sequence.
    preprocessing (PreprocessingTimeSeries): Preprocessing method to apply to the series.

Returns:
    Optional[Tuple[DatasetTimeSeries, DatasetTimeSeries]]: A tuple containing the training and testing datasets, or None if the row is invalid.
"""
def parse_dataset_from_xls(file_path: str, sheet_type: SheetType, row: int, output_len: int, preprocessing: PreprocessingTimeSeries, split: float = 0.75) -> Tuple[DatasetTimeSeries, DatasetTimeSeries]:
    df = pd.read_excel(file_path, header=None)

    # Start from row 2
    data_rows = df.iloc[1:]

    try:
        data_rows = data_rows.iloc[row - 1:row]
    except IndexError:
        raise ValueError(f"Row {row} is out of bounds for the dataset with {len(df)} rows.")
        

    # Extract index
    index = str(data_rows.iloc[0, 0])[2:] # Since it's like "N k"
    try:
        index = int(index)
    except ValueError:
        print(f"Warning: Index '{index}' is not an integer. Using 0 instead.")
        index = 0

    timesteps = sheet_type.to_recurrence()
    category = str(data_rows.iloc[0, 3])
    series = data_rows.iloc[0, 6:].to_numpy(dtype=np.float32)
    train_series = series[:int(len(series) * split)]
    # This is done in order to employ values from the train_series which are only predicted and not trained upon before
    test_series = np.concatenate([series[int(len(series) * split) - timesteps:int(len(series) * split)], series[int(len(series) * split):]])

    train_dataset = DatasetTimeSeries(train_series, sheet_type, index, category, output_len, preprocessing)
    test_dataset = DatasetTimeSeries(test_series, sheet_type, index, category, output_len, preprocessing)

    return train_dataset, test_dataset