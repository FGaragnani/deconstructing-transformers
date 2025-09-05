import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, Dataset

from enum import Enum
from typing import Tuple, List

INPUT_LEN = 12

class SheetType(Enum):
    YEARLY = 0
    QUARTERLY = 1
    MONTHLY = 2

    def to_recurrence(self) -> int:
        if self == SheetType.YEARLY:
            return 12 * 2
        elif self == SheetType.QUARTERLY:
            return 4 * 2
        elif self == SheetType.MONTHLY:
            return INPUT_LEN
        else:
            raise ValueError("Invalid SheetType")


class PreprocessingTimeSeries(Enum):
    NONE = "none"
    MIN_MAX = "min_max"
    STANDARDIZE = "standardize"

    def apply(self, series: pd.Series) -> pd.Series:
        if self == PreprocessingTimeSeries.NONE:
            return series
        elif self == PreprocessingTimeSeries.MIN_MAX:
            range_val = series.max() - series.min()
            if range_val == 0:
                return pd.Series([0.0] * len(series))  # Return zeros if no variation
            return (series - series.min()) / range_val
        elif self == PreprocessingTimeSeries.STANDARDIZE:
            std_val = series.std()
            if std_val == 0:
                return pd.Series([0.0] * len(series))  # Return zeros if no variation
            return (series - series.mean()) / std_val
        else:
            raise ValueError("Invalid PreprocessingTimeSeries")


class DatasetTimeSeries(Dataset):
    def __init__(self, series: np.ndarray, sheet_type: SheetType, id: int, category: str, output_len: int, preprocessing: PreprocessingTimeSeries = PreprocessingTimeSeries.NONE):
        self.sheet_type: SheetType = sheet_type
        self.id: int = id
        self.category: str = category

        timesteps: int = sheet_type.to_recurrence()
        self.tensor_dataset: TensorDataset = self.create_tensor_dataset(series, timesteps=timesteps, output_len=output_len, preprocessing=preprocessing) # type: ignore
        self.np_datasets: Tuple[np.ndarray, np.ndarray] = self.create_tensor_dataset(series, timesteps=timesteps, output_len=output_len, preprocessing=preprocessing, numpy=True) # type: ignore

    @staticmethod
    def create_tensor_dataset(series: np.ndarray, timesteps: int, output_len: int, preprocessing: PreprocessingTimeSeries = PreprocessingTimeSeries.NONE, numpy: bool = False) -> TensorDataset | Tuple[np.ndarray, np.ndarray]:
        series = preprocessing.apply(pd.Series(series)).to_numpy()
        X, y = [], []

        for i in range(len(series) - timesteps - output_len + 1):
            X.append(series[i:i + timesteps])
            y.append(series[i + timesteps:i + timesteps + output_len])

        X = np.array(X)
        y = np.array(y)
        if not numpy:
            return TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        return X, y

    def __len__(self):
        return self.tensor_dataset.__len__()

    def __getitem__(self, idx):
        return (self.tensor_dataset[idx][0].unsqueeze(1), self.tensor_dataset[idx][1].unsqueeze(1))

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
def parse_dataset_from_df(df: pd.DataFrame, sheet_type: SheetType, row: int, output_len: int, preprocessing: PreprocessingTimeSeries, split: float = 0.75) -> Tuple[DatasetTimeSeries, DatasetTimeSeries]:
    # Start from row 2
    data_rows = df.iloc[1:]

    try:
        data_rows = data_rows.iloc[row - 1:row]
    except IndexError:
        raise ValueError(f"Row {row} is out of bounds for the dataset with {len(df)} rows.")
        

    # Extract index
    index = str(data_rows.iloc[0, 0])[1:] # Since it's like "Nk" - with N letter and k number
    try:
        index = int(index)
    except ValueError:
        print(f"Warning: Index '{index}' is not an integer. Using 0 instead.")
        index = 0

    timesteps = sheet_type.to_recurrence()
    category = str(data_rows.iloc[0, 3]).rstrip()
    series = data_rows.iloc[0, 6:].to_numpy(dtype=np.float32)
    series = series[~np.isnan(series)]

    min_needed = timesteps + output_len
    if len(series) >= min_needed:
        train_series = series[: len(series) - output_len]
        test_series = series[-min_needed:]
    else:
        train_series = series[:int(len(series) * split)]
        test_series = np.concatenate([
            series[int(len(series) * split) - timesteps:int(len(series) * split)],
            series[int(len(series) * split):]
        ])

    train_dataset = DatasetTimeSeries(train_series, sheet_type, index, category, output_len, preprocessing)
    test_dataset = DatasetTimeSeries(test_series, sheet_type, index, category, output_len, preprocessing)

    return train_dataset, test_dataset

def parse_dataset_from_xls(file_path: str, sheet_type: SheetType, row: int, output_len: int, preprocessing: PreprocessingTimeSeries, split: float = 0.75) -> Tuple[DatasetTimeSeries, DatasetTimeSeries]:
    df = pd.read_excel(file_path, sheet_name=sheet_type.value, header=None)
    return parse_dataset_from_df(df, sheet_type, row, output_len, preprocessing, split)

def parse_whole_dataset_from_xls(file_path: str, sheet_type: SheetType, output_len: int, preprocessing: PreprocessingTimeSeries, split: float = 0.75, numpy: bool = False, input_len: int = 12) -> List[Tuple[DatasetTimeSeries, DatasetTimeSeries]]:
    
    INPUT_LEN = input_len
    
    df = pd.read_excel(file_path, sheet_name=sheet_type.value, header=None)
    datasets = []

    for row in range(2, len(df)):
        try:
            train_dataset, test_dataset = parse_dataset_from_df(df, sheet_type, row, output_len, preprocessing, split)
            if len(test_dataset) > 0 and len(train_dataset) > 0:
                datasets.append((train_dataset, test_dataset))
        except IndexError as e:
            print(f"Skipping row {row} due to error: {e}")

    return datasets