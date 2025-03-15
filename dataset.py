# dataset.py
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

class CMAPSDataset(Dataset):
    def __init__(self, data_dir, mode="train", window_size=30, return_pairs=False):
        # Initialize dataset with C-MAPSS data directory, mode, window size, and pair flag
        self.data_dir = data_dir
        self.mode = mode  # "train" or "test" mode
        self.window_size = window_size  # Size of the sliding window
        self.return_pairs = return_pairs  # Whether to return positive/negative pairs
        self.data = []  # List to store processed data
        self.ruls = []  # List to store Remaining Useful Life (RUL) values
        self.ids = []  # List to store sample IDs

        # Determine file paths based on mode
        if self.mode == "train":
            data_file = os.path.join(data_dir, "train_FD001.txt")
        else:
            data_file = os.path.join(data_dir, "test_FD001.txt")
            rul_file = os.path.join(data_dir, "RUL_FD001.txt")

        # Read the data file into a DataFrame
        df = pd.read_csv(data_file, delim_whitespace=True, header=None)
        df.columns = ["unit", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]

        # Load RUL values for test mode
        if self.mode == "test":
            rul_df = pd.read_csv(rul_file, header=None)
            rul_values = rul_df[0].values

        # Select important sensor columns based on C-MAPSS literature
        sensor_cols = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
                       "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21"]
        units = df["unit"].unique()

        # Process data for each unit
        for unit in units:
            unit_data = df[df["unit"] == unit][sensor_cols].values
            cycles = df[df["unit"] == unit]["cycle"].values

            # Normalize the data
            unit_data_min = unit_data.min(axis=0)
            unit_data_max = unit_data.max(axis=0)
            unit_data = (unit_data - unit_data_min) / (unit_data_max - unit_data_min + 1e-10)  # Avoid division by zero

            # Calculate RUL
            if self.mode == "train":
                max_cycle = cycles.max()
                unit_ruls = max_cycle - cycles
            else:
                unit_rul = rul_values[unit - 1]
                unit_ruls = unit_rul + (cycles.max() - cycles)

            # Create sliding windows
            for i in range(0, len(unit_data) - window_size + 1, window_size):
                window = unit_data[i:i + window_size]
                if window.shape[0] != window_size:
                    continue
                self.data.append(window)
                self.ruls.append(unit_ruls[i])
                self.ids.append(f"unit_{unit}_window_{i}")

        self.data = np.array(self.data)
        self.ruls = np.array(self.ruls)

        # Save preprocessed data
        preprocessed_dir = os.path.join('output', 'preprocessed')
        os.makedirs(preprocessed_dir, exist_ok=True)
        np.save(os.path.join(preprocessed_dir, f'preprocessed_data_{self.mode}.npy'), self.data)
        np.save(os.path.join(preprocessed_dir, f'preprocessed_ruls_{self.mode}.npy'), self.ruls)

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get a sample by index
        x = torch.FloatTensor(self.data[idx])
        y = torch.FloatTensor([self.ruls[idx]])
        if self.return_pairs:
            pos_idx, neg_idx = self._get_pairs(idx)
            pos_x = torch.FloatTensor(self.data[pos_idx])
            neg_x = torch.FloatTensor(self.data[neg_idx])
            return x, pos_x, neg_x, y
        return x, y

    def _get_pairs(self, idx):
        # Generate positive and negative pairs based on RUL similarity
        rul = self.ruls[idx]
        similar_ruls = np.where(np.abs(self.ruls - rul) <= 5)[0]
        dissimilar_ruls = np.where((np.abs(self.ruls - rul) > 5) & (np.abs(self.ruls - rul) <= 15))[0]
        pos_idx = np.random.choice(similar_ruls) if len(similar_ruls) > 0 else idx
        neg_idx = np.random.choice(dissimilar_ruls) if len(dissimilar_ruls) > 0 else idx
        return pos_idx, neg_idx

    def get_run(self, engine_id):
        # Retrieve data and RUL for a specific engine
        idx = self.ids.index(f"unit_{engine_id}_window_0")
        return self.data[idx], self.ruls[idx]

class PreprocessedDataset(Dataset):
    def __init__(self, preprocessed_data_path, preprocessed_ruls_path, window_size=30, return_pairs=False):
        # Initialize with preprocessed data paths
        self.data = np.load(preprocessed_data_path)
        self.ruls = np.load(preprocessed_ruls_path)
        self.window_size = window_size
        self.return_pairs = return_pairs
        self.ids = [f"sample_{i}" for i in range(len(self.data))]

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get a sample by index
        x = torch.FloatTensor(self.data[idx])
        y = torch.FloatTensor([self.ruls[idx]])
        if self.return_pairs:
            pos_idx, neg_idx = self._get_pairs(idx)
            pos_x = torch.FloatTensor(self.data[pos_idx])
            neg_x = torch.FloatTensor(self.data[neg_idx])
            return x, pos_x, neg_x, y
        return x, y

    def _get_pairs(self, idx):
        # Generate positive and negative pairs based on RUL similarity
        rul = self.ruls[idx]
        similar_ruls = np.where(np.abs(self.ruls - rul) <= 5)[0]
        dissimilar_ruls = np.where((np.abs(self.ruls - rul) > 5) & (np.abs(self.ruls - rul) <= 15))[0]
        pos_idx = np.random.choice(similar_ruls) if len(similar_ruls) > 0 else idx
        neg_idx = np.random.choice(dissimilar_ruls) if len(dissimilar_ruls) > 0 else idx
        return pos_idx, neg_idx

    def get_run(self, engine_id):
        # Retrieve data and RUL for a specific engine
        idx = self.ids.index(engine_id)
        return self.data[idx], self.ruls[idx]

if __name__ == "__main__":
    # Main block for testing the dataset
    from config import config
    dataset = CMAPSDataset(config['data_dir'], mode="train", window_size=config['window_size'], return_pairs=True)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    for batch in dataloader:
        x, pos_x, neg_x, y = batch
        print("Batch shape:", x.shape, pos_x.shape, neg_x.shape, y.shape)
        break