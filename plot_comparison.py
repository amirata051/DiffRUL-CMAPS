# plot_comparison.py
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from config import config
from utils.utils import load_from_pickle

def compare_data(real, sampled):
    mse = np.mean((real - sampled) ** 2)
    mean_diff = np.mean(np.abs(real - sampled))
    var_real = np.var(real)
    var_sampled = np.var(sampled)
    var_ratio = var_sampled / (var_real + 1e-10)
    print(f"MSE: {mse:.4f}, Mean Diff: {mean_diff:.4f}, Var Ratio: {var_ratio:.4f}")

def plot_comparison():
    # Load real data
    df = pd.read_csv(os.path.join(config['data_dir'], "train_FD001.txt"), delim_whitespace=True, header=None)
    df.columns = ["unit", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
    unit_1 = df[df["unit"] == 1]
    print(f"Number of cycles in unit_1: {len(unit_1)}")
    if len(unit_1) < 2:
        print("Unit_1 has too few cycles, trying Unit_2...")
        unit_1 = df[df["unit"] == 2]
        print(f"Number of cycles in unit_2: {len(unit_1)}")
    sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
               "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
               "sensor_17", "sensor_20", "sensor_21"]
    real_data = unit_1[sensors].values
    real_data_normalized = (real_data - real_data.min(axis=0)) / (real_data.max(axis=0) - real_data.min(axis=0) + 1e-10)

    # Load augmented data
    augmented_data = load_from_pickle(os.path.join(config['output_dir'], 'augmented_data.pkl'))

    # Collect all windows for unit_1 (assuming sample_0 to sample_6 correspond to unit_1)
    num_windows = (len(unit_1) + config['window_size'] - 1) // config['window_size']  # Ceiling division
    print(f"Expected number of windows for unit_1: {num_windows}")
    unit_1_samples = []
    unit_1_real = []
    for i in range(num_windows):
        key = f"sample_{i}"
        if key not in augmented_data:
            print(f"Key {key} not found in augmented_data. Available keys: {list(augmented_data.keys())}")
            break
        x, sample_x = augmented_data[key]
        unit_1_real.append(x)
        unit_1_samples.append(sample_x)

    # Concatenate all windows
    if not unit_1_samples:
        raise ValueError("No samples found for unit_1 in augmented data!")
    unit_1_real = torch.cat(unit_1_real, dim=0).numpy()  # Shape: [total_cycles, 14]
    unit_1_samples = torch.cat(unit_1_samples, dim=0).numpy()  # Shape: [total_cycles, 14]

    # Match lengths
    min_len = min(len(real_data_normalized), len(unit_1_samples))
    real_data_normalized = real_data_normalized[:min_len]
    unit_1_samples = unit_1_samples[:min_len]

    # Debug: Print data stats
    print(f"Real data shape: {real_data_normalized.shape}, min: {real_data_normalized.min()}, max: {real_data_normalized.max()}")
    print(f"Sampled data shape: {unit_1_samples.shape}, min: {unit_1_samples.min()}, max: {unit_1_samples.max()}")

    # Plot
    sensor_names = ["T24", "T30", "P30", "Nf", "Nc", "Ps30", "phi", "NRf", "NRc", "BPR", "htBleed", "W31", "W32", "W32"]
    cycles = unit_1["cycle"].values[:min_len]
    
    for i, sensor in enumerate(sensors):
        plt.figure(figsize=(10, 5))
        plt.plot(cycles, real_data_normalized[:, i], label="Real Data", color="blue")
        plt.plot(cycles, unit_1_samples[:, i], label="Sampled Data", color="orange")
        plt.xlabel("Cycle")
        plt.ylabel(sensor_names[i])
        plt.title(f"Degradation Trajectory of {sensor_names[i]}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plot_{sensor_names[i]}.png")
        plt.close()
        compare_data(real_data_normalized[:, i], unit_1_samples[:, i])

if __name__ == "__main__":
    plot_comparison()