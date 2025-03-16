# plot_comparison.py
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from config import config
from utils.utils import load_from_pickle

def compare_data(real, sampled):
    mse = np.mean((real - sampled) ** 2)
    mean_diff = np.mean(np.abs(real - sampled))
    var_real = np.var(real)
    var_sampled = np.var(sampled)
    var_ratio = var_sampled / (var_real + 1e-10)  # Add small epsilon to avoid division by zero
    print(f"MSE: {mse:.4f}, Mean Diff: {mean_diff:.4f}, Var Ratio: {var_ratio:.4f}")

def plot_comparison():
    # Load real data
    df = pd.read_csv(os.path.join(config['data_dir'], "train_FD001.txt"), delim_whitespace=True, header=None)
    df.columns = ["unit", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
    unit_1 = df[df["unit"] == 1]
    sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
               "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
               "sensor_17", "sensor_20", "sensor_21"]
    real_data = unit_1[sensors].values
    real_data_normalized = (real_data - real_data.min(axis=0)) / (real_data.max(axis=0) - real_data.min(axis=0) + 1e-10)

    # Load augmented data
    augmented_data = load_from_pickle(os.path.join(config['output_dir'], 'augmented_data.pkl'))
    engine_id = list(augmented_data.keys())[0]  # Use the first available key
    x, sample_x = augmented_data[engine_id]

    # Check and normalize data
    if len(sample_x) == 0 or not np.any(sample_x):
        raise ValueError(f"Sampled data for {engine_id} is empty or all zeros!")
    if len(real_data_normalized) == 0 or not np.any(real_data_normalized):
        raise ValueError("Real data is empty or all zeros!")

    # Match lengths
    min_len = min(len(real_data_normalized), len(sample_x))
    real_data_normalized = real_data_normalized[:min_len]
    sample_x = sample_x[:min_len]

    # Plot
    sensor_names = ["T24", "T30", "P30", "Nf", "Nc", "Ps30", "phi", "NRf", "NRc", "BPR", "htBleed", "W31", "W32", "W32"]
    cycles = np.linspace(0, min_len - 1, min_len)  # Create a range based on the minimum length
    
    for i, sensor in enumerate(sensors):
        plt.figure(figsize=(10, 5))
        plt.plot(cycles, real_data_normalized[:, i], label="Real Data", color="blue")
        plt.plot(cycles, sample_x[:, i], label="Sampled Data", color="orange")
        plt.xlabel("Cycle")
        plt.ylabel(sensor_names[i])
        plt.title(f"Degradation Trajectory of {sensor_names[i]}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plot_{sensor_names[i]}.png")
        plt.close()
        compare_data(real_data_normalized[:, i], sample_x[:, i])

if __name__ == "__main__":
    plot_comparison()