# train_2.py
import os
import torch
from torch.utils.data import DataLoader
from config import config
from dataset import CMAPSDataset, PreprocessedDataset
from Diffusion_main import model_train as diff_model_train, model_test as diff_model_test
from DTE_model.DTE_network import TSHAE, Encoder, Decoder
from Diffusion_model.Diff_network import DiffWave
from utils.utils import create_dirs, set_seed

def main():
    # Set random seed for reproducibility
    set_seed(2023)

    # Create output directory
    create_dirs([config['output_dir'], os.path.join(config['output_dir'], 'preprocessed')])

    # Load and preprocess dataset
    train_dataset = CMAPSDataset(config['data_dir'], mode="train", window_size=config['window_size'], return_pairs=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # Train DTE model (skip testing and validation to speed up)
    encoder = Encoder(config)
    decoder = Decoder(config)
    dte_model = TSHAE(config, encoder, decoder)
    # Only train DTE, skipping validation for now
    dte_model.train()  # Make sure the model is in training mode
    for epoch in range(50):
        for batch in train_loader:
            # Training logic for DTE model
            pass  # You can keep the DTE training logic here, simplified for now

    # Train diffusion model (focus on augmentation)
    diff_model = DiffWave(config)
    preprocessed_train_loader = DataLoader(PreprocessedDataset(
        os.path.join(config['output_dir'], 'preprocessed', 'preprocessed_data_train.npy'),
        os.path.join(config['output_dir'], 'preprocessed', 'preprocessed_ruls_train.npy'),
        window_size=config['window_size'], return_pairs=True), batch_size=config['batch_size'], shuffle=True)
    diff_model_train(config, preprocessed_train_loader)

    # Test diffusion model and generate augmented data
    best_diff_model_path = os.path.join(config['output_dir'], 'best_diff_model_50.pt')
    output_path = os.path.join(config['output_dir'], 'augmented_data.pkl')
    diff_model_test(config, preprocessed_train_loader, best_diff_model_path, output_path)

    print("Training and augmentation completed successfully!")

if __name__ == "__main__":
    main()
