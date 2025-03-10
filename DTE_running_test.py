# test_dte_with_actual_data.py
import torch
from torch.utils.data import DataLoader
from config import config
from DTE_running import train_epoch, valid_epoch
from DTE_model.DTE_network import Encoder, Decoder, TSHAE
from utils.loss import TotalLoss
from dataset import CMAPSDataset  # Import your CMAPSDataset class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset using the actual data from dataset.py
train_dataset = CMAPSDataset(data_dir=config['data_dir'], mode="train", window_size=config['window_size'], return_pairs=True)
valid_dataset = CMAPSDataset(data_dir=config['data_dir'], mode="test", window_size=config['window_size'], return_pairs=True)

# Create DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)

# Initialize the model
encoder = Encoder(config)
decoder = Decoder(config)
model = TSHAE(config, encoder, decoder)
model.to(device)

# Initialize optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
criterion = TotalLoss(config)

# Test training step
try:
    print("Testing train_epoch with actual data...")
    for epoch in range(2):  # Run for a few epochs to check
        train_epoch(config, epoch, model, optimizer, criterion, train_loader, history={})
    print("Train epoch completed successfully.")
except Exception as e:
    print(f"Error during training: {e}")

# Test validation step
try:
    print("Testing valid_epoch with actual data...")
    for epoch in range(2):  # Run for a few epochs to check
        valid_epoch(config, epoch, model, criterion, valid_loader, history={})
    print("Validation epoch completed successfully.")
except Exception as e:
    print(f"Error during validation: {e}")
