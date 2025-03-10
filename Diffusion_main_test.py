import os
import torch
import pytest
from unittest.mock import patch
from Diffusion_main import model_train, model_test
from utils import utils

# Assume config is imported correctly and pre-loaded for testing
from config import config

# Test Diffusion model training
def test_diffusion_model_training():
    """
    Test function for Diffusion Model Training (model_train).
    """
    # Create a mock DataLoader to simulate the training process
    from torch.utils.data import DataLoader
    from dataset import CMAPSDataset

    # Simulate a dataset for training
    train_data_path = config['data_dir']  # Provide a valid data path
    train_dataset = CMAPSDataset(train_data_path, mode='train', window_size=config['window_size'], return_pairs=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # Use mock of the model_vae object for testing purpose
    with patch('torch.load') as mock_load:
        mock_load.return_value = torch.nn.Module()  # Mocking the VAE model

        # Test the model training
        epoch_loss = model_train(config, train_loader)

        # Assert that training completes and loss values are returned
        assert len(epoch_loss) > 0, "Epoch loss should not be empty"

        print(f"Epoch Loss: {epoch_loss[-1]:.4f}")


# Test Diffusion model testing (model_test)
def test_diffusion_model_testing():
    """
    Test function for Diffusion Model Testing (model_test).
    """
    # Create a mock DataLoader for testing
    from torch.utils.data import DataLoader
    from dataset import CMAPSDataset

    # Simulate a dataset for testing
    test_data_path = config['data_dir']  # Provide a valid data path
    test_dataset = CMAPSDataset(test_data_path, mode='test', window_size=config['window_size'], return_pairs=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Mock the model paths and other required files for testing
    with patch('torch.load') as mock_load, patch('utils.save_to_pickle') as mock_save:
        mock_load.return_value = torch.nn.Module()  # Mocking the VAE model and Diffusion model
        
        # Call the model_test function to check the testing pipeline
        output_path = os.path.join(config['output_dir'], 'test_output.pkl')
        model_test(config, test_loader, best_diff_model_path="fake_model_path", output_path=output_path)

        # Check that the pickle saving function was called
        mock_save.assert_called_with(output_path, {})
        print("Model testing completed successfully")


if __name__ == "__main__":
    # Run the tests
    pytest.main()
