# config.py
# Configuration file for the DiffRUL project
config = {
    "input_size": 14,  # Number of selected sensor features for C-MAPSS dataset
    "hidden_size": 64,  # Hidden layer size for LSTM/VAE
    "latent_dim": 2,   # Latent dimension for VAE
    "window_size": 30,  # Size of the time window for data processing
    "num_layers": 2,   # Number of LSTM layers
    "bidirectional": True,  # Whether to use bidirectional LSTM
    "dropout_lstm_encoder": 0.1,  # Dropout rate for LSTM encoder
    "dropout_lstm_decoder": 0.1,  # Dropout rate for LSTM decoder
    "dropout_layer_encoder": 0.1,  # Dropout rate for encoder layers
    "dropout_layer_decoder": 0.1,  # Dropout rate for decoder layers
    "regression_dims": 64,  # Dimensions for regression layer
    "dropout_regressor": 0.1,  # Dropout rate for regressor
    "reconstruct": True,  # Whether to enable reconstruction in the model
    "lr": 0.001,  # Learning rate
    "max_epochs": 50,  # Maximum number of training epochs
    "batch_size": 32,  # Batch size for training
    "output_dir": "./output",  # Directory to save output files
    "vae_model_path": "./output/best_vae_model.pt",  # Path to save the best VAE model
    "diff_model_path": "./output/best_diff_model.pt",  # Path to save the best diffusion model
    "noise_steps": 1000,  # Number of noise steps for diffusion
    "beta_start": 0.0001,  # Starting value of beta for diffusion schedule
    "beta_end": 0.02,  # Ending value of beta for diffusion schedule
    "schedule_name": "linear",  # Type of noise schedule
    "residual_channels": 64,  # Number of channels in residual layers
    "residual_layers": 30,  # Number of residual layers
    "dilation_cycle_length": 10,  # Length of dilation cycle
    "KLLoss_weight": 1,  # Weight for KL divergence loss
    "RegLoss_weight": 1,  # Weight for regression loss
    "ReconLoss_weight": 1,  # Weight for reconstruction loss
    "TripletLoss_weight": 10,  # Weight for triplet loss
    "TripletLoss_margin": 0.4,  # Margin for triplet loss
    "TripletLoss_p": 2,  # Norm for triplet loss
    "data_dir": "/workspace/BearingGroup/nasa-cmaps/CMaps"  # Directory of C-MAPSS dataset
}