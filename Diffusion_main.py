
"""
Diffision Model train
"""

import os
import json
import pickle
import torch
import logging
import argparse
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import utils
from Diffusion_model.Diff_network import DiffWave, EMA
from Diffusion_model.ddpm import Diffusion as DDPMDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_train(config, train_loader):

    # load DTE Model
    model_vae = torch.load(config['vae_model_path'])
    model_vae.to(device)
    model_vae.eval()
    for param in model_vae.parameters():
        param.requires_grad = False

    # Diff model initialization
    model = DiffWave(config)
    model.to(device)
    diffusion = DDPMDiffusion(config['noise_steps'], config['beta_start'], config['beta_end'], config['schedule_name'], device)

    # Exponential Moving Average (EMA)
    ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)  # EMA model

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.MSELoss()

    # start training
    model.train()
    best_epoch = 0
    best_loss = float('inf')

    epoch_loss = []
    for epoch in tqdm(range(config['max_epochs']), desc='Training'):

        batch_loss = []
        for batch_idx, data in enumerate(train_loader):
            pairs_mode = train_loader.dataset.return_pairs

            if pairs_mode:
                x, pos_x, neg_x, true_rul = data
            else:
                x, true_rul = data

            x = x.to(device)
            with torch.no_grad():
                predicted_rul, z = model_vae(x)[:2]
                conditioner = z.to(device)

            time = diffusion.sample_time_steps(x.shape[0]).to(device)
            noisy_x, noise = diffusion.noise_images(x=x, time=time)

            predicted_noise = model(noisy_x, time, conditioner)
            loss = criterion(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA
            ema.step_ema(ema_model=ema_model, model=model)

            batch_loss.append(loss.item())

        epoch_loss.append(np.mean(batch_loss))
        logging.info("Epoch:{}/{}, Train Loss:{:.4f}".format(epoch, config['max_epochs'], np.mean(batch_loss)))
        print("Epoch:{}/{}, Train Loss:{:.4f}".format(epoch, config['max_epochs'], np.mean(batch_loss)))

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(config['output_dir'], 'best_diff_model_{}.pt'.format(epoch + 1))
            ema_save_path = os.path.join(config['output_dir'], 'ema_best_diff_model_{}.pt'.format(epoch + 1))

            torch.save({'state_dict': model.state_dict()}, save_path)
            torch.save({'state_dict': ema_model.state_dict()}, ema_save_path)

    return epoch_loss


def model_test(config, train_loader, best_diff_model_path, output_path):

    # load DTE Model
    model_vae = torch.load(config['vae_model_path'])
    model_vae.to(device)
    model_vae.eval()

    model_diff = DiffWave(config)
    checkpoint = utils.load_model(best_diff_model_path)
    model_diff.load_state_dict(checkpoint['state_dict'])
    model_diff.to(device)
    model_diff.eval()

    diffusion = DDPMDiffusion(config['noise_steps'], config['beta_start'], config['beta_end'], config['schedule_name'], device)

    sample_result = {}
    engine_ids = train_loader.dataset.ids
    for engine_id in tqdm(engine_ids, desc='Sample'):

        with torch.no_grad():
            x, y = train_loader.dataset.get_run(engine_id)
            x = torch.tensor(x).to(device)  # [B, seq_len, fea_dim]
            y = torch.tensor(y).to(device) # [B, 1]

            x = torch.tensor(x, dtype=torch.float32)

            predicted_rul, z = model_vae(x)[:2]
            sample_x = diffusion.sample(config, model_diff, z)  # [B, L, feature_dim]

        x = x.detach().cpu().numpy()
        sample_x = sample_x.detach().cpu().numpy()
        sample_result[engine_id] = (x, sample_x)

    utils.save_to_pickle(output_path, sample_result)