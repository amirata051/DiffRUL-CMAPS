
import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.latent_dim = config['latent_dim']
        self.num_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        self.num_directions = 2 if self.bidirectional else 1
        self.p_lstm = config['dropout_lstm_encoder']
        self.p = config['dropout_layer_encoder']

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.p_lstm,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.fc_mean = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(
                in_features=self.num_directions * self.hidden_size,
                out_features=self.latent_dim)
        )

        self.fc_log_var = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(
                in_features=self.num_directions * self.hidden_size,
                out_features=self.latent_dim)
        )

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(var.device)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        """
        :param x: [B, seq_len, fea_dim]
        :return:
            z: [B, 2]
            mean: [B, 2]
            log_var: [B, 2]
        """
        batch_size = x.shape[0]
        _, (h_n, _) = self.lstm(x)

        h_n = h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        if self.bidirectional:
            h = torch.cat((h_n[-1, -2, :, :], h_n[-1, -1, :, :]), dim=1)
        else:
            h = h_n[-1, -1, :, :]
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        return z, mean, log_var


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.latent_dim = config['latent_dim']
        self.num_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        self.window_size = config['window_size']
        self.p_lstm = config['dropout_lstm_decoder']
        self.p_dropout_layer = config['dropout_layer_decoder']
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm_to_hidden = nn.LSTM(
            input_size=self.latent_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.p_lstm,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        self.dropout_layer = nn.Dropout(self.p_dropout_layer)

        self.lstm_to_output = nn.LSTM(
            input_size=self.num_directions * self.hidden_size,
            hidden_size=self.input_size,
            batch_first=True
        )

    def forward(self, z):
        """
        :param z: [B, 2]
        :return: [B, seq_len, fea_dim]
        """
        latent_z = z.unsqueeze(1).repeat(1, self.window_size, 1)  # [B, seq_len, 2]
        out, _ = self.lstm_to_hidden(latent_z)
        out = self.dropout_layer(out)
        out, _ = self.lstm_to_output(out)
        return out


class TSHAE(nn.Module):
    def __init__(self, config, encoder, decoder):
        super(TSHAE, self).__init__()

        self.p = config['dropout_regressor']
        self.regression_dims = config['regression_dims']

        self.decode_mode = config['reconstruct']
        if self.decode_mode:
            assert isinstance(decoder, nn.Module), "You should to pass a valid decoder"
            self.decoder = decoder

        self.encoder = encoder

        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, self.regression_dims),
            nn.Tanh(),
            nn.Dropout(self.p),
            nn.Linear(self.regression_dims, 1)
        )

    def forward(self, x):
        """
        :param x: [B, seq_len, fea_dim]
        :return:
            y_hat: [B, 1]
            z: [B, 2]
            mean: [B, 2]
            log_var: [B, 2]
            x_hat: [B, seq_len, fea_dim]
        """
        z, mean, log_var = self.encoder(x)
        y_hat = self.regressor(z)
        if self.decode_mode:
            x_hat = self.decoder(z)
            return y_hat, z, mean, log_var, x_hat

        return y_hat, z, mean, log_var