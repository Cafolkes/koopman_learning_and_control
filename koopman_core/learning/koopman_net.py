import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KoopmanNet(nn.Module):
    def __init__(self, net_params, standardizer=None):
        super(KoopmanNet, self).__init__()
        self.net_params = net_params
        self.standardizer = standardizer

        self.encoder = None
        self.optimization_parameters = []

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')

    def construct_net(self):
        pass

    def forward(self, data):
        pass

    def send_to(self, device):
        pass

    def process(self, data, t, downsample_rate=1, train_data=False):
        pass

    def preprocess_data(self, x, train_data, u=None):
        pass

    def construct_dyn_mat(self):
        pass

    def loss(self, outputs, labels, validation=False):
        # output = [x_pred, x_prime_pred, lin_error]
        # labels = [x, x_prime], penalize when lin_error is not zero
        n = self.net_params['state_dim']
        n_z = self.net_params['encoder_output_dim']
        dt = self.net_params['dt']
        override_kinematics = self.net_params['override_kinematics']

        if override_kinematics:
            x_prime_diff_pred = outputs[:, int(n/2):n]
            x_prime_diff = labels[:, int(n/2):]

        else:
            x_prime_diff_pred = outputs[:, :n]
            x_prime_diff = labels

        z_prime_diff_pred = outputs[:, n:n + n_z]
        z_prime_diff = outputs[:, n + n_z:]

        alpha = self.net_params['lin_loss_penalty']
        criterion = nn.MSELoss()

        pred_loss = criterion(x_prime_diff_pred, x_prime_diff/dt)
        lin_loss = criterion(z_prime_diff_pred, z_prime_diff/dt)

        if validation:
            total_loss = pred_loss + 0.1*lin_loss  #TODO: Think about best validation loss (multistep ?)
        else:
            total_loss = pred_loss + alpha*lin_loss

            if 'l1_reg' in self.net_params and self.net_params['l1_reg'] > 0:
                l1_reg = self.net_params['l1_reg']
                total_loss += l1_reg * torch.norm(self.koopman_fc_drift.weight.flatten(), p=1)

        return total_loss

    def construct_encoder_(self):
        input_dim = self.net_params['state_dim']
        hidden_width = self.net_params['encoder_hidden_width']
        hidden_depth = self.net_params['encoder_hidden_depth']
        output_dim = self.net_params['encoder_output_dim']

        if hidden_depth > 0:
            self.encoder_fc_in = nn.Linear(input_dim, hidden_width)
            self.optimization_parameters.append(self.encoder_fc_in.weight)
            self.optimization_parameters.append(self.encoder_fc_in.bias)
            self.encoder_fc_hid = []
            for ii in range(1, hidden_depth):
                self.encoder_fc_hid.append(nn.Linear(hidden_width, hidden_width))
                self.optimization_parameters.append(self.encoder_fc_hid[-1].weight)
                self.optimization_parameters.append(self.encoder_fc_hid[-1].bias)
            self.encoder_fc_out = nn.Linear(hidden_width, output_dim)
            self.optimization_parameters.append(self.encoder_fc_out.weight)
            self.optimization_parameters.append(self.encoder_fc_out.bias)
        else:
            self.encoder_fc_out = nn.Linear(input_dim, output_dim)
            self.optimization_parameters.append(self.encoder_fc_out.weight)
            self.optimization_parameters.append(self.encoder_fc_out.bias)

    def encode_forward_(self, x):
        if self.net_params['encoder_hidden_depth'] > 0:
            x = F.relu(self.encoder_fc_in(x))
            for layer in self.encoder_fc_hid:
                x = F.relu(layer(x))
        x = self.encoder_fc_out(x)

        return x

    def encode(self, x):
        first_obs_const = int(self.net_params['first_obs_const'])
        self.eval()
        x_t = torch.from_numpy(x).float()
        z = np.concatenate((np.ones((x.shape[0], first_obs_const)), x, self.encode_forward_(x_t).detach().numpy()), axis=1)

        return z
