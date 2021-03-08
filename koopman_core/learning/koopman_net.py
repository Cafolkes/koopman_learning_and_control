import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KoopmanNet(nn.Module):
    def __init__(self, net_params, standardizer_x=None, standardizer_u=None):
        super(KoopmanNet, self).__init__()
        self.net_params = net_params
        self.standardizer_x = standardizer_x
        self.standardizer_u = standardizer_u

        self.encoder = None
        self.opt_parameters_encoder = []
        self.opt_parameters_dyn_mats = []
        self.x_running_mean = None
        self.x_running_var = None

        if self.net_params['override_C']:
            self.n_tot = int(self.net_params['first_obs_const']) + self.net_params['state_dim'] + \
                         self.net_params['encoder_output_dim']
        else:
            self.n_tot = int(self.net_params['first_obs_const']) + self.net_params['encoder_output_dim']
            assert self.net_params['override_kinematics'] is False, \
                'Not overriding C while overriding kinematics not supported'

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')

    def construct_net(self):
        pass

    def forward(self, data):
        pass

    def send_to(self, device):
        pass

    def process(self, data_x, t, data_u=None, downsample_rate=1):
        pass

    def construct_dyn_mat(self):
        pass

    def loss(self, outputs, labels):
        # output = [x_pred, x_prime_pred, lin_error]
        # labels = [x, x_prime], penalize when lin_error is not zero
        n = self.net_params['state_dim']
        n_z = self.net_params['encoder_output_dim']
        #dt = self.net_params['dt']
        n_override_kinematics = int(n/2)*int(self.net_params['override_kinematics'])

        #x_prime_diff_pred = outputs[:, n_override_kinematics:n]
        #x_prime_diff = labels[:, n_override_kinematics:]
        x_prime_pred = outputs[:, n_override_kinematics:n]
        x_prime = labels[:, n_override_kinematics:n]

        x_prime_diff_pred = outputs[:, n+n_override_kinematics:2*n]
        x_prime_diff = labels[:, n+n_override_kinematics:2*n]

        z_prime_diff_pred = outputs[:, 2*n:2*n+n_z]
        z_prime_diff = outputs[:, 2*n+n_z: 2*n+2*n_z]

        alpha = self.net_params['lin_loss_penalty']
        criterion = nn.MSELoss()

        proj_loss = criterion(x_prime_pred, x_prime)

        #pred_loss = criterion(x_prime_diff_pred, x_prime_diff/dt)
        #pred_loss = criterion(x_prime_pred, x_prime)
        pred_loss = criterion(x_prime_diff_pred, torch.divide(x_prime_diff, self.loss_scaler_x[n_override_kinematics:]))
        #lin_loss = criterion(z_prime_diff_pred, z_prime_diff/dt)/n_z
        lin_loss = criterion(z_prime_diff_pred, z_prime_diff / self.loss_scaler_z)/(n_z/n)
        #lin_loss = criterion(z_prime_diff_pred, z_prime_diff) / n_z
        l1_loss = 0.
        if 'l1_reg' in self.net_params and self.net_params['l1_reg'] > 0:
            l1_reg = self.net_params['l1_reg']
            l1_loss = l1_reg*self.get_l1_norm_()

        tot_loss = proj_loss + pred_loss + alpha*lin_loss + l1_loss
        return tot_loss, pred_loss, alpha*lin_loss

    def construct_encoder_(self):
        input_dim = self.net_params['state_dim']
        hidden_width = self.net_params['encoder_hidden_width']
        hidden_depth = self.net_params['encoder_hidden_depth']
        output_dim = self.net_params['encoder_output_dim']
        activation_type = self.net_params['activation_type']

        if hidden_depth > 0:
            self.encoder_fc_in = nn.Linear(input_dim, hidden_width)
            self.opt_parameters_encoder.append(self.encoder_fc_in.weight)
            self.opt_parameters_encoder.append(self.encoder_fc_in.bias)

            self.encoder_fc_hid = nn.ModuleList()
            for ii in range(1, hidden_depth):
                self.encoder_fc_hid.append(nn.Linear(hidden_width, hidden_width))
                self.opt_parameters_encoder.append(self.encoder_fc_hid[-1].weight)
                self.opt_parameters_encoder.append(self.encoder_fc_hid[-1].bias)

            self.encoder_fc_out = nn.Linear(hidden_width, output_dim)
            self.opt_parameters_encoder.append(self.encoder_fc_out.weight)
            self.opt_parameters_encoder.append(self.encoder_fc_out.bias)

        else:
            self.encoder_fc_out = nn.Linear(input_dim, output_dim)
            self.opt_parameters_encoder.append(self.encoder_fc_out.weight)
            self.opt_parameters_encoder.append(self.encoder_fc_out.bias)

        if activation_type == 'relu':
            self.activation_fn = F.relu
        elif activation_type == 'tanh':
            self.activation_fn = torch.tanh
        elif activation_type == 'sigmoid':
            self.activation_fn = torch.sigmoid
        elif activation_type == 'sin':
            self.activation_fn = torch.sin
        else:
            exit("Error: invalid activation function specified")

    def encode_forward_(self, x):
        if self.net_params['encoder_hidden_depth'] > 0:
            x = self.activation_fn(self.encoder_fc_in(x))
            for layer in self.encoder_fc_hid:
                x = self.activation_fn(layer(x))
        x = self.encoder_fc_out(x)

        return x

    def encode(self, x):
        first_obs_const = int(self.net_params['first_obs_const'])
        override_C = self.net_params['override_C']

        self.eval()
        x_t = torch.from_numpy(x).float()
        if override_C:
            z = np.concatenate((np.ones((x.shape[0], first_obs_const)), x, self.encode_forward_(x_t).detach().numpy()), axis=1)
        else:
            z = np.concatenate((np.ones((x.shape[0], first_obs_const)), self.encode_forward_(x_t).detach().numpy()),  axis=1)

        return z

    def preprocess_data(self, data, standardizer):
        if standardizer is None:
            data_scaled = data
        else:
            data_scaled = np.array([standardizer.transform(d) for d in data])

        return data_scaled
