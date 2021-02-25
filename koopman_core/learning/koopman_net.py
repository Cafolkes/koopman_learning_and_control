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
        self.x_running_mean = None
        self.x_running_var = None


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
        dt = self.net_params['dt']
        n_override_kinematics = int(n/2)*int(self.net_params['override_kinematics'])

        x_prime_diff_pred = outputs[:, n_override_kinematics:n]
        x_prime_diff = labels[:, n_override_kinematics:]

        z_prime_diff_pred = outputs[:, n:n+n_z]
        z_prime_diff = outputs[:, n+n_z: n+2*n_z]
        z_norm = outputs[:, n+2*n_z:]

        alpha = self.net_params['lin_loss_penalty']
        criterion = nn.MSELoss()

        pred_loss = criterion(x_prime_diff_pred, x_prime_diff/dt)
        lin_loss = criterion(z_prime_diff_pred, z_prime_diff/dt)/n_z
        #norm_loss = criterion(z_norm, torch.ones_like(z_norm))

        l1_loss = 0.
        if 'l1_reg' in self.net_params and self.net_params['l1_reg'] > 0:
            l1_reg = self.net_params['l1_reg']
            l1_loss = l1_reg*self.get_l1_norm_()

        return pred_loss + alpha*lin_loss + l1_loss #+ 1e-8*norm_loss

    def construct_encoder_(self):
        input_dim = self.net_params['state_dim']
        hidden_width = self.net_params['encoder_hidden_width']
        hidden_depth = self.net_params['encoder_hidden_depth']
        output_dim = self.net_params['encoder_output_dim']
        activation_type = self.net_params['activation_type']

        if hidden_depth > 0:
            self.encoder_fc_in = nn.Linear(input_dim, hidden_width)
            self.encoder_fc_hid = nn.ModuleList()
            for ii in range(1, hidden_depth):
                self.encoder_fc_hid.append(nn.Linear(hidden_width, hidden_width))
            self.encoder_fc_out = nn.Linear(hidden_width, output_dim)

        else:
            self.encoder_fc_out = nn.Linear(input_dim, output_dim)

        #self.encoder_output_norm = nn.BatchNorm1d(output_dim)
        #self.encoder_output_norm = nn.LayerNorm(output_dim)  #TODO: Evaluate output norm

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

    def encode_forward_(self, x, update_stats=True):
        if self.net_params['encoder_hidden_depth'] > 0:
            x = self.activation_fn(self.encoder_fc_in(x))
            for layer in self.encoder_fc_hid:
                x = self.activation_fn(layer(x))
        x = self.encoder_fc_out(x)
        #x = self.encoder_output_norm(x)
        #x = self.batch_normalize_(x, update_stats)

        return x

    def batch_normalize_(self, x, update_stats, eps=1e-5, momentum=0.1):
        if update_stats:
            #self.x_mean = torch.mean(x, 0).reshape(1,-1)
            self.x_var = torch.std(x, 0).reshape(1,-1)

            if self.x_running_mean is None and self.x_running_var is None:
                #self.x_running_mean = self.x_mean
                self.x_running_var = self.x_var
            else:
                #self.x_running_mean = self.x_running_mean * (1-momentum) + self.x_mean * momentum
                self.x_running_var = self.x_running_var * (1-momentum) + self.x_var * momentum

        if self.training:
            #x = torch.divide(x - self.x_mean, torch.sqrt(self.x_var + eps))
            x = torch.divide(x, torch.sqrt(self.x_var + eps))
        else:
            #x = torch.divide(x - self.x_running_mean, torch.sqrt(self.x_running_var + eps))
            #x = torch.divide(x, torch.sqrt(self.x_running_var + eps))  # TODO: Evaluate if any scaling is necessary..
            pass

        return x

    def encode(self, x):
        first_obs_const = int(self.net_params['first_obs_const'])
        self.eval()
        x_t = torch.from_numpy(x).float()
        z = np.concatenate((np.ones((x.shape[0], first_obs_const)), x, self.encode_forward_(x_t, update_stats=False).detach().numpy()), axis=1)

        return z

    def preprocess_data(self, data, standardizer):
        if standardizer is None:
            data_scaled = data
        else:
            data_scaled = np.array([standardizer.transform(d) for d in data])

        return data_scaled
