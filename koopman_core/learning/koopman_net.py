import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KoopmanNet(nn.Module):
    def __init__(self, net_params):
        super(KoopmanNet, self).__init__()
        self.net_params = net_params

        self.encoder = None
        self.construct_net()

        n = self.net_params['state_dim']
        n_encode = self.net_params['encoder_output_dim']
        first_obs_const = self.net_params['first_obs_const']
        if first_obs_const:
            self.C = torch.cat((torch.zeros((n,1)), torch.eye(n), torch.zeros((n, n_encode))), 1)
        else:
            self.C = torch.cat((torch.eye(n), torch.zeros((n, n_encode))), 1)

    def construct_net(self):
        self.construct_encoder_()

        m = self.net_params['ctrl_dim']
        n_tot = self.net_params['state_dim'] + self.net_params['encoder_output_dim'] + int(self.net_params['first_obs_const'])
        self.koopman_fc_drift = nn.Linear(n_tot, n_tot, bias=False)
        self.koopman_fc_act = nn.Linear(m*n_tot, n_tot, bias=False)

    def forward(self, data):
        # data = [x, u, x_prime]
        # output = [x_pred, x_prime_pred, lin_error]
        n = self.net_params['state_dim']
        m = self.net_params['ctrl_dim']
        n_multistep = self.net_params['n_multistep']
        first_obs_const = self.net_params['first_obs_const']

        x_vec = data[:, :n*n_multistep]
        u_vec = data[:, n*n_multistep:n*n_multistep+m*n_multistep]
        x_prime_vec = data[:, n*n_multistep+m*n_multistep:]

        # Define autoencoder networks:
        x = x_vec[:, :n]
        if first_obs_const:
            z = torch.cat((torch.ones((x.shape[0], 1)), x, self.encode_forward_(x)), 1)
        else:
            z = torch.cat((x, self.encode_forward_(x)), 1)

        # Define linearity networks:
        n_tot = n + self.net_params['encoder_output_dim'] + int(self.net_params['first_obs_const'])
        if first_obs_const:
            z = torch.cat((torch.ones((x.shape[0], 1)), x, self.encode_forward_(x)), 1)
            z_prime = torch.cat([torch.cat(
                (torch.ones((x_prime_vec.shape[0],1)),
                 x_prime_vec[:, n * ii:n * (ii + 1)], self.encode_forward_(x_prime_vec[:, n * ii:n * (ii + 1)])), 1) for
                ii in range(n_multistep)], 1)
        else:
            z_prime = torch.cat([torch.cat(
                (x_prime_vec[:, n * ii:n * (ii + 1)], self.encode_forward_(x_prime_vec[:, n * ii:n * (ii + 1)])), 1) for
                                 ii in range(n_multistep)], 1)

        #z_prime_pred = torch.cat([torch.matmul(z, torch.transpose(torch.pow(self.koopman_fc_drift.weight + torch.eye(n_tot), ii+1), 0, 1)) for ii in range(n_multistep)], 1)
        z_u = torch.cat([torch.transpose(torch.mul(torch.transpose(z,0,1), u), 0,1) for u in torch.transpose(u_vec,0,1)], 1)
        z_prime_pred = z + self.koopman_fc_drift(z) + self.koopman_fc_act(z_u)
        # TODO: Implement multi-step prediction

        # Define prediction network:
        x_prime_pred = torch.cat([torch.matmul(z_prime_pred[:,n_tot*ii:n_tot*(ii+1)], torch.transpose(self.C, 0, 1)) for ii in range(n_multistep)], 1)

        outputs = torch.cat((x_prime_pred, z_prime_pred, z_prime), 1)

        return outputs

    def loss(self, outputs, labels):
        # output = [x_pred, x_prime_pred, lin_error]
        # labels = [x, x_prime], penalize when lin_error is not zero

        n = self.net_params['state_dim']
        n_tot = n + self.net_params['encoder_output_dim'] + int(self.net_params['first_obs_const'])
        n_multistep = self.net_params['n_multistep']
        x_prime_pred, x_prime = outputs[:, :n*n_multistep], labels
        z_prime_pred, z_prime = outputs[:, n*n_multistep:n*n_multistep+n_tot*n_multistep], outputs[:, n*n_multistep+n_tot*n_multistep:]

        alpha = self.net_params['lin_loss_penalty']
        criterion = nn.MSELoss()

        pred_loss = criterion(x_prime_pred, x_prime)/n_multistep
        lin_loss = criterion(z_prime_pred, z_prime)/n_multistep

        total_loss = pred_loss + alpha * lin_loss
        if 'l1_reg' in self.net_params and self.net_params['l1_reg'] > 0:  # TODO: Verify correct l1-regularization
            l1_reg = self.net_params['l1_reg']
            total_loss += l1_reg * torch.norm(self.koopman_fc_drift.weight.flatten(), p=1)

        return total_loss

    def construct_encoder_(self):
        input_dim = self.net_params['state_dim']
        hidden_dim = self.net_params['encoder_hidden_dim']
        output_dim = self.net_params['encoder_output_dim']

        if len(hidden_dim) > 0:
            self.encoder_fc_in = nn.Linear(input_dim, hidden_dim[0])
            self.encoder_fc_hid = []
            for ii in range(1, len(hidden_dim)):
                self.encoder_fc_hid.append(nn.Linear(hidden_dim[ii - 1], hidden_dim[ii]))
            self.encoder_fc_out = nn.Linear(hidden_dim[-1], output_dim)
        else:
            self.encoder_fc_out = nn.Linear(input_dim, output_dim)

    def encode_forward_(self, x):
        if len(self.net_params['encoder_hidden_dim']) > 0:
            x = F.relu(self.encoder_fc_in(x))
            for layer in self.encoder_fc_hid:
                x = F.relu(layer(x))
        x = self.encoder_fc_out(x)

        return x

    def encode(self, x):
        first_obs_const = self.net_params['first_obs_const']
        self.eval()
        x_t = torch.from_numpy(x).float()

        if first_obs_const:
            z = np.concatenate((torch.ones((x.shape[0], 1)), x, self.encode_forward_(x_t).detach().numpy()), axis=1)
        else:
            z = np.concatenate((x, self.encode_forward_(x_t).detach().numpy()), axis=1)

        return z

'''
    def construct_decoder_(self):
        input_dim = self.net_params['encoder_output_dim']
        hidden_dim = self.net_params['decoder_hidden_dim']
        output_dim = self.net_params['state_dim']

        if len(hidden_dim) > 0:
            self.decoder_fc_in = nn.Linear(input_dim, hidden_dim[0])
            self.decoder_fc_hid = []
            for ii in range(1, len(hidden_dim)):
                self.decoder_fc_hid.append(nn.Linear(hidden_dim[ii - 1], hidden_dim[ii]))
            self.decoder_fc_out = nn.Linear(hidden_dim[-1], output_dim)
        else:
            self.decoder_fc_out = nn.Linear(input_dim, output_dim)

    def decode_forward_(self, z):
        if len(self.net_params['decoder_hidden_dim']) > 0:
            z = F.relu(self.decoder_fc_in(z))
            for layer in self.decoder_fc_hid:
                z = F.relu(layer(z))
        z = self.decoder_fc_out(z)

        return z

    def decode(self, z):
        self.eval()
        z = torch.from_numpy(z).float()
        x = self.decode_forward_(z)
        return x.detach().numpy()
'''