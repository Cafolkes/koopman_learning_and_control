import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KoopmanNetAut(nn.Module):
    def __init__(self, net_params):
        super(KoopmanNetAut, self).__init__()
        self.net_params = net_params

        self.encoder = None
        self.decoder = None
        self.construct_net()

        n = self.net_params['state_dim']
        n_encode = self.net_params['encoder_output_dim']
        self.C = torch.cat((torch.eye(n), torch.zeros((n, n_encode))), 1)

        # TODO: Implement bias (see controlled version...)
        # TODO: Handle bias terms if one should be add ed to observables
        # TODO: Support adding state to observables
        # TODO: Support overriding kinematics

    def construct_net(self):
        self.construct_encoder_()

        n_tot = self.net_params['state_dim'] + self.net_params['encoder_output_dim']
        self.koopman_fc_drift = nn.Linear(n_tot, n_tot, bias=False)

    def forward(self, data):
        # data = [x, x_prime]
        # output = [x_pred, x_prime_pred, lin_error]
        n = self.net_params['state_dim']
        n_multistep = self.net_params['n_multistep']
        x_vec = data[:, :n*n_multistep]
        x_prime_vec = data[:, n*n_multistep:]

        # Define autoencoder networks:
        x = x_vec[:, :n]
        z = torch.cat((x, self.encode_forward_(x)), 1)

        # Define linearity networks:
        n_tot = n + self.net_params['encoder_output_dim']
        z_prime = torch.cat([torch.cat((x_prime_vec[:, n*ii:n*(ii+1)], self.encode_forward_(x_prime_vec[:, n*ii:n*(ii+1)])), 1) for ii in range(n_multistep)], 1)
        z_prime_pred = torch.cat([torch.matmul(z, torch.transpose(torch.pow(self.koopman_fc_drift.weight + torch.eye(n_tot), ii+1), 0, 1)) for ii in range(n_multistep)], 1)

        # Define prediction network:
        x_prime_pred = torch.cat([torch.matmul(z_prime_pred[:,n_tot*ii:n_tot*(ii+1)], torch.transpose(self.C, 0, 1)) for ii in range(n_multistep)], 1)

        outputs = torch.cat((x_prime_pred, z_prime_pred, z_prime), 1)

        return outputs

    def loss(self, outputs, labels):
        # output = [x_pred, x_prime_pred, lin_error]
        # labels = [x, x_prime], penalize when lin_error is not zero

        n = self.net_params['state_dim']
        n_tot = n + self.net_params['encoder_output_dim']
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
        self.eval()
        x_t = torch.from_numpy(x).float()
        z = np.concatenate((x, self.encode_forward_(x_t).detach().numpy()), axis=1)
        return z

    def send_to(self, device):
        hidden_dim = self.net_params['encoder_hidden_dim']

        if len(hidden_dim) > 0:
            self.encoder_fc_in.to(device)
            for ii in range(len(hidden_dim)-1):
                self.encoder_fc_hid[ii].to(device)
            self.encoder_fc_out.to(device)
        else:
            self.encoder_fc_out.to(device)

        self.koopman_fc_drift.to(device)
        self.C = self.C.to(device)


    def process(self, data, t, downsample_rate=1):
        n = self.net_params['state_dim']
        n_traj = data.shape[0]

        traj_length = data.shape[1]
        n_multistep = self.net_params['n_multistep']
        x = np.zeros((n_traj, traj_length-n_multistep, n*n_multistep))
        x_prime = np.zeros((n_traj, traj_length - n_multistep, n * n_multistep))
        for ii in range(n_multistep):
            x[:, :, n*ii:n*(ii+1)] = data[:, ii:-(n_multistep-ii), :]
            if ii + 1 < n_multistep:
                x_prime[:, :, n*ii:n*(ii+1)] = data[:, ii+1:-(n_multistep - ii - 1), :]
            else:
                x_prime[:, :, n*ii:n*(ii+1)] = data[:, ii+1:, :]

        order = 'F'
        n_data_pts = n_traj * (t[0,:].shape[0] - n_multistep)
        x_flat = x.T.reshape((n*n_multistep, n_data_pts), order=order)
        x_prime_flat = x_prime.T.reshape((n*n_multistep, n_data_pts), order=order)

        #if self.standardizer is None:
        x_flat, x_prime_flat = x_flat.T, x_prime_flat.T
        #else:
            #pass
            #self.standardizer.fit(x_flat.T)
            #x_flat, x_prime_flat = self.standardizer.transform(x_flat.T), x_prime_flat.T

        X = np.concatenate((x_flat, x_prime_flat), axis=1)

        return X[::downsample_rate,:], x_prime_flat[::downsample_rate,:]

    def construct_dyn_mat_discrete(self):
        n = self.net_params['state_dim']
        encoder_output_dim = self.net_params['encoder_output_dim']
        first_obs_const = self.net_params['first_obs_const']
        n_tot = n + encoder_output_dim + int(first_obs_const)
        n_kinematics = int(first_obs_const) + int(n/2)
        dt = self.net_params['dt']

        self.A = self.koopman_fc_drift.weight.data.numpy()

        if self.net_params['override_kinematics']:
            self.A = np.concatenate((np.zeros((n_kinematics, n_tot)), self.A), axis=0)
            self.A += np.eye(self.A.shape[0])
            self.A[int(first_obs_const):n_kinematics, n_kinematics:n_kinematics+int(n/2)] += np.eye(int(n/2))*dt
        else:
            self.A += np.eye(self.A.shape[0])

    def construct_dyn_mat_continuous(self):
        pass
