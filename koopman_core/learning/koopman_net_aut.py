import torch
import torch.nn as nn
from koopman_core.learning import KoopmanNet
import numpy as np

class KoopmanNetAut(KoopmanNet):
    def __init__(self, net_params, standardizer_x=None):
        super(KoopmanNetAut, self).__init__(net_params, standardizer_x=standardizer_x)

    def construct_net(self):
        n = self.net_params['state_dim']
        encoder_output_dim = self.net_params['encoder_output_dim']
        first_obs_const = int(self.net_params['first_obs_const'])
        n_tot = n + encoder_output_dim + first_obs_const

        self.C = torch.cat((torch.zeros((n, first_obs_const)), torch.eye(n), torch.zeros((n, encoder_output_dim))), 1)
        self.construct_encoder_()
        if self.net_params['override_kinematics']:
            self.koopman_fc_drift = nn.Linear(n_tot, n_tot-(first_obs_const + int(n/2)), bias=False)
        else:
            self.koopman_fc_drift = nn.Linear(n_tot, n_tot-first_obs_const, bias=False)

        self.opt_parameters_dyn_mats.append(self.koopman_fc_drift.weight)
        self.parameters_to_prune = [(self.koopman_fc_drift, "weight")]

    def forward(self, data):
        # data = [x, x_prime]
        # output = [x_pred, x_prime_pred, lin_error]
        n = self.net_params['state_dim']
        first_obs_const = int(self.net_params['first_obs_const'])

        x_vec = data[:, :n]
        x_prime_vec = data[:, n:]

        # Define autoencoder networks:
        x = x_vec[:, :n]
        z = torch.cat((torch.ones((x.shape[0], first_obs_const), device=self.device), x, self.encode_forward_(x)), 1)
        z_prime_diff = self.encode_forward_(x_prime_vec) - z[:, first_obs_const+n:]  # TODO: Assumes z = [x phi]^T, generalize?

        # Define linearity networks:
        drift_matrix = self.construct_drift_matrix_()
        z_prime_diff_pred = torch.matmul(z, torch.transpose(drift_matrix, 0, 1))

        # Define prediction network:
        x_prime_diff_pred = torch.matmul(z_prime_diff_pred, torch.transpose(self.C, 0, 1))

        z_prime_diff_pred = z_prime_diff_pred[:, first_obs_const+n:]  # TODO: Assumes z = [x phi]^T, generalize?

        return torch.cat((x_prime_diff_pred, z_prime_diff_pred, z_prime_diff), 1)

    def construct_drift_matrix_(self):
        n = self.net_params['state_dim']
        override_kinematics = self.net_params['override_kinematics']
        first_obs_const = int(self.net_params['first_obs_const'])
        n_tot = n + self.net_params['encoder_output_dim'] + first_obs_const
        dt = self.net_params['dt']

        if override_kinematics:
            const_obs_dyn = torch.zeros((first_obs_const, n_tot), device=self.koopman_fc_drift.weight.device)
            kinematics_dyn = torch.zeros((int(n/2), n_tot), device=self.koopman_fc_drift.weight.device)
            kinematics_dyn[:, first_obs_const+int(n/2):first_obs_const+n] = torch.eye(int(n/2))*dt
            drift_matrix = torch.cat((const_obs_dyn,
                                      kinematics_dyn,
                                      self.koopman_fc_drift.weight), 0)

        else:
            const_obs_dyn = torch.zeros((first_obs_const, n_tot), device=self.koopman_fc_drift.weight.device)
            drift_matrix = torch.cat((const_obs_dyn, self.koopman_fc_drift.weight), 0)

        return drift_matrix

    def send_to(self, device):
        hidden_depth = self.net_params['encoder_hidden_depth']

        if hidden_depth > 0:
            self.encoder_fc_in.to(device)
            for ii in range(hidden_depth-1):
                self.encoder_fc_hid[ii].to(device)
            self.encoder_fc_out.to(device)
        else:
            self.encoder_fc_out.to(device)

        self.koopman_fc_drift.to(device)
        self.C = self.C.to(device)

    def process(self, data_x, t, data_u=None, downsample_rate=1):
        n = self.net_params['state_dim']
        n_traj = data_x.shape[0]

        data_scaled = self.preprocess_data(data_x, self.standardizer_x)
        x = data_scaled[:, :-1, :]
        x_prime = data_scaled[:, 1:, :]

        order = 'F'
        n_data_pts = n_traj * (t[0,:].shape[0] - 1)
        x_flat = x.T.reshape((n, n_data_pts), order=order)
        x_prime_flat = x_prime.T.reshape((n, n_data_pts), order=order)

        X = np.concatenate((x_flat.T, x_prime_flat.T), axis=1)
        y = x_prime_flat.T - x_flat.T

        return X[::downsample_rate,:], y[::downsample_rate,:]

    def construct_dyn_mat(self):
        n = self.net_params['state_dim']
        first_obs_const = int(self.net_params['first_obs_const'])
        override_kinematics = self.net_params['override_kinematics']
        n_tot = n + self.net_params['encoder_output_dim'] + first_obs_const
        dt = self.net_params['dt']

        self.A = self.construct_drift_matrix_().data.numpy()
        if override_kinematics:
            self.A[first_obs_const+int(n/2):, :] *= dt
            if self.standardizer_x is not None:
                x_dot_scaling = np.divide(self.standardizer_x.scale_[int(n/2):], self.standardizer_x.scale_[:int(n/2)]).reshape(-1,1)
                self.A[first_obs_const: first_obs_const+int(n/2), :] = \
                    np.multiply(self.A[first_obs_const: first_obs_const+int(n/2), :], x_dot_scaling)
        else:
            self.A[first_obs_const:, :] *= dt

        self.A += np.eye(n_tot)

    def get_l1_norm_(self):
        return torch.norm(self.koopman_fc_drift.weight.view(-1), p=1)
