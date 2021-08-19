import torch
import torch.nn as nn
from . import KoopmanNet
import numpy as np

class KoopmanNetAut(KoopmanNet):
    def __init__(self, net_params, standardizer_x=None):
        super(KoopmanNetAut, self).__init__(net_params, standardizer_x=standardizer_x)

    def construct_net(self):
        n = self.net_params['state_dim']
        encoder_output_dim = self.net_params['encoder_output_dim']
        first_obs_const = int(self.net_params['first_obs_const'])

        self.construct_encoder_()
        if self.net_params['override_kinematics']:
            self.koopman_fc_drift = nn.Linear(self.n_tot, self.n_tot-(first_obs_const + int(n/2)), bias=False)
        else:
            self.koopman_fc_drift = nn.Linear(self.n_tot, self.n_tot-first_obs_const, bias=False)

        if self.net_params['override_C']:
            self.C = torch.cat((torch.zeros((n, first_obs_const)), torch.eye(n), torch.zeros((n, encoder_output_dim))), 1)
        else:
            self.projection_fc = nn.Linear(self.n_tot, n, bias=False)

        self.parameters_to_prune = [(self.koopman_fc_drift, "weight")]

    def forward(self, data):
        # data = [x, x_prime]
        # output = [x_pred, x_prime_pred, lin_error]
        n = self.net_params['state_dim']
        first_obs_const = int(self.net_params['first_obs_const'])
        encoder_output_dim = self.net_params['encoder_output_dim']
        override_C = self.net_params['override_C']

        x = data[:, :n]
        x_prime = data[:, n:]

        # Define autoencoder networks:
        if override_C:
            z = torch.cat((torch.ones((x.shape[0], first_obs_const), device=self.device), x, self.encode_forward_(x)), 1)
            z_prime_diff = self.encode_forward_(x_prime) - z[:, first_obs_const+n:]
        else:
            z = torch.cat((torch.ones((x.shape[0], first_obs_const), device=self.device), self.encode_forward_(x)), 1)
            z_prime_diff = self.encode_forward_(x_prime) - z[:, first_obs_const:]

        # Define linearity networks:
        drift_matrix = self.construct_drift_matrix_()
        z_prime_diff_pred = torch.matmul(z, torch.transpose(drift_matrix, 0, 1))

        # Define prediction network:
        if override_C:
            #x_prime_diff_pred = torch.matmul(z_prime_diff_pred, torch.transpose(self.C, 0, 1))
            # TODO: Debug and test
            scaler = torch.cat((torch.ones(first_obs_const), self.loss_scaler_x, self.loss_scaler_z*torch.ones(encoder_output_dim)))
            #x_prime_pred = torch.matmul(z + torch.multiply(z_prime_diff_pred, scaler), torch.transpose(self.C, 0, 1))
            x_proj = torch.matmul(z, torch.transpose(self.C, 0, 1))
            #x_prime_diff_pred = x_prime_pred - x
            x_prime_diff_pred = torch.matmul(z_prime_diff_pred, torch.transpose(self.C, 0, 1))
            z_prime_diff_pred = z_prime_diff_pred[:, first_obs_const + n:]
        else:
            scaler = torch.cat((torch.ones(first_obs_const), self.loss_scaler_z * torch.ones(encoder_output_dim)))
            #x_prime_pred = self.projection_fc(z + z_prime_diff_pred*dt)
            #x_prime_pred = self.projection_fc(z) + \
            #               torch.multiply(self.projection_fc(torch.multiply(z_prime_diff_pred, scaler)), self.loss_scaler_x) # TODO: Split up in 2 parts, adjust for scale_x in diff
            x_proj = self.projection_fc(z)
            x_prime_diff_pred = self.projection_fc(z_prime_diff_pred)
            z_prime_diff_pred = z_prime_diff_pred[:, first_obs_const:]

        #return torch.cat((x_prime_diff_pred, z_prime_diff_pred, z_prime_diff), 1)
        return torch.cat((x_proj, x_prime_diff_pred, z_prime_diff_pred, z_prime_diff), 1)

    def construct_drift_matrix_(self):
        n = self.net_params['state_dim']
        override_kinematics = self.net_params['override_kinematics']
        first_obs_const = int(self.net_params['first_obs_const'])
        dt = self.net_params['dt']
        override_C = self.net_params['override_C']
        if override_C:
            n_tot = n + self.net_params['encoder_output_dim'] + first_obs_const
        else:
            n_tot = self.net_params['encoder_output_dim'] + first_obs_const

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
        override_C = self.net_params['override_C']

        if hidden_depth > 0:
            self.encoder_fc_in.to(device)
            for ii in range(hidden_depth-1):
                self.encoder_fc_hid[ii].to(device)
            self.encoder_fc_out.to(device)
        else:
            self.encoder_fc_out.to(device)

        self.koopman_fc_drift.to(device)
        if override_C:
            self.C = self.C.to(device)
        else:
            self.projection_fc.to(device)

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
        #y = x_prime_flat.T - x_flat.T
        #y = x_prime_flat.T
        y = np.concatenate((x_flat.T, x_prime_flat.T - x_flat.T), axis=1)
        self.loss_scaler_x = torch.Tensor(np.std(x_prime_flat.T - x_flat.T, axis=0))
        self.loss_scaler_z = np.std(x_prime_flat.T - x_flat.T)

        return X[::downsample_rate,:], y[::downsample_rate,:]

    def construct_dyn_mat(self):
        n = self.net_params['state_dim']
        first_obs_const = int(self.net_params['first_obs_const'])
        override_kinematics = self.net_params['override_kinematics']
        encoder_output_dim = self.net_params['encoder_output_dim']
        override_C = self.net_params['override_C']

        if override_C:
            loss_scaler = np.concatenate((np.ones(first_obs_const), self.loss_scaler_x.numpy(), self.loss_scaler_z*np.ones(encoder_output_dim)))
        else:
            loss_scaler = np.concatenate((np.ones(first_obs_const), self.loss_scaler_z * np.ones(encoder_output_dim)))

        self.A = self.construct_drift_matrix_().data.numpy()
        if override_kinematics:
            #self.A[first_obs_const+int(n/2):, :] *= dt
            self.A[first_obs_const + int(n/2):, :] = np.multiply(self.A[first_obs_const + int(n/2):, :],
                                                                   loss_scaler[first_obs_const + int(n/2):].reshape(-1,1))
            if self.standardizer_x is not None:
                x_dot_scaling = np.divide(self.standardizer_x.scale_[int(n/2):], self.standardizer_x.scale_[:int(n/2)]).reshape(-1,1)
                self.A[first_obs_const: first_obs_const+int(n/2), :] = \
                    np.multiply(self.A[first_obs_const: first_obs_const+int(n/2), :], x_dot_scaling)
        else:
            #self.A[first_obs_const:, :] *= dt
            self.A[first_obs_const:, :] = np.multiply(self.A[first_obs_const:, :],
                                                                   loss_scaler[first_obs_const:].reshape(-1, 1))
            #self.A[first_obs_const:, :] *= self.loss_scaler_z
            #self.A = np.multiply(self.A, loss_scaler.reshape(-1, 1))

        self.A += np.eye(self.n_tot)

        if not override_C:
            self.C = self.projection_fc.weight.detach().numpy()

    def get_l1_norm_(self):
        return torch.norm(self.koopman_fc_drift.weight.view(-1), p=1)
