import torch
import torch.nn as nn
from . import KoopmanNet
import numpy as np

class KoopmanNetCtrl(KoopmanNet):
    def __init__(self, net_params, standardizer_x=None, standardizer_u=None):
        super(KoopmanNetCtrl, self).__init__(net_params, standardizer_x=standardizer_x, standardizer_u=standardizer_u)

    def construct_net(self):
        n = self.net_params['state_dim']
        m = self.net_params['ctrl_dim']
        encoder_output_dim = self.net_params['encoder_output_dim']
        first_obs_const = int(self.net_params['first_obs_const'])
        n_fixed_states = self.net_params['n_fixed_states']
        override_C = self.net_params['override_C']
        override_kinematics = self.net_params['override_kinematics']

        if override_C:
            self.n_tot = int(first_obs_const) + n_fixed_states + encoder_output_dim
        else:
            self.n_tot = int(first_obs_const) + encoder_output_dim
            assert override_kinematics is False, \
                'Not overriding C while overriding kinematics not supported'

        #self.C = torch.cat((torch.zeros((n_fixed_states, first_obs_const)), torch.eye(n_fixed_states), torch.zeros((n_fixed_states, encoder_output_dim))), 1)
        self.construct_encoder_()
        if self.net_params['override_kinematics']:
            self.koopman_fc_drift = nn.Linear(self.n_tot, self.n_tot-(first_obs_const + int(n/2)), bias=False)
            self.koopman_fc_act = nn.Linear(m * self.n_tot, self.n_tot-(first_obs_const + int(n/2)), bias=False)
        else:
            self.koopman_fc_drift = nn.Linear(self.n_tot, self.n_tot-first_obs_const, bias=False)
            self.koopman_fc_act = nn.Linear(m*self.n_tot, self.n_tot-first_obs_const, bias=False)

        if self.net_params['override_C']:
            #self.C = torch.cat((torch.zeros((n, first_obs_const)), torch.eye(n), torch.zeros((n, encoder_output_dim))), 1)
            self.C = torch.cat((torch.zeros((n_fixed_states, first_obs_const)), torch.eye(n_fixed_states),
                                torch.zeros((n_fixed_states, encoder_output_dim))), 1)
        else:
            self.projection_fc = nn.Linear(self.n_tot, n, bias=False)

        self.parameters_to_prune = [(self.koopman_fc_drift, "weight"), (self.koopman_fc_act, "weight")]

    def forward(self, data):
        # data = [x, u, x_prime]
        # output = [x_prime_pred, z_prime_pred, z_prime]
        n = self.net_params['state_dim']
        m = self.net_params['ctrl_dim']
        first_obs_const = int(self.net_params['first_obs_const'])
        override_C = self.net_params['override_C']
        n_fixed_states = self.net_params['n_fixed_states']

        x = data[:, :n]
        u = data[:, n:n+m]
        x_prime = data[:, n+m:]

        # Define linearity networks:
        if override_C:
            z = torch.cat((torch.ones((x.shape[0], first_obs_const), device=self.device),
                           x[:, :n_fixed_states],
                           self.encode_forward_(x)), 1)
            z_prime_diff = self.encode_forward_(x_prime) - z[:, first_obs_const+n_fixed_states:]
        else:
            z = torch.cat((torch.ones((x.shape[0], first_obs_const), device=self.device), self.encode_forward_(x)), 1)
            z_prime_diff = self.encode_forward_(x_prime) - z[:, first_obs_const:]

        z_u = torch.cat(
            [torch.transpose(torch.mul(torch.transpose(z, 0, 1), u_i), 0, 1) for u_i in torch.transpose(u, 0, 1)], 1)

        drift_matrix, act_matrix = self.construct_drift_act_matrix_()
        z_prime_diff_pred = torch.matmul(z, torch.transpose(drift_matrix, 0, 1)) \
                            + torch.matmul(z_u, torch.transpose(act_matrix, 0, 1))

        # Define prediction network:
        if override_C:
            # x_prime_diff_pred = torch.matmul(z_prime_diff_pred, torch.transpose(self.C, 0, 1))
            #x_prime_pred = torch.matmul(z + z_prime_diff_pred * self.loss_scaler, torch.transpose(self.C, 0, 1))
            x_proj = torch.matmul(z, torch.transpose(self.C, 0, 1))
            x_prime_diff_pred = torch.matmul(z_prime_diff_pred, torch.transpose(self.C, 0, 1))
            z_prime_diff_pred = z_prime_diff_pred[:, first_obs_const + n_fixed_states:]
        else:
            # x_prime_pred = self.projection_fc(z + z_prime_diff_pred*dt)
            x_proj = self.projection_fc(z)
            #x_prime_pred = self.projection_fc(z + z_prime_diff_pred * self.loss_scaler)
            x_prime_diff_pred = self.projection_fc(z_prime_diff_pred)
            z_prime_diff_pred = z_prime_diff_pred[:, first_obs_const:]

        return torch.cat((x_proj, x_prime_diff_pred, z_prime_diff_pred, z_prime_diff), 1)

    def construct_drift_act_matrix_(self):
        n = self.net_params['state_dim']
        m = self.net_params['ctrl_dim']
        override_kinematics = self.net_params['override_kinematics']
        first_obs_const = int(self.net_params['first_obs_const'])
        dt = self.net_params['dt']

        if override_kinematics:
            const_obs_dyn_drift = torch.zeros((first_obs_const, self.n_tot), device=self.koopman_fc_drift.weight.device)
            kinematics_dyn_drift = torch.zeros((int(n/2), self.n_tot), device=self.koopman_fc_drift.weight.device)
            kinematics_dyn_drift[:, first_obs_const+int(n/2):first_obs_const+n] = dt*torch.eye(int(n/2), device=self.koopman_fc_drift.weight.device)
            drift_matrix = torch.cat((const_obs_dyn_drift,
                                      kinematics_dyn_drift,
                                      self.koopman_fc_drift.weight), 0)

            const_obs_dyn_act = torch.zeros((first_obs_const, m * self.n_tot), device=self.koopman_fc_drift.weight.device)
            kinematics_dyn_act = torch.zeros((int(n / 2), m * self.n_tot), device=self.koopman_fc_drift.weight.device)
            act_matrix = torch.cat((const_obs_dyn_act, kinematics_dyn_act, self.koopman_fc_act.weight), 0)
        else:
            const_obs_dyn_drift = torch.zeros((first_obs_const, self.n_tot), device=self.koopman_fc_drift.weight.device)
            drift_matrix = torch.cat((const_obs_dyn_drift, self.koopman_fc_drift.weight), 0)

            const_obs_dyn_act = torch.zeros((first_obs_const, m*self.n_tot), device=self.koopman_fc_drift.weight.device)
            act_matrix = torch.cat((const_obs_dyn_act, self.koopman_fc_act.weight), 0)

        return drift_matrix, act_matrix

    def send_to(self, device):
        hidden_dim = self.net_params['encoder_hidden_depth']
        override_C = self.net_params['override_C']

        if hidden_dim > 0:
            self.encoder_fc_in.to(device)
            for ii in range(hidden_dim-1):
                self.encoder_fc_hid[ii].to(device)
            self.encoder_fc_out.to(device)
        else:
            self.encoder_fc_out.to(device)

        self.koopman_fc_drift.to(device)
        self.koopman_fc_act.to(device)
        if override_C:
            self.C = self.C.to(device)
        else:
            self.projection_fc.to(device)

        try:
            self.loss_scaler_x = self.loss_scaler_x.to(device)
        except:
            pass

    def process(self, data_x, t, data_u=None, downsample_rate=1, train_mode=True):
        n = self.net_params['state_dim']
        m = self.net_params['ctrl_dim']
        n_fixed_states = self.net_params['n_fixed_states']
        n_traj = data_x.shape[0]

        data_scaled_x = self.preprocess_data(data_x, self.standardizer_x)
        data_scaled_u = self.preprocess_data(data_u, self.standardizer_u)
        x = data_scaled_x[:, :-1, :]
        u = data_scaled_u
        x_prime = data_scaled_x[:,1:,:]

        order = 'F'
        n_data_pts = n_traj * (x.shape[1])
        x_flat = x.T.reshape((n, n_data_pts), order=order)
        u_flat = u.T.reshape((m, n_data_pts), order=order)
        x_prime_flat = x_prime.T.reshape((n, n_data_pts), order=order)

        X = np.concatenate((x_flat.T, u_flat.T, x_prime_flat.T), axis=1)
        #y = x_prime_flat.T
        y = np.concatenate((x_flat.T, x_prime_flat.T - x_flat.T), axis=1)

        if train_mode:
            self.loss_scaler_x = torch.Tensor(np.std(x_prime_flat[:n_fixed_states, :].T - x_flat[:n_fixed_states, :].T, axis=0))
            #self.loss_scaler_x = torch.ones_like(self.loss_scaler_x)  # TODO: Remove
            #self.loss_scaler_z = torch.Tensor(np.std(x_prime_flat.T - x_flat.T, axis=0), device=self.device)
            self.loss_scaler_z = np.std(x_prime_flat.T - x_flat.T)
            #self.loss_scaler_z = 1 # TODO: Remove

        return X[::downsample_rate,:], y[::downsample_rate,:]

    def construct_dyn_mat(self):
        n = self.net_params['state_dim']
        m = self.net_params['ctrl_dim']
        encoder_output_dim = self.net_params['encoder_output_dim']
        first_obs_const = int(self.net_params['first_obs_const'])
        override_kinematics = self.net_params['override_kinematics']
        override_C = self.net_params['override_C']

        if override_C:
            loss_scaler = np.concatenate((np.ones(first_obs_const), self.loss_scaler_x.numpy(), self.loss_scaler_z*np.ones(encoder_output_dim)))
        else:
            loss_scaler = np.concatenate((np.ones(first_obs_const), self.loss_scaler_z * np.ones(encoder_output_dim)))

        drift_matrix, act_matrix = self.construct_drift_act_matrix_()

        self.A = drift_matrix.data.numpy()
        if override_kinematics:
            #self.A[first_obs_const+int(n/2):,:] *= self.loss_scaler
            self.A[first_obs_const + int(n / 2):, :] = np.multiply(self.A[first_obs_const + int(n / 2):, :],
                                                                   loss_scaler[first_obs_const + int(n / 2):].reshape(-1, 1))
            if self.standardizer_x is not None:
                x_dot_scaling = np.divide(self.standardizer_x.scale_[int(n/2):], self.standardizer_x.scale_[:int(n/2)]).reshape(-1,1)
                self.A[first_obs_const: first_obs_const+int(n/2), :] = \
                    np.multiply(self.A[first_obs_const: first_obs_const+int(n/2), :], x_dot_scaling)
        else:
            #self.A[first_obs_const:, :] *= self.loss_scaler
            self.A[first_obs_const:, :] = np.multiply(self.A[first_obs_const:, :],
                                                      loss_scaler[first_obs_const:].reshape(-1, 1))
        self.A += np.eye(self.n_tot)

        B_vec = act_matrix.data.numpy()
        self.B = [np.multiply(B_vec[:, self.n_tot * ii:self.n_tot * (ii + 1)], loss_scaler.reshape(-1,1)) for ii in range(m)]

        if not override_C:
            self.C = self.projection_fc.weight.detach().numpy()

    def get_l1_norm_(self):
        return torch.norm(self.koopman_fc_drift.weight.view(-1), p=1) + torch.norm(self.koopman_fc_act.weight.view(-1), p=1)


