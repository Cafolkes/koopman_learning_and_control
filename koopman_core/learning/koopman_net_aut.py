import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KoopmanNetAut(nn.Module):
    def __init__(self, net_params, standardizer=None):
        super(KoopmanNetAut, self).__init__()
        self.net_params = net_params
        self.standardizer = standardizer

        self.encoder = None
        self.decoder = None
        self.optimization_parameters = []

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'

    # TODO: Master TODO:
        # TODO: Implement multistep prediction
        # TODO: Clean up code
        # TODO: Verify tuning script and run parameter tuning

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
            self.koopman_fc_drift = nn.Linear(n_tot, n_tot, bias=False)

        self.optimization_parameters.append(self.koopman_fc_drift.weight)

    def forward(self, data):
        # data = [x, x_prime]
        # output = [x_pred, x_prime_pred, lin_error]
        n = self.net_params['state_dim']
        n_multistep = self.net_params['n_multistep']
        first_obs_const = int(self.net_params['first_obs_const'])
        n_tot = n + self.net_params['encoder_output_dim'] + first_obs_const
        override_kinematics = self.net_params['override_kinematics']

        x_vec = data[:, :n*n_multistep]
        x_prime_vec = data[:, n*n_multistep:]

        # Define autoencoder networks:
        x = x_vec[:, :n]
        z = torch.cat((torch.ones((x.shape[0], first_obs_const), device=torch.device(self.device)), x, self.encode_forward_(x)), 1)
        #z_prime_diff = torch.cat([torch.cat(
        #    (torch.ones((x_prime_vec.shape[0], first_obs_const), device=torch.device(self.device)),
        #     x_prime_vec[:, n*ii:n*(ii+1)],
        #     self.encode_forward_(x_prime_vec[:, n*ii:n*(ii+1)])), 1) - z for ii in range(n_multistep)], 1)  # TODO: Verify that - z correct

        z_prime_diff = torch.cat([self.encode_forward_(x_prime_vec[:, n*ii:n*(ii+1)]) - z[:, first_obs_const + n:] for ii in range(n_multistep)], 1)  # TODO: Verify that - z correct
        # TODO: Assumes z = [x phi]^T, generalize?, implement for multistep

        # Define linearity networks:
        drift_matrix = self.construct_drift_matrix_()
        # TODO: Implement multistep pred with difference matrix
        id_matrix = torch.eye(n_tot, device=torch.device(self.device))
        z_prime_diff_pred = torch.cat([torch.matmul(z, torch.transpose(torch.pow(drift_matrix + id_matrix, ii + 1) - id_matrix, 0, 1)) for ii in range(n_multistep)], 1)

        # Define prediction network:
        x_prime_diff_pred = torch.cat([torch.matmul(z_prime_diff_pred[:,n_tot*ii:n_tot*(ii+1)], torch.transpose(self.C, 0, 1)) for ii in range(n_multistep)], 1)

        z_prime_diff_pred = torch.cat([z_prime_diff_pred[:, ii*n_tot + first_obs_const+n:(ii+1)*n_tot] for ii in range(n_multistep)], 1)  # TODO: Assumes z = [x phi]^T, generalize?

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
                                      self.koopman_fc_drift.weight), 0) #+ torch.eye(n_tot, device=self.koopman_fc_drift.weight.device)

        else:
            drift_matrix = self.koopman_fc_drift.weight #+ torch.eye(n_tot, device=self.koopman_fc_drift.weight.device)

        return drift_matrix

    def loss(self, outputs, labels, validation=False):
        # output = [x_pred, x_prime_pred, lin_error]
        # labels = [x, x_prime], penalize when lin_error is not zero
        n = self.net_params['state_dim']
        n_z = self.net_params['encoder_output_dim']
        n_multistep = self.net_params['n_multistep']
        dt = self.net_params['dt']
        override_kinematics = self.net_params['override_kinematics']

        if override_kinematics:
            #x_prime_diff_pred = outputs[:, int(n/2):n*n_multistep]
            x_prime_diff_pred = torch.cat([outputs[:, ii*n + int(n/2):(ii+1)*n] for ii in range(n_multistep)], 1)
            #x_prime_diff = labels[:,int(n/2):]
            x_prime_diff = torch.cat([labels[:, ii*n + int(n/2):(ii + 1)*n] for ii in range(n_multistep)], 1)
        else:
            x_prime_diff_pred = outputs[:, :n*n_multistep]
            x_prime_diff = labels

        z_prime_diff_pred = outputs[:, n*n_multistep:n*n_multistep + n_z*n_multistep]
        z_prime_diff = outputs[:, n*n_multistep + n_z*n_multistep:]

        alpha = self.net_params['lin_loss_penalty']
        criterion = nn.MSELoss()

        pred_loss = criterion(x_prime_diff_pred, x_prime_diff/dt)/n_multistep
        lin_loss = criterion(z_prime_diff_pred, z_prime_diff/dt)/n_multistep

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

        #self.encoder_fc_out_norm = nn.BatchNorm1d(output_dim)
        #self.optimization_parameters.append(self.encoder_fc_out_norm.weight)

    def encode_forward_(self, x):
        if self.net_params['encoder_hidden_depth'] > 0:
            x = F.relu(self.encoder_fc_in(x))
            for layer in self.encoder_fc_hid:
                x = F.relu(layer(x))
        x = self.encoder_fc_out(x)
        #x = self.encoder_fc_out_norm(x)  

        return x

    def encode(self, x):
        first_obs_const = int(self.net_params['first_obs_const'])
        self.eval()
        x_t = torch.from_numpy(x).float()
        z = np.concatenate((np.ones((x.shape[0], first_obs_const)), x, self.encode_forward_(x_t).detach().numpy()), axis=1)
        return z

    def send_to(self, device):
        hidden_depth = self.net_params['encoder_hidden_depth']

        if hidden_depth > 0:
            self.encoder_fc_in.to(device)
            for ii in range(hidden_depth-1):
                self.encoder_fc_hid[ii].to(device)
            self.encoder_fc_out.to(device)
        else:
            self.encoder_fc_out.to(device)
        #self.encoder_fc_out_norm.to(device)

        self.koopman_fc_drift.to(device)
        self.C = self.C.to(device)

    def process_single_step(self, data, t, downsample_rate=1, train_data=False):
        n = self.net_params['state_dim']
        n_traj = data.shape[0]
        traj_length = data.shape[1]
        x = np.zeros((n_traj, traj_length - 1, n))
        x_prime = np.zeros((n_traj, traj_length - 1, n))

        data_scaled = self.preprocess_data(data, train_data)
        x = data_scaled[:, :-1, :]
        x_prime = data_scaled[:, 1:, :]

        order = 'F'
        n_data_pts = n_traj * (t[0,:].shape[0] - 1)
        x_flat = x.T.reshape((n, n_data_pts), order=order)
        x_prime_flat = x_prime.T.reshape((n, n_data_pts), order=order)

        X = np.concatenate((x_flat.T, x_prime_flat.T), axis=1)
        y = x_prime_flat[::downsample_rate,:].T - x_flat[::downsample_rate,:].T

        return X[::downsample_rate,:], y[::downsample_rate,:]


    def process(self, data, t, downsample_rate=1, train_data=False):
        n = self.net_params['state_dim']
        n_traj = data.shape[0]
        traj_length = data.shape[1]
        n_multistep = self.net_params['n_multistep']
        x = np.zeros((n_traj, traj_length-n_multistep, n*n_multistep))
        x_prime = np.zeros((n_traj, traj_length - n_multistep, n * n_multistep))

        data_scaled = self.preprocess_data(data, train_data)
        for ii in range(n_multistep):
            x[:, :, n*ii:n*(ii+1)] = data_scaled[:, ii:-(n_multistep-ii), :]
            if ii + 1 < n_multistep:
                x_prime[:, :, n*ii:n*(ii+1)] = data_scaled[:, ii+1:-(n_multistep - ii - 1), :]
            else:
                x_prime[:, :, n*ii:n*(ii+1)] = data_scaled[:, ii+1:, :]

        order = 'F'
        n_data_pts = n_traj * (t[0,:].shape[0] - n_multistep)
        x_flat = x.T.reshape((n*n_multistep, n_data_pts), order=order)
        x_prime_flat = x_prime.T.reshape((n*n_multistep, n_data_pts), order=order)

        X = np.concatenate((x_flat.T, x_prime_flat.T), axis=1)
        y = x_prime_flat[::downsample_rate,:].T - x_flat[::downsample_rate,:].T

        return X[::downsample_rate,:], y[::downsample_rate,:]

    def preprocess_data(self, data, train_data):
        n = self.net_params['state_dim']
        n_traj = data.shape[0]
        traj_length = data.shape[1]
        data_flat = data.T.reshape((n, n_traj*traj_length), order='F')

        if train_data and self.standardizer is not None:
            self.standardizer.fit(data_flat.T)

        if self.standardizer is None:
            data_scaled = data
        else:
            data_scaled = np.array([self.standardizer.transform(d) for d in data])

        return data_scaled

    def construct_dyn_mat(self):
        n = self.net_params['state_dim']
        first_obs_const = int(self.net_params['first_obs_const'])
        override_kinematics = self.net_params['override_kinematics']
        n_tot = n + self.net_params['encoder_output_dim'] + first_obs_const
        dt = self.net_params['dt']

        self.A = self.construct_drift_matrix_().data.numpy()
        if override_kinematics:
            self.A[first_obs_const+int(n/2):,:] *= dt
        else:
            self.A[first_obs_const:, :] *= dt
        self.A += np.eye(n_tot)
