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

        n = self.net_params['state_dim']
        m = self.net_params['ctrl_dim']
        encoder_output_dim = self.net_params['encoder_output_dim']
        first_obs_const = self.net_params['first_obs_const']
        n_tot = n + encoder_output_dim + int(first_obs_const)
        if self.net_params['override_kinematics']:
            self.koopman_fc_drift = nn.Linear(n_tot, n_tot-(int(first_obs_const) + int(n/2)), bias=False)
            self.koopman_fc_act = nn.Linear(m * n_tot, n_tot-(int(first_obs_const) + int(n/2)), bias=False)
        else:
            self.koopman_fc_drift = nn.Linear(n_tot, n_tot, bias=False)
            self.koopman_fc_act = nn.Linear(m*n_tot, n_tot, bias=False)

    def forward(self, data):
        # data = [x, u, x_prime]
        # output = [x_prime_pred, z_prime_pred, z_prime]
        n = self.net_params['state_dim']
        m = self.net_params['ctrl_dim']
        n_multistep = self.net_params['n_multistep']
        first_obs_const = self.net_params['first_obs_const']
        dt = self.net_params['dt']

        x_vec = data[:, :n*n_multistep]
        u_vec = data[:, n*n_multistep:n*n_multistep+m*n_multistep]
        x_prime_vec = data[:, n*n_multistep+m*n_multistep:]

        # Define linearity networks:
        x = x_vec[:, :n]
        x_prime = x_prime_vec[:, :n]
        n_tot = int(self.net_params['first_obs_const']) + n + self.net_params['encoder_output_dim']
        if first_obs_const:
            z = torch.cat((torch.ones((x.shape[0], 1)), x, self.encode_forward_(x)), 1)
            #z_prime = torch.cat((torch.ones((x_prime.shape[0], 1)), x_prime, self.encode_forward_(x_prime)), 1)
            z_prime = torch.cat([torch.cat(
                (torch.ones((x_prime_vec.shape[0],1)),
                 x_prime_vec[:, n*ii:n * (ii+1)], self.encode_forward_(x_prime_vec[:, n*ii:n * (ii+1)])), 1) for
                ii in range(n_multistep)], 1)
        else:
            z = torch.cat((x, self.encode_forward_(x)), 1)
            #z_prime = torch.cat((x_prime, self.encode_forward_(x_prime)), 1)
            z_prime = torch.cat([torch.cat(
                (x_prime_vec[:, n*ii:n*(ii+1)], self.encode_forward_(x_prime_vec[:, n*ii:n*(ii+1)])), 1) for
                                 ii in range(n_multistep)], 1)
        #z_prime_pred = torch.cat([torch.matmul(z, torch.transpose(torch.pow(self.koopman_fc_drift.weight + torch.eye(n_tot), ii+1), 0, 1)) for ii in range(n_multistep)], 1)

        #z_u = torch.cat([torch.transpose(torch.mul(torch.transpose(z,0,1), u), 0,1) for u in torch.transpose(u_vec,0,1)], 1)

        #z_pred = torch.zeros((z.shape[0], n_tot*(n_multistep+1)))
        z_pred = z
        z_prime_pred = torch.empty((z.shape[0],0))
        #z_u_pred = torch.zeros((z_prime_pred.shape[0], m*n_tot*n_multistep))
        z_u_pred = torch.cat(
            [torch.transpose(torch.mul(torch.transpose(z, 0, 1), u), 0, 1) for u in torch.transpose(u_vec, 0, 1)], 1)
        if self.net_params['override_kinematics']:
            for kk in range(n_multistep):
                z_prime_pred_tmp = z_pred[:, n_tot * kk + int(first_obs_const) + int(n / 2):n_tot * (kk + 1)] \
                                   + self.koopman_fc_drift(z_pred[:, n_tot * kk:n_tot * (kk + 1)]) \
                                   + self.koopman_fc_act(z_u_pred[:, m * n_tot * kk:m * n_tot * (kk + 1)])

                z_prime_pred = torch.cat((z_prime_pred, torch.cat((torch.ones((x_prime.shape[0], 1)),
                                                                   z_pred[:, n_tot*kk:n_tot*kk+int(n/2)] + z_pred[:, n_tot*kk+int(n/2):n_tot*kk+n]*dt,
                                                                   z_prime_pred_tmp), 1)), 1)

                z_u_pred = torch.cat((z_u_pred, torch.cat([torch.transpose(torch.mul(torch.transpose(
                    z_pred[:, n_tot * kk:n_tot * (kk + 1)], 0, 1), u), 0, 1) for u in
                    torch.transpose(u_vec[:, m * kk:m * (kk + 1)], 0, 1)], 1)), 1)

                if kk < n_multistep:
                    z_pred = torch.cat((z_pred, z_prime_pred[:, n_tot * kk:n_tot * (kk + 1)]), 1)
        else:
            for kk in range(n_multistep):
                z_prime_pred = torch.cat((z_prime_pred, z_pred[:, n_tot*kk:n_tot*(kk+1)] \
                                                         + self.koopman_fc_drift(z_pred[:, n_tot*kk:n_tot*(kk+1)]) \
                                                         + self.koopman_fc_act(z_u_pred[:, m*n_tot*kk:m*n_tot*(kk+1)])),1)

                z_u_pred = torch.cat((z_u_pred, torch.cat([torch.transpose(torch.mul(torch.transpose(
                    z_pred[:, n_tot * kk:n_tot * (kk + 1)], 0, 1), u), 0, 1) for u in
                    torch.transpose(u_vec[:, m * kk:m * (kk + 1)], 0, 1)], 1)), 1)

                if kk < n_multistep:
                    z_pred = torch.cat((z_pred, z_prime_pred[:, n_tot*kk:n_tot*(kk+1)]),1)

        # Define prediction network:
        x_prime_pred = torch.cat([torch.matmul(z_prime_pred[:,n_tot*ii:n_tot*(ii+1)], torch.transpose(self.C, 0, 1)) for ii in range(n_multistep)], 1)

        outputs = torch.cat((x_prime_pred, z_prime_pred, z_prime), 1)

        return outputs

    def loss(self, outputs, labels):
        # output = [x_prime_pred, z_prime_pred, z_prime]
        # labels = [x_prime]

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
            total_loss += l1_reg * (torch.norm(self.koopman_fc_drift.weight.flatten(), p=1)
                                    + torch.norm(self.koopman_fc_act.weight.flatten(), p=1))

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