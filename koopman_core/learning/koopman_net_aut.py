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

    def construct_net(self):
        self.construct_encoder_()
        #self.construct_decoder_()

        n_tot = self.net_params['state_dim'] + self.net_params['encoder_output_dim']
        self.koopman_fc = nn.Linear(n_tot, n_tot, bias=False)  #TODO: Evaluate how to handle bias

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
        z_prime = torch.tensor([torch.cat((x_prime_vec[:, n*ii:n*(ii+1)], self.encode_forward_(x_prime_vec[:, n*ii:n*(ii+1)])), 1) for ii in range(n_multistep)])
        #z_prime_pred = z + self.koopman_fc(z)
        z_prime_pred = torch.tensor([torch.matmul(z, torch.transpose(torch.pow(self.koopman_fc.weight + torch.eye(n_tot), ii+1), 0, 1)) for ii in range(n_multistep)])

        # Define prediction network:
        x_prime_pred = torch.tensor([torch.matmul(z,torch.transpose(self.C, 0, 1)) ])  # TODO: Continue

        outputs = torch.cat((x_prime_pred, z_prime_pred, z_prime), 1)

        return outputs

    def loss(self, outputs, labels):
        # output = [x_pred, x_prime_pred, lin_error]
        # labels = [x, x_prime], penalize when lin_error is not zero

        n = self.net_params['state_dim']
        n_tot = n + self.net_params['encoder_output_dim']
        x_prime_pred, x_prime = outputs[:, :n], labels
        z_prime_pred, z_prime = outputs[:, n:n+n_tot], outputs[:, n+n_tot:]

        alpha = self.net_params['lin_loss_penalty']
        criterion = nn.MSELoss()

        pred_loss = criterion(x_prime_pred, x_prime)
        lin_loss = criterion(z_prime_pred, z_prime)

        total_loss = pred_loss + alpha * lin_loss
        if 'l1_reg' in self.net_params and self.net_params['l1_reg'] > 0:  # TODO: Verify correct l1-regularization
            l1_reg = self.net_params['l1_reg']
            print('l1 shape', self.koopman_fc.weight.view(-1).shape)
            print('l1 shape', self.koopman_fc.weight.flatten().shape)
            total_loss += l1_reg * torch.norm(self.koopman_fc.weight.flatten(), p=1)
            #total_loss += l1_reg*torch.norm(self.koopman_fc.weight, p=1)

        #print(total_loss)
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