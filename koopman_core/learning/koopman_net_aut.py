import torch
import torch.nn as nn
import torch.nn.functional as F

class KoopmanNetAut(nn.Module):
    def __init__(self, net_params):
        super(KoopmanNetAut, self).__init__()
        self.net_params = net_params

        self.encoder = None
        self.decoder = None
        self.construct_net()

    def construct_net(self):
        self.construct_encoder_()
        self.construct_decoder_()

        self.koopman_fc = nn.Linear(self.net_params['encoder_output_dim'], self.net_params['encoder_output_dim'], bias=False)  #TODO: Evaluate how to handle bias

    def forward(self, data):
        # data = [x, x_prime]
        # output = [x_pred, x_prime_pred, lin_error]

        x = data[:, :self.net_params['state_dim']]
        x_prime = data[:, self.net_params['state_dim']:]

        # Define autoencoder networks:
        z = self.encode_forward_(x)
        x_pred = self.decode_forward_(z)

        # Define linearity networks:
        z_prime_pred = self.koopman_fc(z)
        z_prime = self.encode_forward_(x_prime)
        lin_error = z_prime - z_prime_pred

        # Define prediction network:
        x_prime_pred = self.decode_forward_(z_prime_pred)

        outputs = torch.cat((x_pred, x_prime_pred, lin_error), 1)

        return outputs

    def loss(self, outputs, labels):
        # output = [x_pred, x_prime_pred, lin_error]
        # labels = [x, x_prime], penalize when lin_error is not zero

        n = self.net_params['state_dim']
        x_pred, x = outputs[:,:n], labels[:, :n]
        x_prime_pred, x_prime = outputs[:, n:2*n], labels[:, n:]
        lin_error, zero_lin_error = outputs[:, 2*n:], torch.zeros_like(outputs[:, 2*n:])

        alpha = self.net_params['lin_loss_penalty']
        criterion = nn.MSELoss()

        autoencoder_loss = criterion(x_pred, x)
        pred_loss = criterion(x_prime_pred, x_prime)
        lin_loss = criterion(lin_error, zero_lin_error)
        return autoencoder_loss + pred_loss + alpha*lin_loss

    def construct_encoder_(self):
        input_dim = self.net_params['state_dim']
        hidden_dim = self.net_params['encoder_hidden_dim']
        output_dim = self.net_params['encoder_output_dim']

        self.encoder_fc_in = nn.Linear(input_dim, hidden_dim[0])
        self.encoder_fc_hid = []

        if len(hidden_dim) > 0:
            for ii in range(1, len(hidden_dim)):
                self.encoder_fc_hid.append(nn.Linear(hidden_dim[ii - 1], hidden_dim[ii]))
            self.encoder_fc_out = nn.Linear(hidden_dim[-1], output_dim)
        else:
            self.encoder_fc_out = nn.Linear(input_dim, output_dim)

    def construct_decoder_(self):
        input_dim = self.net_params['encoder_output_dim']
        hidden_dim = self.net_params['decoder_hidden_dim']
        output_dim = self.net_params['state_dim']

        self.decoder_fc_in = nn.Linear(input_dim, hidden_dim[0])
        self.decoder_fc_hid = []

        if len(hidden_dim) > 0:
            for ii in range(1, len(hidden_dim)):
                self.decoder_fc_hid.append(nn.Linear(hidden_dim[ii - 1], hidden_dim[ii]))
            self.decoder_fc_out = nn.Linear(hidden_dim[-1], output_dim)
        else:
            self.decoder_fc_out = nn.Linear(input_dim, output_dim)

    def encode_forward_(self, x):
        z = F.relu(self.encoder_fc_in(x))
        for layer in self.encoder_fc_hid:
            z = F.relu(layer(z))
        z = self.encoder_fc_out(z)

        return z

    def decode_forward_(self, z):
        x_pred = F.relu(self.decoder_fc_in(z))
        for layer in self.decoder_fc_hid:
            x_pred = F.relu(layer(x_pred))
        x_pred = self.decoder_fc_out(x_pred)

        return x_pred

    def encode(self, x):
        self.eval()
        x = torch.from_numpy(x).float()
        z = self.encode_forward_(x)
        return z.detach().numpy()

    def decode(self, z):
        self.eval()
        z = torch.from_numpy(z).float()
        x = self.decode_forward_(z)
        return x.detach().numpy()
