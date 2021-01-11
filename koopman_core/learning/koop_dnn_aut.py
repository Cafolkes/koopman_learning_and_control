from .utils import differentiate_vec
import numpy as np
from koopman_core.learning.koopman_net_aut import KoopmanNetAut
import torch
import torch.optim as optim

class KoopDnnAut():
    '''
    Base class for edmd-type methods. Implements baseline edmd with the possible addition of l1 and/or l2 regularization.
    Overload fit for more specific methods.
    '''
    def __init__(self, n_traj, net_params, cv=None, standardizer=None, C=None, first_obs_const=True,
                 continuous_mdl=False, dt=None):
        self.n_traj = n_traj
        self.A = None

        self.net_params = net_params
        self.koopman_net = KoopmanNetAut(self.net_params)
        self.optimizer = None
        self.set_optimizer_()
        self.C = self.koopman_net.C.data.numpy()

        self.first_obs_const = first_obs_const
        self.cv = cv
        self.standardizer = standardizer
        self.continuous_mdl = continuous_mdl
        self.dt = dt

        # TODO: Handle bias terms if one should be added to observables
        # TODO: Support adding state to observables
        # TODO: Support overriding kinematics

    def fit(self, X, y, cv=False, override_kinematics=False):

        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).float()
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=self.net_params['batch_size'], shuffle=True)

        for epoch in range(self.net_params['epochs']):
            running_loss = 0.0
            for i, data in enumerate(trainloader):
                inputs, labels = data

                self.optimizer.zero_grad()
                outputs = self.koopman_net(inputs)
                loss = self.koopman_net.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 199:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.8f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        self.construct_dyn_mat_()
        self.construct_basis_()

    def construct_dyn_mat_(self):
        self.A = self.koopman_net.koopman_fc.weight.data.numpy()
        self.A += np.eye(self.A.shape[0])

        #self.C = self.koopman_net.decoder_fc_out.weight.data.numpy().T

    def construct_basis_(self):
        self.basis_encode = lambda x: self.koopman_net.encode(np.atleast_2d(x))
        self.basis_decode = lambda z: self.koopman_net.decode(np.atleast_2d(z))

    def construct_dyn_mat_continuous_(self, coefs):
        pass

    def construct_dyn_mat_discrete_(self, coefs):
        pass

    def process(self, data, t, downsample_rate=1):
        n = self.net_params['state_dim']
        assert data.shape[0] == self.n_traj
        assert data.shape[2] == n

        x = data[:,:-1,:]
        x_prime = data[:,1:,:]

        order = 'F'
        n_data_pts = self.n_traj * (t[0,:].shape[0] - 1)
        x_flat = x.T.reshape((n, n_data_pts), order=order)
        x_prime_flat = x_prime.T.reshape((n, n_data_pts), order=order)

        if self.standardizer is None:
            x_flat, x_prime_flat = x_flat.T, x_prime_flat.T
        else:
            self.standardizer.fit(x_flat.T)
            x_flat, x_prime_flat = self.standardizer.transform(x_flat.T), x_prime_flat.T

        X = np.concatenate((x_flat, x_prime_flat), axis=1)
        return X[::downsample_rate,:], x_prime_flat[::downsample_rate,:]

    def predict(self, x):
        pass

    def score(self, x, u):
        pass

    def lift(self, x):
        pass

    def set_optimizer_(self):
        if self.net_params['optimizer'] == 'sgd':
            lr = self.net_params['lr']
            momentum = self.net_params['momentum']
            self.optimizer = optim.SGD(self.koopman_net.parameters(), lr=lr, momentum=momentum)
        elif self.net_params['optimizer'] == 'adam':
            lr = self.net_params['lr']
            weight_decay = self.net_params['weight_decay']
            self.optimizer = optim.Adam(self.koopman_net.parameters(), lr=lr, weight_decay=weight_decay)
        # TODO: Implement other optimizers as needed
