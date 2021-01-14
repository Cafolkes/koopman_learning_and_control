import numpy as np
from koopman_core.learning import KoopmanNet
import torch
import torch.optim as optim

class KoopDnn():
    '''
    Base class for edmd-type methods. Implements baseline edmd with the possible addition of l1 and/or l2 regularization.
    Overload fit for more specific methods.
    '''
    def __init__(self, n_traj, net_params, cv=None, standardizer=None, first_obs_const=True,
                 continuous_mdl=False, dt=None):
        self.n_traj = n_traj
        self.A = None
        self.B = None

        self.net_params = net_params
        self.koopman_net = KoopmanNet(self.net_params)
        self.optimizer = None
        self.set_optimizer_()
        self.C = self.koopman_net.C.data.numpy()

        self.first_obs_const = first_obs_const
        self.cv = cv
        self.standardizer = standardizer
        self.continuous_mdl = continuous_mdl
        self.dt = dt

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
                if i % 100 == 99:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.8f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        self.construct_dyn_mat_()
        self.construct_basis_()

    def construct_dyn_mat_(self):
        if self.continuous_mdl:
            self.construct_dyn_mat_continuous_()
        else:
            self.construct_dyn_mat_discrete_()

    def construct_basis_(self):
        self.basis_encode = lambda x: self.koopman_net.encode(np.atleast_2d(x))

    def construct_dyn_mat_continuous_(self):
        pass

    def construct_dyn_mat_discrete_(self):
        n = self.net_params['state_dim']
        m = self.net_params['ctrl_dim']
        encoder_output_dim = self.net_params['encoder_output_dim']
        first_obs_const = self.net_params['first_obs_const']
        n_tot = n + encoder_output_dim + int(first_obs_const)
        n_kinematics = int(first_obs_const) + int(n/2)
        dt = self.net_params['dt']

        self.A = self.koopman_net.koopman_fc_drift.weight.data.numpy()
        B_vec = self.koopman_net.koopman_fc_act.weight.data.numpy()

        if self.net_params['override_kinematics']:
            self.A = np.concatenate((np.zeros((n_kinematics, n_tot)), self.A), axis=0)
            self.A += np.eye(self.A.shape[0])
            self.A[int(first_obs_const):n_kinematics, n_kinematics:n_kinematics+int(n/2)] += np.eye(int(n/2))*dt

            self.B = [np.concatenate((np.zeros((n_kinematics, n_tot)), B_vec[:, n_tot*ii:n_tot*(ii+1)]), axis=0)
                      for ii in range(m)]

        else:
            self.A += np.eye(self.A.shape[0])
            self.B = [B_vec[:, n_tot * ii:n_tot * (ii + 1)] for ii in range(m)]

    def process(self, data_x, data_u, t, downsample_rate=1):
        n = self.net_params['state_dim']
        assert data_x.shape[0] == self.n_traj
        assert data_x.shape[2] == n

        m = self.net_params['ctrl_dim']
        traj_length = data_x.shape[1]
        n_multistep = self.net_params['n_multistep']

        #x = data_x[:,:-1,:]
        #u = data_u
        #x_prime = data_x[:,1:,:]
        x = np.zeros((self.n_traj, traj_length-n_multistep, n*n_multistep))
        u = np.zeros((self.n_traj, traj_length-n_multistep, m*n_multistep))
        x_prime = np.zeros((self.n_traj, traj_length - n_multistep, n * n_multistep))
        for ii in range(n_multistep):
            x[:, :, n*ii:n*(ii+1)] = data_x[:, ii:-(n_multistep-ii), :]
            if ii + 1 < n_multistep:
                u[:, :, m*ii:m*(ii+1)] = data_u[:, ii:-(n_multistep-ii-1), :]
                x_prime[:, :, n*ii:n*(ii+1)] = data_x[:, ii+1:-(n_multistep - ii - 1), :]
            else:
                u[:, :, m * ii:m * (ii + 1)] = data_u[:, ii:, :]
                x_prime[:, :, n*ii:n*(ii+1)] = data_x[:, ii+1:, :]

        order = 'F'
        n_data_pts = self.n_traj * (t[0,:].shape[0] - n_multistep)
        x_flat = x.T.reshape((n*n_multistep, n_data_pts), order=order)
        u_flat = u.T.reshape((m*n_multistep, n_data_pts), order=order)
        x_prime_flat = x_prime.T.reshape((n*n_multistep, n_data_pts), order=order)

        if self.standardizer is None:
            x_flat, u_flat, x_prime_flat = x_flat.T, u_flat.T, x_prime_flat.T
        else:
            pass
            #self.standardizer.fit(x_flat.T)
            #x_flat, x_prime_flat = self.standardizer.transform(x_flat.T), x_prime_flat.T

        X = np.concatenate((x_flat, u_flat, x_prime_flat), axis=1)

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
