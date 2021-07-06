import numpy as np
from koopman_core.learning.koopman_net_aut import KoopmanNetAut
import torch
import torch.optim as optim
from torch.utils.data import random_split

class KoopDnnAut():
    '''
    Base class for edmd-type methods. Implements baseline edmd with the possible addition of l1 and/or l2 regularization.
    Overload fit for more specific methods.
    '''
    def __init__(self, n_traj, net_params, cv=None, standardizer=None, first_obs_const=True, continuous_mdl=False, dt=None):
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

        self.x_trainval = None
        self.t_eval = None
        # TODO: Handle bias terms if one should be added to observables
        # TODO: Support adding state to observables
        # TODO: Support overriding kinematics

    def set_datasets(self, x_trainval, t_eval):
        self.x_trainval = x_trainval
        self.t_eval = t_eval

    def model_pipeline(self, net_params, val_frac=0.2, print_epoch=True, tune_run=False, early_stop=False):
        self.net_params = net_params
        self.set_optimizer_()
        self.koopman_net.net_params = net_params

        X_kdnn, y_kdnn = self.process(self.x_trainval, np.tile(self.t_eval, (self.x_trainval.shape[0], 1)))
        X_t, y_t = torch.from_numpy(X_kdnn).float(), torch.from_numpy(y_kdnn).float()
        dataset_trainval = torch.utils.data.TensorDataset(X_t, y_t)

        val_abs = int(len(dataset_trainval) * val_frac)
        dataset_train, dataset_val = random_split(dataset_trainval, [len(dataset_trainval) - val_abs, val_abs])
        self.train_model(dataset_train, dataset_val, print_epoch=print_epoch, tune_run=tune_run, early_stop=early_stop)

    def train_model(self, dataset_train, dataset_val, print_epoch=True, tune_run=False, early_stop=False,
                    early_stop_crit=5e-4, early_stop_max_count=5):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:0'
        self.koopman_net.send_to(device)

        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=self.net_params['batch_size'], shuffle=True)
        valloader = torch.utils.data.DataLoader(dataset_val, batch_size=self.net_params['batch_size'], shuffle=True)

        val_loss_prev = np.inf
        no_improv_counter = 0
        for epoch in range(self.net_params['epochs']):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.koopman_net(inputs)
                loss = self.koopman_net.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1

            # Validation loss:
            val_loss = 0.0
            val_steps = 0
            for i, data in enumerate(valloader):
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = self.koopman_net(inputs)
                    loss = self.koopman_net.loss(outputs, labels)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

            # Print epoch loss:
            if print_epoch:
                print('Epoch %3d: train loss: %.8f, validation loss: %.8f' % (
                epoch + 1, running_loss / epoch_steps, val_loss / val_steps))

            # Early stop if no improvement:
            if early_stop:
                if (val_loss / val_steps) / val_loss_prev >= 1 - early_stop_crit:
                    no_improv_counter += 1
                else:
                    no_improv_counter = 0

                if no_improv_counter >= early_stop_max_count:
                    print('Early stopping activated, less than %.4f improvement for the last %2d epochs.'
                          % (early_stop_crit, no_improv_counter))
                    break
                val_loss_prev = val_loss / val_steps

            # Save Ray Tune checkpoint:
            if tune_run:
                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((self.koopman_net.state_dict(), self.optimizer.state_dict()), path)
                tune.report(loss=(val_loss / val_steps))

        print("Finished Training")

    def test_loss(self, x_test, t_eval_test):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:0'
        self.koopman_net.send_to(device)

        X_kdnn, y_kdnn = self.process(x_test, np.tile(t_eval_test, (x_test.shape[0], 1)))
        X_t, y_t = torch.from_numpy(X_kdnn).float(), torch.from_numpy(y_kdnn).float()
        dataset_test = torch.utils.data.TensorDataset(X_t, y_t)
        testloader = torch.utils.data.DataLoader(dataset_test, batch_size=self.net_params['batch_size'], shuffle=True)

        test_loss = 0.0
        test_steps = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.koopman_net(inputs)
                loss = self.koopman_net.loss(outputs, labels)
                test_loss += loss.cpu().numpy()
                test_steps += 1

        return test_loss/test_steps

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

        traj_length = data.shape[1]
        n_multistep = self.net_params['n_multistep']
        x = np.zeros((self.n_traj, traj_length-n_multistep, n*n_multistep))
        x_prime = np.zeros((self.n_traj, traj_length - n_multistep, n * n_multistep))
        for ii in range(n_multistep):
            x[:, :, n*ii:n*(ii+1)] = data[:, ii:-(n_multistep-ii), :]
            if ii + 1 < n_multistep:
                x_prime[:, :, n*ii:n*(ii+1)] = data[:, ii+1:-(n_multistep - ii - 1), :]
            else:
                x_prime[:, :, n*ii:n*(ii+1)] = data[:, ii+1:, :]

        order = 'F'
        n_data_pts = self.n_traj * (t[0,:].shape[0] - n_multistep)
        x_flat = x.T.reshape((n*n_multistep, n_data_pts), order=order)
        x_prime_flat = x_prime.T.reshape((n*n_multistep, n_data_pts), order=order)

        if self.standardizer is None:
            x_flat, x_prime_flat = x_flat.T, x_prime_flat.T
        else:
            pass
            #self.standardizer.fit(x_flat.T)
            #x_flat, x_prime_flat = self.standardizer.transform(x_flat.T), x_prime_flat.T

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
