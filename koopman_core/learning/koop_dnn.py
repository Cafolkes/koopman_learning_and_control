import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch.nn.utils import prune
from ray import tune
from koopman_core.learning.utils import ThresholdPruning
import os

class KoopDnn():
    '''
    Class for neural network-based Koopman methods to learn dynamics models of autonomous and controlled dynamical systems.
    '''
    def __init__(self, net, first_obs_const=True,
                 continuous_mdl=False, dt=None):
        self.A = None
        self.B = None

        self.net = net
        self.optimizer = None
        self.C = None

        self.first_obs_const = first_obs_const
        self.continuous_mdl = continuous_mdl
        self.dt = dt

        self.x_train = None
        self.u_train = None
        self.t_train = None
        self.x_val = None
        self.u_val = None
        self.t_val = None

    def set_datasets(self, x_train, t_train, u_train=None, x_val=None, t_val=None, u_val=None):
        self.x_train = x_train
        self.t_train = np.tile(t_train, (self.x_train.shape[0], 1))
        self.u_train = u_train

        self.x_val = x_val
        self.t_val = np.tile(t_val, (self.x_val.shape[0], 1))
        self.u_val = u_val

    def model_pipeline(self, net_params, print_epoch=True, tune_run=False, early_stop=False, plot_data=False):
        self.net.net_params = net_params
        self.net.construct_net()
        self.set_optimizer_()

        X_train, y_train = self.net.process(self.x_train, self.t_train, data_u=self.u_train)
        X_val, y_val = self.net.process(self.x_val, self.t_val, data_u=self.u_val)

        if plot_data:
            self.plot_train_data_(X_train, y_train)

        X_train_t, y_train_t = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
        X_val_t, y_val_t = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
        dataset_train = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        dataset_val = torch.utils.data.TensorDataset(X_val_t, y_val_t)

        self.train_model(dataset_train, dataset_val, print_epoch=print_epoch, tune_run=tune_run, early_stop=early_stop)

    def train_model(self, dataset_train, dataset_val, print_epoch=True, tune_run=False, early_stop=False,
                    early_stop_crit=1e-3, early_stop_max_count=5):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:0'
        self.net.send_to(device)

        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=self.net.net_params['batch_size'],
                                                  shuffle=True, num_workers=0, pin_memory=True)
        valloader = torch.utils.data.DataLoader(dataset_val, batch_size=self.net.net_params['batch_size'],
                                                shuffle=True, num_workers=0, pin_memory=True)

        val_loss_prev = np.inf
        no_improv_counter = 0
        for epoch in range(self.net.net_params['epochs']):
            running_loss = 0.0
            epoch_steps = 0


            for data in trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                if epoch > int(self.net.net_params['epochs'] * 0.8):
                    prune.global_unstructured(
                        self.net.parameters_to_prune, pruning_method=ThresholdPruning, threshold=1e-5
                    )

                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.net.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.detach()
                epoch_steps += 1

            # Validation loss:
            val_loss = 0.0
            val_steps = 0
            for data in valloader:
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = self.net(inputs)
                    loss = self.net.loss(outputs, labels)
                    val_loss += float(loss.detach())
                    val_steps += 1

            # Print epoch loss:
            if print_epoch:
                print('Epoch %3d: train loss: %.8f, validation loss: %.8f' %(epoch + 1, running_loss/epoch_steps, val_loss/val_steps))

            # Early stop if no improvement:
            if early_stop:
                improvement = (val_loss/val_steps)/val_loss_prev
                if improvement >= 1 - early_stop_crit and improvement <= 1+early_stop_crit:
                    no_improv_counter += 1
                else:
                    no_improv_counter = 0

                if no_improv_counter >= early_stop_max_count:
                    print('Early stopping activated, less than %.4f improvement for the last %2d epochs.'
                          %(early_stop_crit, no_improv_counter))
                    break
                val_loss_prev = val_loss/val_steps

            # Save Ray Tune checkpoint:
            if tune_run:
                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((self.net.state_dict(), self.optimizer.state_dict()), path)
                tune.report(loss=(val_loss / val_steps))


        print("Finished Training")
        prune.global_unstructured(
            self.net.parameters_to_prune, pruning_method=ThresholdPruning, threshold=1e-5
        )
        prune.remove(self.net.koopman_fc_drift, 'weight')
        prune.remove(self.net.koopman_fc_act, 'weight')

    def test_loss(self, x_test, t_test, u_test=None):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:0'
        self.net.send_to(device)

        if u_test is None:
            X_test, y_test = self.net.process(x_test, np.tile(t_test, (x_test.shape[0], 1)))
        else:
            X_test, y_test = self.net.process(x_test, np.tile(t_test, (x_test.shape[0], 1)), data_u=u_test)

        X_t, y_t = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
        dataset_test = torch.utils.data.TensorDataset(X_t, y_t)
        testloader = torch.utils.data.DataLoader(dataset_test, batch_size=self.net.net_params['batch_size'], shuffle=True)

        test_loss = 0.0
        test_steps = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.net(inputs)
                loss = self.net.loss(outputs, labels, validation=True)
                test_loss += loss.detach()
                test_steps += 1

        return test_loss/test_steps

    def construct_koopman_model(self):
        self.net.send_to('cpu')
        self.construct_dyn_mat_()
        self.C = np.array(self.net.C)

    def construct_dyn_mat_(self):
        self.net.construct_dyn_mat()
        self.A = self.net.A
        try:
            self.B = self.net.B
        except AttributeError:
            pass

    def basis_encode(self, x):
        if self.net.standardizer_x is None:
            x_scaled = np.atleast_2d(x)
        else:
            x_scaled = self.net.standardizer_x.transform(x)

        return self.net.encode(x_scaled)

    def set_optimizer_(self):
        if self.net.net_params['optimizer'] == 'sgd':
            lr = self.net.net_params['lr']
            momentum = self.net.net_params['momentum']
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
        elif self.net.net_params['optimizer'] == 'adam':
            lr = self.net.net_params['lr']
            weight_decay = self.net.net_params['l2_reg']
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

    def plot_train_data_(self, X, y):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(X[:, 2], X[:, 3], y[:, 2], color="orange")
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel("$x_1'$")
        ax.set_title('One-step-ahead state, $x_1$')
        ax.view_init(30, 70)

        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(X[:, 2], X[:, 3], y[:, 3], color="orange")
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel("$x_2'$")
        ax.set_title('One-step-ahead state, $x_2$')
        ax.view_init(30, 70)
        plt.show()


