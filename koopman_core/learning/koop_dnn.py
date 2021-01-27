import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import random_split
from ray import tune
import os

class KoopDnn():
    '''
    Class for neural network-based Koopman methods to learn dynamics models of autonomous and controlled dynamical systems.
    '''
    def __init__(self, net, standardizer=None, first_obs_const=True,
                 continuous_mdl=False, dt=None):
        self.A = None
        self.B = None

        self.net = net
        self.optimizer = None
        #self.set_optimizer_()
        self.C = self.net.C.data.numpy()

        self.first_obs_const = first_obs_const
        self.standardizer = standardizer
        self.continuous_mdl = continuous_mdl
        self.dt = dt

        self.x_trainval = None
        self.u_trainval = None
        self.t_eval = None

    def set_datasets(self, x_trainval, t_eval, u_trainval=None):
        self.x_trainval = x_trainval
        self.t_eval = t_eval
        self.u_trainval = u_trainval

    def model_pipeline(self, net_params, val_frac=0.2, print_epoch=True, tune_run=False, early_stop=False):
        #self.net_params = net_params
        self.net.net_params = net_params
        self.set_optimizer_()

        if self.u_trainval is None:
            X_kdnn, y_kdnn = self.net.process(self.x_trainval, np.tile(self.t_eval, (self.x_trainval.shape[0], 1)))
        else:
            X_kdnn, y_kdnn = self.net.process(self.x_trainval, self.u_trainval, np.tile(self.t_eval, (self.x_trainval.shape[0], 1)))

        X_t, y_t = torch.from_numpy(X_kdnn).float(), torch.from_numpy(y_kdnn).float()
        dataset_trainval = torch.utils.data.TensorDataset(X_t, y_t)

        val_abs = int(len(dataset_trainval) * val_frac)
        dataset_train, dataset_val = random_split(dataset_trainval, [len(dataset_trainval) - val_abs, val_abs])
        self.train_model(dataset_train, dataset_val, print_epoch=print_epoch, tune_run=tune_run, early_stop=early_stop)

    def train_model(self, dataset_train, dataset_val, print_epoch=True, tune_run=False, early_stop=False, early_stop_crit=5e-4, early_stop_max_count=5):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:0'
        self.net.send_to(device)

        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=self.net.net_params['batch_size'], shuffle=True)
        valloader = torch.utils.data.DataLoader(dataset_val, batch_size=self.net.net_params['batch_size'], shuffle=True)

        val_loss_prev = np.inf
        no_improv_counter = 0
        for epoch in range(self.net.net_params['epochs']):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.net.loss(outputs, labels)
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

                    outputs = self.net(inputs)
                    loss = self.net.loss(outputs, labels, validation=True)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

            # Print epoch loss:
            if print_epoch:
                print('Epoch %3d: train loss: %.8f, validation loss: %.8f' %(epoch + 1, running_loss/epoch_steps, val_loss/val_steps))

            # Early stop if no improvement:
            if early_stop:
                if (val_loss/val_steps)/val_loss_prev >= 1 - early_stop_crit:
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

    def test_loss(self, x_test, t_eval_test, u_test=None):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:0'
        self.net.send_to(device)

        if self.u_trainval is None:
            X_kdnn, y_kdnn = self.net.process(x_test, np.tile(t_eval_test, (x_test.shape[0], 1)))
        else:
            X_kdnn, y_kdnn = self.net.process(x_test, u_test, np.tile(t_eval_test, (x_test.shape[0], 1)))

        X_t, y_t = torch.from_numpy(X_kdnn).float(), torch.from_numpy(y_kdnn).float()
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
                test_loss += loss.cpu().numpy()
                test_steps += 1

        return test_loss/test_steps

    def construct_koopman_model(self):
        self.construct_dyn_mat_()
        self.construct_basis_()

    def construct_dyn_mat_(self):
        self.net.send_to('cpu')
        if self.continuous_mdl:
            self.net.construct_dyn_mat_continuous()
        else:
            self.net.construct_dyn_mat_discrete()
        self.A = self.net.A
        try:
            self.B = self.net.B
        except AttributeError:
            pass

    def construct_basis_(self):
        self.basis_encode = lambda x: self.net.encode(np.atleast_2d(x))

    def set_optimizer_(self):
        if self.net.net_params['optimizer'] == 'sgd':
            lr = self.net.net_params['lr']
            momentum = self.net.net_params['momentum']
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
        elif self.net.net_params['optimizer'] == 'adam':
            lr = self.net.net_params['lr']
            weight_decay = self.net.net_params['l2_reg']
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
            # TODO: How does net.parameters() command know about all weights that are supposed to be optimized?

