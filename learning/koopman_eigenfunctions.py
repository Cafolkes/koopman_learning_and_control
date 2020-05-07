from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title
from numpy import array, linalg, transpose, diag, dot, ones, zeros, unique, power, prod, exp, log, divide, real, iscomplex, any, ones_like
from numpy import concatenate as npconcatenate
import numpy as np
from itertools import combinations_with_replacement, permutations
from .utils import differentiate_vec
from .basis_functions import BasisFunctions
from ..dynamics.linear_system_dynamics import LinearSystemDynamics
from ..controllers.constant_controller import ConstantController
from .diffeomorphism_net import DiffeomorphismNet
from torch import nn, cuda, optim, from_numpy, manual_seed, no_grad, save, load, cat, transpose as t_transpose
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
from torch.autograd.gradcheck import zero_gradients

class KoopmanEigenfunctions(BasisFunctions):
    """
    Class for construction and lifting using Koopman eigenfunctions
    """
    def __init__(self, n, max_power, A_cl, BK, traj_input=False):
        """KoopmanEigenfunctions 
        
        Arguments:
            BasisFunctions {basis_function} -- function to lift the state
            n {integer} -- number of states
            max_power {integer} -- maximum number to exponenciate each original principal eigenvalue
            A_cl {numpy array [Ns,Ns]} -- closed loop matrix in continuous time
            BK {numpy array [Ns,Nu]} -- control matrix 
        """
        self.n = n
        self.max_power = max_power
        self.A_cl = A_cl
        self.BK = BK
        self.traj_input = traj_input
        self.Nlift = None
        self.Lambda = None
        self.basis = None
        self.eigfuncs_lin = None  #Eigenfunctinos for linearized autonomous dynamics xdot = A_cl*x
        self.scale_func = None  #Scaling function scaling relevant state space into unit cube
        self.diffeomorphism_model = None

    def construct_basis(self, ub=None, lb=None):
        """construct_basis define basis functions
        
        Keyword Arguments:
            ub {numpy array [Ns,]} -- upper bound for unit scaling (default: {None})
            lb {numpy array [Ns,]} -- lower bound for unit scaling (default: {None})
        """
        self.eigfunc_lin = self.construct_linear_eigfuncs()
        self.scale_func = self.construct_scaling_function(ub,lb)
        self.basis = lambda q, t: self.eigfunc_lin(self.scale_func(self.diffeomorphism(q, t)))
        #print('Dimensional test: ', self.lift(ones((self.n,2))).shape)

    def construct_linear_eigfuncs(self):

        lambd, v = linalg.eig(self.A_cl)
        _, w = linalg.eig(transpose(self.A_cl))

        if any(iscomplex(lambd)) or any(iscomplex(w)):
            Warning("Complex eigenvalues and/or eigenvalues. Complex part supressed.")
            lambd = real(lambd)
            w = real(w)

        # Scale up w to get kronecker delta
        w_scaling = diag(dot(v.T, w))
        w = divide(w, w_scaling.reshape(1,w.shape[0]))

        p = array([ii for ii in range(self.max_power+1)])
        combinations = array(list(combinations_with_replacement(p, self.n)))
        powers = array([list(permutations(c,self.n)) for c in combinations]) # Find all permutations of powers
        powers = unique(powers.reshape((powers.shape[0] * powers.shape[1], powers.shape[2])),axis=0)  # Remove duplicates

        linfunc = lambda q: dot(transpose(w), q)  # Define principal eigenfunctions of the linearized system
        eigfunc_lin = lambda q: prod(power(linfunc(q), transpose(powers)), axis=0)  # Create desired number of eigenfunctions
        self.Nlift = eigfunc_lin(ones((self.n,1))).shape[0]
        self.Lambda = log(prod(power(exp(lambd).reshape((self.n,1)), transpose(powers)), axis=0))  # Calculate corresponding eigenvalues

        return eigfunc_lin

    def construct_scaling_function(self,ub,lb):
        scale_factor = (ub-lb).reshape((self.n,1))
        scale_func = lambda q: divide(q, scale_factor)

        return scale_func

    def diffeomorphism(self, q, q_d):
        q = q.transpose()
        q_d = q_d.transpose()
        self.diffeomorphism_model.eval()
        if self.traj_input:
            input = npconcatenate((q, q_d),axis=1)
        else:
            input = q
        diff_pred = self.diffeomorphism_model.predict(from_numpy(input))
        return (q + diff_pred).T

    def build_diffeomorphism_model(self, jacobian_penalty=1., n_hidden_layers=2, layer_width=50, batch_size=64, dropout_prob=0.1):
        """build_diffeomorphism_model 
        
        Keyword Arguments:
            n_hidden_layers {int} --  (default: {2})
            layer_width {int} --  (default: {50})
            batch_size {int} --  (default: {64})
            dropout_prob {float} --  (default: {0.1})
        """

        self.A_cl = from_numpy(self.A_cl)
        self.diffeomorphism_model = DiffeomorphismNet(self.n, self.A_cl, jacobian_penalty=jacobian_penalty,
                                                      n_hidden_layers=n_hidden_layers, layer_width=layer_width,
                                                      batch_size=batch_size, dropout_prob=dropout_prob, traj_input=self.traj_input)

    def fit_diffeomorphism_model(self, X, t, X_d, learning_rate=1e-2, learning_decay=0.95, n_epochs=50, train_frac=0.8, l2=1e1, batch_size=64, initialize=True, verbose=True, X_val=None, t_val=None, Xd_val=None):
        """fit_diffeomorphism_model 
        
        Arguments:
            X {numpy array [Ntraj,Nt,Ns]} -- state
            t {numpy array [Ntraj,Nt]} -- time vector
            X_d {numpy array [Ntraj,Nt,Ns]} -- desired state
        
        Keyword Arguments:
            learning_rate {[type]} --  (default: {1e-2})
            learning_decay {float} --  (default: {0.95})
            n_epochs {int} --  (default: {50})
            train_frac {float} -- ratio of training and testing (default: {0.8})
            l2 {[type]} -- L2 penalty term (default: {1e1})
            jacobian_penalty {[type]} --  (default: {1.})
            batch_size {int} --  (default: {64})
            initialize {bool} -- flag to warm start (default: {True})
            verbose {bool} --  (default: {True})
            X_val {numpy array [Ntraj,Nt,Ns]} -- state in validation set (default: {None})
            t_val {numpy array [Ntraj,Nt]} -- time in validation set (default: {None})
            Xd_val {numpy array [Ntraj,Nt,Ns]} -- desired state in validation set (default: {None})
        
        Returns:
            float -- val_losses[-1]
        """
        device = 'cuda' if cuda.is_available() else 'cpu'
        X, X_dot, X_d, X_d_dot, t = self.process(X=X, t=t, X_d=X_d)

        # Prepare data for pytorch:
        manual_seed(42)  # Fix seed for reproducibility
        if self.traj_input:
            X_tensor = from_numpy(npconcatenate((X, X_d, X_dot, X_d_dot, np.zeros_like(X)),axis=1)) #[x (1,n), x_d (1,n), x_dot (1,n), zeros (1,n)]
        else:
            X_tensor = from_numpy(npconcatenate((X, X_dot, np.zeros_like(X)), axis=1))  # [x (1,n), x_d (1,n), x_dot (1,n), zeros (1,n)]
        y_target = X_dot - (dot(self.A_cl, X.T) + dot(self.BK, X_d.T)).T
        y_tensor = from_numpy(y_target)
        X_tensor.requires_grad_(True)

        # Builds dataset with all data
        dataset = TensorDataset(X_tensor, y_tensor)

        if X_val is None or t_val is None or Xd_val is None:
            # Splits randomly into train and validation datasets
            n_train = int(train_frac*X.shape[0])
            n_val = X.shape[0]-n_train
            train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
            # Builds a loader for each dataset to perform mini-batch gradient descent
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)
        else:
            #Uses X,... as training data and X_val,... as validation data
            X_val, X_dot_val, Xd_val, Xd_dot_val, t_val = self.process(X=X_val, t=t_val, X_d=Xd_val)
            if self.traj_input:
                X_val_tensor = from_numpy(npconcatenate((X_val, Xd_val, X_dot_val, Xd_dot_val, np.zeros_like(X_val)),axis=1)) #[x (1,n), x_d (1,n), x_dot (1,n), zeros (1,n)]
            else:
                X_val_tensor = from_numpy(npconcatenate((X_val, X_dot_val, np.zeros_like(X_val)),axis=1))  # [x (1,n), x_dot (1,n), zeros (1,n)]
            y_target_val = X_dot_val - dot(self.A_cl, X_val.T + dot(self.BK, Xd_val.T)).T
            y_val_tensor = from_numpy(y_target_val)
            X_val_tensor.requires_grad_(True)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            # Builds a loader for each dataset to perform mini-batch gradient descent
            train_loader = DataLoader(dataset=dataset, batch_size=int(batch_size), shuffle=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=int(batch_size))

        # Set up optimizer and learning rate scheduler:
        optimizer = optim.Adam(self.diffeomorphism_model.parameters(),lr=learning_rate,weight_decay=l2)
        lambda1 = lambda epoch: learning_decay ** epoch
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        def make_train_step(model, loss_fn, optimizer):
            def train_step(x, y):
                model.train() # Set model to training mode
                y_pred = model(x)
                loss = loss_fn(y, y_pred, model.training)
                loss.backward()
                optimizer.step()
                return loss.item()
            return train_step

        batch_loss = []
        losses = []
        batch_val_loss = []
        val_losses = []
        train_step = make_train_step(self.diffeomorphism_model , self.diffeomorphism_model.diffeomorphism_loss, optimizer)

        # Initialize model weights:
        def init_normal(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

        if initialize:
            self.diffeomorphism_model.apply(init_normal)

        # Training loop
        for i in range(n_epochs):
            # Uses loader to fetch one mini-batch for training
            #print('Training epoch ', i)
            for x_batch, y_batch in train_loader:
                # Send mini batch data to same location as model:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                #print('Training: ', x_batch.shape, y_batch.shape)
                # Train based on current batch:
                batch_loss.append(train_step(x_batch, y_batch))
                optimizer.zero_grad()
            losses.append(sum(batch_loss)/len(batch_loss))
            batch_loss = []

            #print('Validating epoch ', i)
            with no_grad():
                for x_val, y_val in val_loader:
                    # Sends data to same device as model
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)

                    #print('Validation: ', x_val.shape, y_val.shape)

                    self.diffeomorphism_model.eval() # Change model model to evaluation
                    #xt_val = x_val[:, :2*self.n]  # [x, x_d]
                    #xdot_val = x_val[:, 2*self.n:]  # [xdot]
                    y_pred = self.diffeomorphism_model(x_val)  # Predict
                    #jacobian_xdot_val, zero_jacobian_val = calc_gradients(xt_val, xdot_val, yhat, None, None, self.diffeomorphism_model.training)
                    batch_val_loss.append(float(self.diffeomorphism_model.diffeomorphism_loss(y_val, y_pred, self.diffeomorphism_model.training))) # Compute validation loss
                val_losses.append(sum(batch_val_loss)/len(batch_val_loss))  # Save validation loss
                batch_val_loss = []

            scheduler.step(i)
            if verbose:
                print(' - Epoch: ',i,' Training loss:', format(losses[-1], '08f'), ' Validation loss:', format(val_losses[-1], '08f'))
                print('Improvement metric (for early stopping): ', sum(abs(array(val_losses[-min(3,len(val_losses)):])-val_losses[-1]))/(3*val_losses[-min(3,len(val_losses))]))
            if i > n_epochs/4 and sum(abs(array(val_losses[-min(3,len(val_losses)):])-val_losses[-1]))/(3*val_losses[-min(3,len(val_losses))]) < 0.01:
                #print('Early stopping activated')
                break

        return val_losses[-1]

    def process(self, X, t, X_d):
        # Shift dynamics to make origin a fixed point
        X_f = X_d[:,-1,:]
        X_shift = array([X[ii,:,:] - X_f[ii,:] for ii in range(len(X))])
        X_d = array([X_d[ii,:,:].reshape((X_d.shape[1],X_d.shape[2])) - X_f[ii,:] for ii in range(len(X))])

        # Calculate numerical derivatives
        X_dot = array([differentiate_vec(X_shift[ii, :, :], t[ii, :]) for ii in range(X_shift.shape[0])])
        X_d_dot = array([differentiate_vec(X_d[ii, :, :], t[ii, :]) for ii in range(X_d.shape[0])])

        assert(X_shift.shape == X_dot.shape)
        assert(X_d.shape == X_dot.shape)
        assert(X_d_dot.shape == X_dot.shape)
        assert(t.shape == X_shift[:,:,0].shape)

        # Reshape to have input-output data
        X_shift = X_shift.reshape((X_shift.shape[0]*X_shift.shape[1], X_shift.shape[2]))
        X_dot = X_dot.reshape((X_dot.shape[0] * X_dot.shape[1], X_dot.shape[2]))
        X_d = X_d.reshape((X_d.shape[0] * X_d.shape[1], X_d.shape[2]))
        X_d_dot = X_d_dot.reshape((X_d_dot.shape[0] * X_d_dot.shape[1], X_d_dot.shape[2]))
        t = t.reshape((t.shape[0] * t.shape[1],))

        return X_shift, X_dot, X_d, X_d_dot, t

    def save_diffeomorphism_model(self, filename):
        save(self.diffeomorphism_model.state_dict(), filename)

    def load_diffeomorphism_model(self, filename):
        self.diffeomorphism_model.load_state_dict(load(filename))

    def plot_eigenfunction_evolution(self, X, X_d, t):
        #X = X.transpose()
        #X_d = X_d.transpose()
        eigval_system = LinearSystemDynamics(A=diag(self.Lambda),B=zeros((self.Lambda.shape[0],1)))
        eigval_ctrl = ConstantController(eigval_system,0.)

        eigval_evo = []
        eigfunc_evo = []
        for ii in range(X.shape[0]):
            x0 = X[ii,:1,:].T
            x0_d = X_d[ii,:1,:].T
            z0 = self.lift(x0, x0_d)
            eigval_evo_tmp,_ = eigval_system.simulate(z0.flatten(), eigval_ctrl, t)
            eigval_evo_tmp = eigval_evo_tmp.transpose()
            eigfunc_evo_tmp = self.lift(X[ii,:,:].T, X_d[ii,:,:].T).transpose()
            eigval_evo.append(eigval_evo_tmp)
            eigfunc_evo.append(eigfunc_evo_tmp)

        # Calculate error statistics
        eigval_evo = array(eigval_evo)
        eigfunc_evo = array(eigfunc_evo)
        norm_factor = np.sum(np.sum(eigval_evo**2, axis=2), axis=0)
        norm_factor = ones_like(norm_factor)  #TODO: Remove if plotting normalization is desired
        eig_error = np.abs(eigval_evo - eigfunc_evo)
        eig_error_norm = array([eig_error[:,ii,:]/norm_factor[ii] for ii in range(eigval_evo.shape[1])])
        eig_error_mean = np.mean(eig_error_norm, axis=1)
        eig_error_std = np.std(eig_error_norm, axis=1)

        figure(figsize=(15,15))
        #suptitle('Eigenfunction VS Eigenvalue Evolution')
        for ii in range(1,26):
            subplot(5, 5, ii)
            plot(t, eig_error_mean[ii-1,:], linewidth=2, label='Mean')
            plot(t, eig_error_std[ii-1,:], linewidth=1, label='Standard dev')
            title('efunc ' + str(ii-1))
            grid()
        legend(fontsize=12)
        show()

    def lift(self, q, q_d):
        """lift 
        
        Arguments:
            q {numpy array [Ns,Nt]} -- state    
            q_d {numpy array [Ns,Nt]} -- desired state 
        
        Returns:
            [type] -- [description]
        """
        return array([self.basis(q[:,ii].reshape((self.n,1)), q_d[:,ii].reshape((self.n,1))) for ii in range(q.shape[1])])
