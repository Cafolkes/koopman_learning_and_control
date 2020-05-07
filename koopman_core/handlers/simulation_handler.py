from .handler import Handler
from ..controllers import AggregatedMpcController
from numpy import array, random

class SimulationHandler(Handler):
    def __init__(self,n,m,Nlift,Nep,w,initial_controller,pert_noise,dynamics, q_d, t_d):
        """SimulationHandler
        
        Arguments:
            n {integer} -- Ns
            m {integer} -- number of commands, Nu
            Nlift {integer} -- number of lifting functions
            Nep {integer} -- number of episodes
            w {numpy array [Nep,]} -- weights for the controllers
            initial_controller {controller} -- initial controller
            pert_noise {float>0} -- noise to learn controllers
            dynamics {dynamical system} -- dynamics
            q_d {numpy array [Ns,Nt]} -- desired trajectory
            t_d {numpy array [Nt,]} -- time vector for the desired trajectory
        """
        super().__init__(n,m,Nlift,Nep,w,initial_controller,pert_noise)
        self.dynamical_system = dynamics
        self.q_d = q_d
        self.t_d = t_d

    def run(self):
        """run Run one episode
        
        Returns:
            numpy array [?][][][] -- state, desired state, control, nominal control, time
        """
        controller_agg = AggregatedMpcController(self.dynamical_system, self.controller_list, self.weights, self.pert_noise)
        x0 = self.q_d[:,:1] + 0.05*random.randn(self.q_d.shape[0],1)
        xs, us = self.dynamical_system.simulate(x0.squeeze(), controller_agg, self.t_d)
        us_nom = array(controller_agg.u_pert_lst)

        return xs, self.q_d.transpose(), us, us_nom, self.t_d

    def process(self, X, Xd, U, Upert, t):
        """process Remove the perturbation from the total controller to get nominal controller
        
        Arguments:
            X {numpy array []} -- state
            Xd {numpy array []} -- desired state
            U {numpy arrray []} -- command
            Upert {numpy array []} -- perturbation command
            t {numpy array [Nt,]} -- time vector
        
        Returns:
            numpy array [][][][] -- state, desired state, control, nominal control, times
        """
        Unom = U-Upert
        return X, Xd, U, Unom, t

    def get_ctrl(self, q, q_d):
        pass