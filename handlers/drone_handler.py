from .handler import Handler

from numpy import zeros, random

class DroneHandler(Handler):
    """
    Class to use with Bintel Drone at CAST. Not used. See Bintel repository
    """
    def __init__(self,n,m,Nlift,Nep,w):
        super().__init__(n,m,Nlift,Nep,w)

    def run(self):
        pass

    def process(self):
        pass

    def get_ctrl(self, q, q_d):
        assert(q.shape[0] == self.n)
        assert(q_d.shape[0] == self.n)
        u_nom = zeros((self.m,q_d.shape[0]))
        for ii in range(len(self.controller_list)):
            u_nom += self.w[ii]*self.controller_list[ii](q, q_d, u_nom)

        u = u_nom + self.pert_noise*random.randn((u_nom.shape))

        return u, u_nom