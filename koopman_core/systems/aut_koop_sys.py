from numpy import array, zeros, random, dot
from core.dynamics import SystemDynamics

class AutKoopSys(SystemDynamics):
    def __init__(self, mu, lambd, B_w=zeros((2,2)), pn_mean=zeros(2), pn_var=zeros(2), pn_min=zeros(2), pn_max=zeros(2),
                 pn_type=None):
        SystemDynamics.__init__(self, 2, 0)
        self.params = mu, lambd
        self.B_w = B_w
        self.pn_mean = pn_mean
        self.pn_var = pn_var
        self.pn_type = pn_type
        self.pn_min = pn_min
        self.pn_max = pn_max

    def drift(self, x, t):
        mu, lambd = self.params

        if self.pn_type == 'normal':
            w = random.normal(self.pn_mean, self.pn_var, (2,1))
            process_noise = dot(self.B_w, w).squeeze()
        elif self.pn_type == 'uniform':
            w = random.uniform(self.pn_min, self.pn_max, (2,1))
            process_noise = dot(self.B_w, w).squeeze()
        else:
            process_noise = zeros(2)

        return array([mu*x[0], -lambd*x[0]**2 + lambd*x[1]]) + process_noise

    def eval_dot(self, x, u, t):
        return self.drift(x, t)