from numpy import dot
from numpy.linalg import inv

from .convex_body import ConvexBody
from ..dynamics import AffineDynamics

class Derived(ConvexBody):
    def __init__(self, primitive, T, c):
        ConvexBody.__init__(self, primitive.dim)
        self.primitive = primitive
        self.T = T
        self.T_inv = inv(T)
        self.c = c
        
    def from_primitive(self, xs):
        return dot(self.T_inv, xs.T).T + self.c
    
    def to_primitive(self, xs):
        return dot(self.T, (xs - self.c).T).T
    
    def sample(self, N):
        return self.from_primitive(self.primitive.sample(N))
    
    def safety(self, affine_dynamics):
        derived_dynamics = self.DerivedDynamics(self, affine_dynamics)
        return self.primitive.safety(derived_dynamics)
    
    class DerivedDynamics(AffineDynamics):
        def __init__(self, derived, affine_dynamics):
            self.derived = derived
            self.dynamics = affine_dynamics
        
        def eval(self, x, t):
            return self.derived.to_primitive(self.dynamics.eval(x, t))
        
        def drift(self, x, t):
            return dot(self.derived.T, self.dynamics.drift(x, t))
        
        def act(self, x, t):
            return dot(self.derived.T, self.dynamics.act(x, t))