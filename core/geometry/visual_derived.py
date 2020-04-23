from .derived import Derived
from .visual import Visual

class VisualDerived(Visual, Derived):
    def __init__(self, visual_primitive, T, c):
        Derived.__init__(self, visual_primitive, T, c)

    def boundary(self, N):
        return self.from_primitive(self.primitive.boundary(N))
