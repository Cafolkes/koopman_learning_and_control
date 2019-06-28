from core.dynamics import AffineDynamics, ScalarDynamics

def differentiate(xs, ts, L=3):
    half_L = (L - 1) // 2
    b = zeros(L)
    b[1] = 1

    def diff(xs, ts):
        t_0 = ts[half_L]
        t_diffs = reshape(ts - t_0, (L, 1))
        pows = reshape(arange(L), (1, L))
        A = (t_diffs ** pows).T
        w = solve(A, b)
        return dot(w, xs)

    return array([diff(xs[k - half_L:k + half_L + 1], ts[k - half_L:k + half_L + 1]) for k in range(half_L, len(ts) - half_L)])

class AffineResidualDynamics(AffineDynamics):
    def __init__(self, affine_dynamics, drift_res, act_res):
        self.dynamics = affine_dynamics
        self.drift_res = drift_res
        self.act_res = act_res

    def eval(self, x, t):
        return self.dynamics.eval(x, t)

    def drift(self, x, t):
        return self.dynamics.drift(x, t) + self.drift_res(x, t)

    def act(self, x, t):
        return self.dynamics.act(x, t) + self.act_res(x, t)

class ScalarResidualDynamics(AffineResidualDynamics, ScalarDynamics):
    def __init__(self, scalar_dynamics, drift_res, act_res):
        AffineResidualDynamics.__init__(self, scalar_dynamics, drift_res, act_res)
