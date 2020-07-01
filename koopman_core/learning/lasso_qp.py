import cvxpy as cvx
import time

class LassoQp:
    def __init__(self, alpha=None):
        self.alpha=alpha
        self.coef_ = None

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0], 'Input data has inconsistent shape'
        start_time = time.time()
        A = cvx.Variable((y.shape[1], X.shape[1]))

        cost = cvx.norm(y.T - A@X.T, 'fro')
        if self.alpha is not None:
            cost += self.alpha * cvx.norm(A, p=1)

        mosek_params = {'MSK_IPAR_NUM_THREADS': 8}
        prob = cvx.Problem(cvx.Minimize(cost))
        prob.solve(solver=cvx.MOSEK, mosek_params=mosek_params)

        self.coef_ = A.value
        print('Training time: ', time.time()-start_time)