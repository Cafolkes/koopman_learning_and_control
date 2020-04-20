from matplotlib.pyplot import figure
from numpy import arange, array, dot, reshape, zeros
from numpy.linalg import solve

def arr_map(func, *arr):
    return array(list(map(func, *arr)))

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

def default_fig(fig, ax):
    if fig is None:
        fig = figure(figsize=(6, 6), tight_layout=True)

    if ax is None:
        ax = fig.add_subplot(1, 1, 1)

    return fig, ax
