import numpy as np
import diagnostics as diag
from scipy.interpolate import RegularGridInterpolator as rgi


def pad_grid(xg):
    if xg.size == 1:
        l_point = [xg[0]-1]
        r_point = [xg[0]+1]
    else:
        l_point = [2*xg[0] - xg[1]]
        r_point = [2*xg[-1] - xg[-2]]
    return np.concatenate((l_point, xg, r_point))


def pad_array(x):
    return np.pad(x, (1, 1), 'wrap')


def generate_grid(Ns, Ls):
    Xs = []
    for i in range(3):
        # z = 0, y = 1, x = 2
        Xe = np.linspace(0, Ls[i], Ns[i]+1)
        # Get cell-centered coordinates and extend
        Xg = 0.5*(Xe[1:] + Xe[:-1])
        Xs.append(Xg)
    return np.meshgrid(*Xs, indexing='ij')


def reinterpolate(data, old_Xgrid, new_Ns, Ls):
    xg, yg, zg = old_Xgrid
    xg, yg, zg = pad_grid(xg), pad_grid(yg), pad_grid(zg)
    data_interp = rgi((zg, yg, xg), pad_array(data))

    Zg_hires, Yg_hires, Xg_hires = generate_grid(new_Ns, Ls)
    pts = np.array([Zg_hires.ravel(), Yg_hires.ravel(), Xg_hires.ravel()]).T
    data_hires = data_interp(pts).reshape(*new_Ns)
    return data_hires


