import numpy as np
from numpy.random import uniform
from scipy.interpolate import RegularGridInterpolator as rgi

import diagnostics as diag

def get_grid_info(data, a):
    Ns = data['RootGridSize'][::-1]
    Ls = []
    for i in range(1, 4):
        string = 'RootGridX' + str(i)
        L = data[string][1] - data[string][0]
        # Expand y and z lengths
        L *= 1 if i == 1 else a
        Ls.append(L)
    Ls = np.array(Ls[::-1])
    return Ns, Ls

def pad_grid(Xg):
    # Extending the grid one grid point before 0
    # and one grid point after Ls
    # Helps with interpolation of a periodic box
    l_point = [2*Xg[0] - Xg[1]]
    r_point = [2*Xg[-1] - Xg[-2]]
    return np.concatenate((l_point, Xg, r_point))

def generate_grid(Ns, Ls):
    Xs = []
    for i in range(3):
        # z = 0, y = 1, x = 2
        Xe = np.linspace(0, Ls[i], Ns[i]+1)
        # Get cell-centered coordinates and extend
        Xg = pad_grid(0.5*(Xe[1:] + Xe[:-1]))
        Xs.append(Xg)
    return Xs


def pad_array(x):
    # Pad data arrays at edges in order to make 'periodic'
    # Helps ensure that the interpolation returns a valid result at box edges.
    return np.pad(x, (1, 1), 'wrap')



def flyby(output_dir, flyby_a, flyby_n, do_rand_start=1, l_start=None):
    
    # assuming always from_array
    data = diag.load_data(output_dir, flyby_n, prob='from_array')
    Ns, Ls = get_grid_info(data, flyby_a)
    
    zg, yg, xg = generate_grid(Ns, Ls)
    Zg, Yg, Xg = np.meshgrid(zg, yg, xg, indexing='ij')
    
    # expand perpendicular values
    Bx = pad_array(data['Bcc1'])
    By = flyby_a*pad_array(data['Bcc2'])
    Bz = flyby_a*pad_array(data['Bcc3'])
    rho = pad_array(data['rho'])
    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

    # interpolaters
    Bx_i = rgi((zg, yg, xg), Bx)
    By_i = rgi((zg, yg, xg), By)
    Bz_i = rgi((zg, yg, xg), Bz)
    Bmag_i = rgi((zg, yg, xg), Bmag)
    rho_i = rgi((zg, yg, xg), rho)

    # N_linepts = 4*np.max(Ns)
    # N_loop = 25  # number of times along an axis?
    # lvec = np.linspace(0, N_loop, N_linepts).reshape(N_linepts, 1)    
    N_y = Ns[1]
    total_length = flyby_a*N_y**2 if N_y <= 256 else flyby_a*N_y
    dl = yg[1] - yg[0]
    N_dl = int(total_length / dl)
    lvec = np.linspace(-total_length/2, total_length/2, N_dl).reshape(N_dl, 1)

    if do_rand_start:
        # start at random point in box
        l_start = uniform(high=Ls)
    else:
        assert l_start is not None, 'l_start must be a valid numpy array!'
    # direction biased in x direction (z, y, x)
    l_dir = np.array([np.pi/8, np.sqrt(0.5), 1.])
    l_dir /= np.sqrt(np.sum(l_dir**2))
    pts = np.mod(l_start + lvec*l_dir, Ls)

    # Interpolate data along line running through box
    FB = {}
    FB['Bx'], FB['By'], FB['Bz'] = Bx_i(pts), By_i(pts), Bz_i(pts)
    FB['Bmag'], FB['rho'] = Bmag_i(pts), rho_i(pts)
    FB['start_point'], FB['direction'] = l_start, l_dir
    FB['l_param'], FB['points'] = lvec, pts
    FB['a'], FB['snapshot_number'] = flyby_a, flyby_n

    return FB