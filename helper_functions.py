import os
from shutil import copy

import h5py
import numpy as np
import numpy.fft as fft
from scipy.interpolate import RegularGridInterpolator as rgi

import diagnostics as diag

if diag.COMPUTER == 'local':
    from_array_path = '/home/zade/masters_2021/templates_athinput/athinput.from_array'
    from_array_reinterp_path = '/home/zade/masters_2021/templates_athinput/athinput.from_array_reinterp'
    athena_path = '/home/zade/masters_2021/athena/bin/from_array/athena'
else:
    from_array_path = '/home/johza721/masters_2021/templates/athinput.from_array'
    athena_path = '/home/johza721/masters_2021/athena/bin/athena_mauivis/athena'

#--- REINTERPOLATION FUNCTIONS ---#

def pad_array(x):
    # Pad data arrays at edges in order to make 'periodic'
    # Helps ensure that the interpolation returns a valid result at box edges.
    return np.pad(x, (1, 1), 'wrap')

def pad_grid(Xg):
    # Extending the grid one grid point before 0
    # and one grid point after Ls
    # Helps with interpolation of a periodic box
    if Xg.size == 1:
        l_point = [Xg[0]-1]
        r_point = [Xg[0]+1]
    else:
        l_point = [2*Xg[0] - Xg[1]]
        r_point = [2*Xg[-1] - Xg[-2]]
    return np.concatenate((l_point, Xg, r_point))

def generate_grid_reinterp(Ns, Ls, return_mesh=0, return_edges=0, pad=1):
    Xs = []
    for i in range(3):
        # z = 0, y = 1, x = 2
        Xe = np.linspace(0, Ls[i], Ns[i]+1)
        if return_edges:
            if pad:
                Xs.append(pad_grid(Xe))
            else:
                Xs.append(Xe)
        else:
            # Get cell-centered coordinates and extend
            Xg = 0.5*(Xe[1:] + Xe[:-1])
            if pad:
                Xs.append(pad_grid(Xg))
            else:
                Xs.append(Xg)
    
    if return_mesh:
        return np.meshgrid(*Xs, indexing='ij')
    else:
        return Xs

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


# --- GENERATING ICS FUNCTIONS --- #

# edit_athinput - edits the corresponding athinput file with quantites input below
# ONLY WORKS FOR athinput.from_array layout
def edit_athinput(athinput, save_folder, n_X, X_min, X_max, meshblock, h5name,
                  time_lim, dt, iso_sound_speed, expand, exp_rate,
                  reinterpolate=0, exp_init=1, start_time=1., n_hst=0, n_hdf5=0):
    ath_path = save_folder + athinput.split('/')[-1] + '_' + h5name.split('/')[-1].split('.')[0]
    copy(athinput, ath_path)
    ath = open(ath_path, 'r')
    list_of_lines = ath.readlines()

    if reinterpolate:
        # time limit
        list_of_lines[10] = 'tlim       = ' + str(time_lim) + '  # time limit\n'
        list_of_lines[11] = 'start_time = ' + str(start_time) + '  # start time\n'
        # hst
        list_of_lines[19] = 'file_number = ' + str(n_hst) + '\n'
        # hdf5
        list_of_lines[24] = 'dt        = ' + str(dt) + '   # time increment between outputs\n'
        list_of_lines[25] = 'file_number = ' + str(n_hdf5) + '\n'
        # X1
        list_of_lines[28] = 'nx1     = ' + str(n_X[0]) + '        # number of zones in x1-direction\n'
        list_of_lines[29] = 'x1min   = ' + str(X_min[0]) + '     # minimum value of x1\n'
        list_of_lines[30] = 'x1max   = ' + str(X_max[0]) + '     # maximum value of x1\n'
        # X2
        list_of_lines[34] = 'nx2     = ' + str(n_X[1]) + '        # number of zones in x2-direction\n'
        list_of_lines[35] = 'x2min   = ' + str(X_min[1]) + '     # minimum value of x2\n'
        list_of_lines[36] = 'x2max   = ' + str(X_max[1]) + '     # maximum value of x2\n'
        # X3
        list_of_lines[40] = 'nx3     = ' + str(n_X[2]) + '        # number of zones in x3-direction\n'
        list_of_lines[41] = 'x3min   = ' + str(X_min[2]) + '     # minimum value of x3\n'
        list_of_lines[42] = 'x3max   = ' + str(X_max[2]) + '     # maximum value of x3\n'
        # Meshblocks
        list_of_lines[50] = 'nx1 = ' + str(meshblock[0]) + '  # block size in x1-direction\n'
        list_of_lines[51] = 'nx2 = ' + str(meshblock[1]) + '  # block size in x2-direction\n'
        list_of_lines[52] = 'nx3 = ' + str(meshblock[2]) + '  # block size in x3-direction\n'
        # sound speed
        list_of_lines[56] = 'iso_sound_speed = ' + str(iso_sound_speed) + '  # isothermal sound speed (for barotropic EOS)\n'
        # hdf5 file name
        list_of_lines[59] = 'input_filename = ' + h5name + '  # name of HDF5 file containing initial conditions\n'
        # expansion
        expanding = 'true' if expand else 'false'
        list_of_lines[69] = 'expanding = ' + expanding + '\n'
        list_of_lines[70] = 'expand_rate = ' + str(exp_rate) + '\n'
        # list_of_lines[71] = 'expand_init = ' + str(exp_init) + '\n'
    else:
        # time limit
        list_of_lines[10] = 'tlim       = ' + str(time_lim) + ' # time limit\n'
        list_of_lines[22] = 'dt        = ' + str(dt) + '   # time increment between outputs\n'
        # X1
        list_of_lines[25] = 'nx1     = ' + str(n_X[0]) + '        # number of zones in x1-direction\n'
        list_of_lines[26] = 'x1min   = ' + str(X_min[0]) + '     # minimum value of x1\n'
        list_of_lines[27] = 'x1max   = ' + str(X_max[0]) + '     # maximum value of x1\n'
        # X2
        list_of_lines[31] = 'nx2     = ' + str(n_X[1]) + '        # number of zones in x2-direction\n'
        list_of_lines[32] = 'x2min   = ' + str(X_min[1]) + '     # minimum value of x2\n'
        list_of_lines[33] = 'x2max   = ' + str(X_max[1]) + '     # maximum value of x2\n'
        # X3
        list_of_lines[37] = 'nx3     = ' + str(n_X[2]) + '        # number of zones in x3-direction\n'
        list_of_lines[38] = 'x3min   = ' + str(X_min[2]) + '     # minimum value of x3\n'
        list_of_lines[39] = 'x3max   = ' + str(X_max[2]) + '     # maximum value of x3\n'
        # Meshblocks
        list_of_lines[47] = 'nx1 = ' + str(meshblock[0]) + '  # block size in x1-direction\n'
        list_of_lines[48] = 'nx2 = ' + str(meshblock[1]) + '  # block size in x2-direction\n'
        list_of_lines[49] = 'nx3 = ' + str(meshblock[2]) + '  # block size in x3-direction\n'
        # sound speed
        list_of_lines[53] = 'iso_sound_speed = ' + str(iso_sound_speed) + '  # isothermal sound speed (for barotropic EOS)\n'
        # hdf5 file name
        list_of_lines[56] = 'input_filename = ' + h5name + '  # name of HDF5 file containing initial conditions\n'
        # expansion
        expanding = 'true' if expand else 'false'
        list_of_lines[66] = 'expanding = ' + expanding + '\n'
        list_of_lines[67] = 'expand_rate = ' + str(exp_rate) + '\n'

    ath = open(ath_path, 'w')
    ath.writelines(list_of_lines)
    ath.close()
    return ath_path

def read_athinput(athinput, reinterpolate=0):

    def get_from_string(i, dtype):
        s = list_of_lines[i].split('=')[1].split('#')[0]
        if dtype == 'int':
            return int(s)
        elif dtype == 'float':
            return float(s)

    def get_value_indices():
        nx = [i for i, j in enumerate(list_of_lines) if ('nx1' in j or 'nx2' in j or 'nx3' in j)]
        xmin = [i for i, j in enumerate(list_of_lines) if ('x1min' in j or 'x2min' in j or 'x3min' in j)]
        xmax = [i for i, j in enumerate(list_of_lines) if ('x1max' in j or 'x2max' in j or 'x3max' in j)]
        return nx[:3], xmin, xmax, nx[3:]

    ath = open(athinput, 'r')
    list_of_lines = ath.readlines()
    nx_idx, xmin_idx, xmax_idx, mesh_idx = get_value_indices()
    
    n_X = np.array([get_from_string(i, 'int') for i in nx_idx])
    X_min = np.array([get_from_string(i, 'float') for i in xmin_idx])
    X_max = np.array([get_from_string(i, 'float') for i in xmax_idx])
    meshblock = np.array([get_from_string(i, 'int') for i in mesh_idx])

    if reinterpolate:
        dt_hst = float(list_of_lines[17].split('=')[1].split('#')[0])

        return n_X, X_min, X_max, meshblock, dt_hst
    else:
        return n_X, X_min, X_max, meshblock 

def generate_grid(X_min, X_max, n_X):
    # cell-edge grid
    xe = np.linspace(X_min[0], X_max[0], n_X[0]+1)
    ye = np.linspace(X_min[1], X_max[1], n_X[1]+1)
    ze = np.linspace(X_min[2], X_max[2], n_X[2]+1)
    # cell-centered grid
    xg = 0.5*(xe[:-1] + xe[1:])
    yg = 0.5*(ye[:-1] + ye[1:])
    zg = 0.5*(ze[:-1] + ze[1:])
    
    # grid spacings
    dx = xg[1] - xg[0]
    dy = np.inf if n_X[1] == 1 else yg[1] - yg[0]
    dz = np.inf if n_X[2] == 1 else zg[1] - zg[0]
    Xgrid, dX = (xg, yg, zg), (dx, dy, dz)
    return Xgrid, dX

def generate_mesh_structure(folder, athinput):
    cdir = os.getcwd()
    os.chdir(folder)
    os.system(athena_path + ' -i ' + athinput + ' -m 1 > /dev/null')  # > /dev/null supresses output, remove if need to see meshblock details
    blocks = read_mesh_structure('mesh_structure.dat')
    os.remove('mesh_structure.dat')
    n_blocks = blocks.shape[1]
    os.chdir(cdir)
    return n_blocks, blocks

# read meshblock - gets structure from mesh_structure.dat
def read_mesh_structure(data_fname):
    blocks = []
    with open(data_fname) as f:
        s = f.readline()
        while len(s) > 0:
            s = f.readline()
            # Looking for 'location = (%d %d %d)' and obtaining numbers in brackets
            if 'location =' in s:
                loc = s.split('=')[1].split('(')[1].split(')')[0].replace(' ', '')
                temp = [int(c) for c in loc]
                blocks.append(temp)
    return np.array(blocks).T

# meshblock check - checks that athinput file and meshblock input match
def check_mesh_structure(blocks, n_X, meshblock):
    n_blocks = n_X / meshblock
    if n_blocks.prod() != blocks.shape[1]:
        raise AssertionError('Number of meshblocks doesnt match: must have input wrong in athinput or script')
    if np.any(blocks.max(axis=1) + 1 != n_blocks):
        raise AssertionError('Meshblock structure doesnt match: must have input wrong in athinput or script')


def make_meshblocks(folder, athinput, n_X, meshblock, one_D):
    if one_D:
        n_blocks = n_X[0] / meshblock[0]
        blocks = np.array([np.arange(n_blocks), np.zeros(n_blocks), np.zeros(n_blocks)])
    else:
        n_blocks, blocks = generate_mesh_structure(folder, athinput)
    check_mesh_structure(blocks, n_X, meshblock)
    return n_blocks, blocks

# shift and extend A - moves A from cell-centre to cell-faces
# and makes it periodic allowing for numerical derivatives to be computed at
# the boundary
def shift_and_extend_A(Ax, Ay, Az):
    Ax = reshape_helper(Ax, 0)
    Ax = 0.5*(Ax[:, :-1, :] + Ax[:, 1:, :])
    Ax = 0.5*(Ax[:-1, :, :] + Ax[1:, :, :])

    Ay = reshape_helper(Ay, 1)
    Ay = 0.5*(Ay[:-1, :, :] + Ay[1:, :, :])
    Ay = 0.5*(Ay[:, :, :-1] + Ay[:, :, 1:])

    Az = reshape_helper(Az, 2)
    Az = 0.5*(Az[:, :, :-1] + Az[:, :, 1:])
    Az = 0.5*(Az[:, :-1, :] + Az[:, 1:, :])
    return Ax, Ay, Az

# reshape helper - concatenates A differently along different axes to allow
# for periodicity
# Based off of Jono's MATLAB script
def reshape_helper(A, component):
    # component = 0 ⟺ x, 1 ⟺ y, 2 ⟺ z
    # pad notation: first tuple pad on (axis 0, axis 1, axis 2)
    # second tuple: (x, y) add last x entries to front of axis 
    # and first y entries to end of array
    # e.g. x = [0, 1, 2], pad(x, (1, 1), 'wrap') = [2, 0, 1, 2, 0]
    if component != 0:
        pad = ((1, 1), (0, 1), (1, 1)) if component == 1 else ((0, 1), (1, 1), (1, 1))
    else:
        pad = ((1, 1), (1, 1), (0, 1))
    return np.pad(A, pad, 'wrap')

def remove_prev_h5file(h5name):
    if os.path.isfile(h5name):  # 'overwrite' old ICs
        os.remove(h5name)

def setup_hydro_grid(n_X, X_grid, N_HYDRO, Dnf, UXf, UYf, UZf, BXf, BYf, BZf):
    # Athena orders cooridinates (Z, Y, X) while n_X is in the form (X, Y, Z)
    # It's easier to start from this ordering instead of having to do array
    # manipulations at the end.
    Hy_grid = np.zeros(shape=(N_HYDRO, *n_X[::-1]))

    # --- GRID CREATION --- #

    xg, yg, zg = X_grid
    Zg, Yg, Xg = np.meshgrid(zg, yg, xg, indexing='ij')

    # Place quantites on grid
    Hy_grid[0] = Dnf(Xg, Yg, Zg)
    Hy_grid[1] = Hy_grid[0] * UXf(Xg, Yg, Zg)  # using momentum for conserved values
    Hy_grid[2] = Hy_grid[0] * UYf(Xg, Yg, Zg)
    Hy_grid[3] = Hy_grid[0] * UZf(Xg, Yg, Zg)
    BXcc = BXf(Xg, Yg, Zg)
    BYcc = BYf(Xg, Yg, Zg)
    BZcc = BZf(Xg, Yg, Zg)

    # ignoring NHYDRO > 4 for now
    # if NHYDRO == 5: etc for adiabatic or CGL

    Xg, Yg, Zg = None, None, None  # I think this clears memory?
    return Hy_grid, BXcc, BYcc, BZcc

def save_hydro_grid(h5name, Hy_grid, N_HYDRO, n_blocks, blocks, meshblock, remove_h5=1):
    Hy_h5 = np.zeros(shape=(N_HYDRO, n_blocks, *meshblock[::-1]))
    for h in range(N_HYDRO):
        for m in range(n_blocks):  # save to each meshblock individually
            off = blocks[:, m]
            ind_s = (meshblock*off)[::-1]
            ind_e = (meshblock*off + meshblock)[::-1]
            Hy_h5[h, m, :, :, :] = Hy_grid[h, ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]]

    Hy_grid = None
    with h5py.File(h5name, 'a') as f:
        f['cons'] = Hy_h5
    Hy_h5 = None

def calc_and_save_B(BXcc, BYcc, BZcc, h5name, n_X, X_min, X_max, meshblock, n_blocks, blocks, dx, dy, dz):
    # Get mean of B-field (inverting and redoing curl takes this away)
    B_mean = np.array([BXcc.mean(), BYcc.mean(), BZcc.mean()])

    # Calculate A from B using Fourier space by inverting the curl
    K = {}
    for k in range(3):
        if n_X[k] > 1:
            K[k] = 2j*np.pi/(X_max[k] - X_min[k])*diag.ft_array(n_X[k])
        else:
            K[k] = np.array(0j)


    K_z, K_y, K_x = np.meshgrid(K[2], K[1], K[0], indexing='ij')
    K_2 = abs(K_x)**2 + abs(K_y)**2 + abs(K_z)**2
    K_2[0, 0, 0] = 1
    K_x /= K_2
    K_y /= K_2
    K_z /= K_2

    ftBX = fft.fftn(BXcc)
    ftBY = fft.fftn(BYcc)
    ftBZ = fft.fftn(BZcc)

    BXcc, BYcc, BZcc = None, None, None

    A_x = np.real(fft.ifftn(K_y*ftBZ - K_z*ftBY))
    A_y = np.real(fft.ifftn(K_z*ftBX - K_x*ftBZ))
    A_z = np.real(fft.ifftn(K_x*ftBY - K_y*ftBX))

    # A is cell-centred; shift to edges and make periodic
    A_x, A_y, A_z = shift_and_extend_A(A_x, A_y, A_z) 

    # Calculate B from A using finite-difference curl (this is how Athena++ calculates B from A)
    # Copied from Jono's MATLAB script
    # Bx
    B_mesh = meshblock + [1, 0, 0]
    B_h5 = np.zeros(shape=(n_blocks, *B_mesh[::-1]))
    for m in range(n_blocks):
        off = blocks[:, m]
        ind_s = (meshblock*off)[::-1]
        ind_e = (meshblock*off + meshblock)[::-1]
        B_h5[m, :, :, :] = B_mean[0] + \
                        ((A_z[ind_s[0]:ind_e[0], ind_s[1]+1:ind_e[1]+1, ind_s[2]:ind_e[2]+1] - \
                            A_z[ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]+1]) / dy - \
                        (A_y[ind_s[0]+1:ind_e[0]+1, ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]+1] - \
                            A_y[ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]+1]) / dz)
    with h5py.File(h5name, 'a') as f:
        f['bf1'] = B_h5

    # By
    B_mesh = meshblock + [0, 1, 0]
    B_h5 = np.zeros(shape=(n_blocks, *B_mesh[::-1]))
    for m in range(n_blocks):
        off = blocks[:, m]
        ind_s = (meshblock*off)[::-1]
        ind_e = (meshblock*off + meshblock)[::-1]
        B_h5[m, :, :, :] = B_mean[1] + \
                        ((A_x[ind_s[0]+1:ind_e[0]+1, ind_s[1]:ind_e[1]+1, ind_s[2]:ind_e[2]] - \
                            A_x[ind_s[0]:ind_e[0], ind_s[1]:ind_e[1]+1, ind_s[2]:ind_e[2]]) / dz - \
                        (A_z[ind_s[0]:ind_e[0], ind_s[1]:ind_e[1]+1, ind_s[2]+1:ind_e[2]+1] - \
                            A_z[ind_s[0]:ind_e[0], ind_s[1]:ind_e[1]+1, ind_s[2]:ind_e[2]]) / dx)
    with h5py.File(h5name, 'a') as f:
        f['bf2'] = B_h5

    # Bz
    B_mesh = meshblock + [0, 0, 1]
    B_h5 = np.zeros(shape=(n_blocks, *B_mesh[::-1]))
    for m in range(n_blocks):
        off = blocks[:, m]
        ind_s = (meshblock*off)[::-1]
        ind_e = (meshblock*off + meshblock)[::-1]
        B_h5[m, :, :, :] = B_mean[2] + \
                        ((A_y[ind_s[0]:ind_e[0]+1, ind_s[1]:ind_e[1], ind_s[2]+1:ind_e[2]+1] - \
                            A_y[ind_s[0]:ind_e[0]+1, ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]]) / dx - \
                        (A_x[ind_s[0]:ind_e[0]+1, ind_s[1]+1:ind_e[1]+1, ind_s[2]:ind_e[2]] - \
                            A_x[ind_s[0]:ind_e[0]+1, ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]]) / dy)
    with h5py.File(h5name, 'a') as f:
        f['bf3'] = B_h5


def constB2_faceinterp(BXcc, BYcc, BZcc, h5name, n_X, X_min, X_max, meshblock, n_blocks, blocks):
    Ns, Ls = n_X[::-1], (X_max - X_min)[::-1]
    ze, ye, xe = generate_grid_reinterp(Ns, Ls, return_edges=1, pad=0) 
    zg, yg, xg = generate_grid_reinterp(Ns, Ls, pad=0) 

    BX_interp = rgi((pad_grid(zg), pad_grid(yg), pad_grid(xg)), pad_array(BXcc))
    BY_interp = rgi((pad_grid(zg), pad_grid(yg), pad_grid(xg)), pad_array(BYcc))
    BZ_interp = rgi((pad_grid(zg), pad_grid(yg), pad_grid(xg)), pad_array(BZcc))
    interps = [BX_interp, BY_interp, BZ_interp]

    BX_grid = np.meshgrid(zg, yg, xe, indexing='ij')
    BY_grid = np.meshgrid(zg, ye, xg, indexing='ij')
    BZ_grid = np.meshgrid(ze, yg, xg, indexing='ij')
    B_grids = [BX_grid, BY_grid, BZ_grid]

    B_faces = []

    for idx, B_grid in enumerate(B_grids):
        faces_Ns = Ns + np.roll([0, 0, 1], -idx)
        B_grid_z, B_grid_y, B_grid_x = B_grid
        pts = np.array([B_grid_z.ravel(), B_grid_y.ravel(), B_grid_x.ravel()]).T
        B_faces.append(interps[idx](pts).reshape(*faces_Ns))
    
    # Bx
    B_mesh = meshblock + [1, 0, 0]
    B_h5 = np.zeros(shape=(n_blocks, *B_mesh[::-1]))
    for m in range(n_blocks):
        off = blocks[:, m]
        ind_s = (meshblock*off)[::-1]
        ind_e = (meshblock*off + B_mesh)[::-1]
        B_h5[m, :, :, :] = B_faces[0][ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]]
    with h5py.File(h5name, 'a') as f:
        f['bf1'] = B_h5

    # By
    B_mesh = meshblock + [0, 1, 0]
    B_h5 = np.zeros(shape=(n_blocks, *B_mesh[::-1]))
    for m in range(n_blocks):
        off = blocks[:, m]
        ind_s = (meshblock*off)[::-1]
        ind_e = (meshblock*off + B_mesh)[::-1]
        B_h5[m, :, :, :] = B_faces[1][ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]]
    with h5py.File(h5name, 'a') as f:
        f['bf2'] = B_h5

    # Bz
    B_mesh = meshblock + [0, 0, 1]
    B_h5 = np.zeros(shape=(n_blocks, *B_mesh[::-1]))
    for m in range(n_blocks):
        off = blocks[:, m]
        ind_s = (meshblock*off)[::-1]
        ind_e = (meshblock*off + B_mesh)[::-1]
        B_h5[m, :, :, :] = B_faces[2][ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]]
    with h5py.File(h5name, 'a') as f:
        f['bf3'] = B_h5
