# --- IMPORTS --- #
import os
from shutil import copy

import h5py
import numpy as np
import numpy.fft as fft

import diagnostics as diag
import generate_spectrum as genspec

# --- SUPPORTING FUNCTIONS --- #

# edit_athinput - edits the corresponding athinput file with quantites 
# input below
def edit_athinput(athinput, save_folder, n_X, X_min, X_max, meshblock, h5name):
    ath_path = save_folder + athinput.split('/')[-1] + '_' + h5name.split('/')[-1].split('.')[0]
    copy(athinput, ath_path)
    ath = open(ath_path, 'r')
    list_of_lines = ath.readlines()

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
    # hdf5 file name
    list_of_lines[56] = 'input_filename = ' + h5name + '  # name of HDF5 file containing initial conditions\n'

    ath = open(ath_path, 'w')
    ath.writelines(list_of_lines)
    ath.close()
    return ath_path

def read_athinput(athinput):

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

    return n_X, X_min, X_max, meshblock 

def generate_grid(X_min, X_max, n_X):
    xe = np.linspace(X_min[0], X_max[0], n_X[0]+1)
    ye = np.linspace(X_min[1], X_max[1], n_X[1]+1)
    ze = np.linspace(X_min[2], X_max[2], n_X[2]+1)
    xg = 0.5*(xe[:-1] + xe[1:])
    yg = 0.5*(ye[:-1] + ye[1:])
    zg = 0.5*(ze[:-1] + ze[1:])
    
    dx = xg[1] - xg[0]
    dy = np.inf if n_X[1] == 1 else yg[1] - yg[0]
    dz = np.inf if n_X[2] == 1 else zg[1] - zg[0]
    Xgrid, dX = (xg, yg, zg), (dx, dy, dz)
    return Xgrid, dX

def generate_mesh_structure(folder, athinput):
    cdir = os.getcwd()
    os.chdir(folder)
    os.system('./athena -i ' + athinput + ' -m 1 > /dev/null')  # > /dev/null supresses output, remove if need to see meshblock details
    blocks = read_mesh_structure('mesh_structure.dat')
    os.remove('mesh_structure.dat')
    n_blocks = blocks.shape[1]
    os.chdir(cdir)
    return n_blocks, blocks

# read meshblock - gets structure from mesh_structure.dat
def read_mesh_structure(dat_fname):
    blocks = []
    with open(dat_fname) as f:
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
def reshape_helper(A, component):
    # uncomment this if it does not work
    # appears to work, I'll keep this here out of superstition
    # if component == 0:  # x-component
    #     A = np.concatenate((A, A[:, :, 0].reshape((*A.shape[:2], 1))), axis=2)
    #     A = np.concatenate((A[:, -1, :].reshape((A.shape[0], 1, A.shape[2])), A, A[:, 0, :].reshape((A.shape[0], 1, A.shape[2]))), axis=1)
    #     A = np.concatenate((A[-1, :, :].reshape((1, *A.shape[1:])), A, A[0, :, :].reshape((1, *A.shape[1:]))), axis=0)
    # elif component == 1:  # y-component
    #     A = np.concatenate((A[:, :, -1].reshape((*A.shape[:2], 1)), A, A[:, :, 0].reshape((*A.shape[:2], 1))), axis=2)
    #     A = np.concatenate((A, A[:, 0, :].reshape((A.shape[0], 1, A.shape[2]))), axis=1)
    #     A = np.concatenate((A[-1, :, :].reshape((1, *A.shape[1:])), A, A[0, :, :].reshape((1, *A.shape[1:]))), axis=0)
    # elif component == 2:  # z-component
    #     A = np.concatenate((A[:, :, -1].reshape((*A.shape[:2], 1)), A, A[:, :, 0].reshape((*A.shape[:2], 1))), axis=2)
    #     A = np.concatenate((A[:, -1, :].reshape((A.shape[0], 1, A.shape[2])), A, A[:, 0, :].reshape((A.shape[0], 1, A.shape[2]))), axis=1)
    #     A = np.concatenate((A, A[0, :, :].reshape((1, *A.shape[1:]))), axis=0)
    # return A
    if component != 0:
        pad = ((1, 1), (0, 1), (1, 1)) if component == 1 else ((0, 1), (1, 1), (1, 1))
    else:
        pad = ((1, 1), (1, 1), (0, 1))
    return np.pad(A, pad, 'wrap')

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
    # Need to transpose x and y for it to work for some reason
    # K_x, K_y, K_z = K_x.transpose((0, 2, 1)), K_y.transpose((0, 2, 1)), K_z.transpose((0, 2, 1))
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

    # Test
    # K_x *= K_2; K_y *= K_2; K_z *= K_2
    # ftBX = fft.fftn(A_x)
    # ftBY = fft.fftn(A_y)
    # ftBZ = fft.fftn(A_z)
    # B_x = np.real(fft.ifftn(K_y*ftBZ - K_z*ftBY))
    # B_y = np.real(fft.ifftn(K_z*ftBX - K_x*ftBZ))
    # B_z = np.real(fft.ifftn(K_x*ftBY - K_y*ftBX))

    # A is cell-centred; shift to edges and make periodic
    A_x, A_y, A_z = shift_and_extend_A(A_x, A_y, A_z) 

    # Calculate B from A using finite-difference curl (MATLAB script is how Athena does it)
    # Bx
    B_mesh = meshblock + [1, 0, 0]
    B_h5 = np.zeros(shape=(n_blocks, *B_mesh[::-1]))
    for m in range(n_blocks):
        off = blocks[:, m]
        ind_s = (meshblock*off)[::-1]
        ind_e = (meshblock*off + meshblock)[::-1]
        B_h5[m, :, :, :] = B_mean[0] + \
                        (A_z[ind_s[0]:ind_e[0], ind_s[1]+1:ind_e[1]+1, ind_s[2]:ind_e[2]+1] - \
                            A_z[ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]+1]) / dy - \
                        (A_y[ind_s[0]+1:ind_e[0]+1, ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]+1] - \
                            A_y[ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]+1]) / dz
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
                        (A_x[ind_s[0]+1:ind_e[0]+1, ind_s[1]:ind_e[1]+1, ind_s[2]:ind_e[2]] - \
                            A_x[ind_s[0]:ind_e[0], ind_s[1]:ind_e[1]+1, ind_s[2]:ind_e[2]]) / dz - \
                        (A_z[ind_s[0]:ind_e[0], ind_s[1]:ind_e[1]+1, ind_s[2]+1:ind_e[2]+1] - \
                            A_z[ind_s[0]:ind_e[0], ind_s[1]:ind_e[1]+1, ind_s[2]:ind_e[2]]) / dx
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
                        (A_y[ind_s[0]:ind_e[0]+1, ind_s[1]:ind_e[1], ind_s[2]+1:ind_e[2]+1] - \
                            A_y[ind_s[0]:ind_e[0]+1, ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]]) / dx - \
                        (A_x[ind_s[0]:ind_e[0]+1, ind_s[1]+1:ind_e[1]+1, ind_s[2]:ind_e[2]] - \
                            A_x[ind_s[0]:ind_e[0]+1, ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]]) / dy
    with h5py.File(h5name, 'a') as f:
        f['bf3'] = B_h5

# --- GENERATING FUNCTIONS --- #

def create_athena_fromics(folder, h5name, n_X, X_min, X_max, meshblock, athinput='/home/zade/masters_2021/templates_athinput/athinput.from_array'):
    '''Function to generate an h5 file containing user set initial conditions 
       for the Athena++ from_array problem generator to start from. Generates a grid the same size as specified in the
       athinput file and calculates the quantities at each grid point, then writes to h5 file.

    Arguments:
        folder {string} -- path to base folder that contains the athena binary, athinput file, and where h5 file is to be output
        athinput {string} -- name of athinput file, of the form athinput.<***>
        h5name {string} -- name of h5 file for data to be output to
        n_X {numpy array} -- array containing the number of grid points in the 3 coordinate directions x1,x2,x3
        X_min {numpy array} -- array containing the minimum x_n coordinate in each direction
        X_max {numpy array} -- array containing the maximum x_n coordinate in each direction
        meshblock {numpy array} -- array containing the meshblock dimensions in each direction
    '''
    # --- INPUTS --- #

    h5name = folder + h5name  # eg 'ICs_template.h5'

    ath_copy = edit_athinput(athinput, folder, n_X, X_min, X_max, meshblock, h5name)
    N_HYDRO = 4  # number of hydro variables (e.g. density and momentum); assuming isothermal here
    
    # Dimension setting: 1D if only x has more than one gridpoint
    one_D = 1 if np.all(n_X[1:] == 1) else 0
    
    # Athena orders cooridinates (Z, Y, X) while n_X is in the form (X, Y, Z)
    # It's easier to start from this ordering instead of having to do array
    # manipulations at the end.
    Hy_grid = np.zeros(shape=(N_HYDRO, *n_X[::-1]))

    # User set initial conditions
    # Density
    Dnf = lambda X, Y, Z: np.ones(X.shape)
    # Velocity components
    UXf = lambda X, Y, Z: np.zeros(X.shape)
    UYf = lambda X, Y, Z: np.zeros(X.shape)
    UZf = lambda X, Y, Z: -0.01*np.sin((2*np.pi/X_max[0]) * X + (2*np.pi/X_max[1]) * Y)
    # Magnetic components
    BXf = lambda X, Y, Z: np.ones(X.shape)
    BYf = lambda X, Y, Z: np.zeros(X.shape)
    BZf = lambda X, Y, Z: -0.01*np.sin((2*np.pi/X_max[0]) * X + (2*np.pi/X_max[1]) * Y)

    # --- GRID CREATION --- #

    (xg, yg, zg), (dx, dy, dz) = generate_grid(X_min, X_max, n_X)

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

    # --- MESHBLOCK STRUCTURE --- #

    if one_D:
        n_blocks = n_X[0] / meshblock[0]
        blocks = np.array([np.arange(n_blocks), np.zeros(n_blocks), np.zeros(n_blocks)])
    else:
        n_blocks, blocks = generate_mesh_structure(folder, ath_copy)
    check_mesh_structure(blocks, n_X, meshblock)

    # --- SAVING VARIABLES --- #

    # - HYDRO
    Hy_h5 = np.zeros(shape=(N_HYDRO, n_blocks, *meshblock[::-1]))
    for h in range(N_HYDRO):
        for m in range(n_blocks):  # save to each meshblock individually
            off = blocks[:, m]
            ind_s = (meshblock*off)[::-1]
            ind_e = (meshblock*off + meshblock)[::-1]
            Hy_h5[h, m, :, :, :] = Hy_grid[h, ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]]

    Hy_grid = None
    if os.path.isfile(h5name):  # 'overwrite' old ICs
        os.remove(h5name)
    with h5py.File(h5name, 'a') as f:
        f['cons'] = Hy_h5
    Hy_h5 = None
    print('Hydro Saved Succesfully')

    # - MAGNETIC
    calc_and_save_B(BXcc, BYcc, BZcc, h5name, n_X, X_min, X_max, meshblock, n_blocks, blocks, dx, dy, dz)
    print('Magnetic Saved Successfully')
    print('Done!')

def create_athena_fromh5(save_folder, athinput_in_folder, athinput_in, h5name, athdf_input,
                         athinput_out='/home/zade/masters_2021/templates_athinput/athinput.from_array'):
    '''[summary]

    Parameters
    ----------
    save_folder : string
        Base folder where h5 file is to be saved.\n
    athinput_in_folder : string
        Base folder containing original athinput file that generated athdf.\n
    athinput_in : string
        Name of athinput file that generated initial athdf file.\n
    h5name : string
        Name of h5 file to be saved.\n
    athdf_input : string
        Path to initial athdf to be read.\n
    athinput_out : string
        Path to from_array athinput to be copied and editied.
    '''
    
    h5name = save_folder + h5name
    h5name += '.h5' if '.h5' not in h5name else ''
    if os.path.isfile(h5name):  # 'overwrite' old ICs
        os.remove(h5name)
    if athinput_in_folder not in athinput_in:
        athinput_in = athinput_in_folder + athinput_in

    f_athdf = h5py.File(athdf_input, 'r')

    if 'cons' in list(f_athdf.keys()):  # conserved variables if 1, primitive if 0
        with h5py.File(h5name, 'a') as f:
            f['cons'] = np.array(f_athdf['cons'])  # no need to unwrap meshblocks
    else:
        hydro_prim = np.copy(f_athdf['prim'])  # don't want to modify the original data set
        for i in range(3):
            hydro_prim[i] *= hydro_prim[0]  # getting momentum variables (vel * density)
        with h5py.File(h5name, 'a') as f:
            f['cons'] = hydro_prim
        hydro_prim = None
    
    n_X, X_min, X_max, meshblock = read_athinput(athinput_in)
    athinput_out = edit_athinput(athinput_out, save_folder, n_X, X_min, X_max, meshblock, h5name)

    dx, dy, dz = generate_grid(X_min, X_max, n_X)[1]
    
    n_blocks, blocks = generate_mesh_structure(athinput_in_folder, athinput_in)
    B_unpacked = np.zeros(shape=(3, *n_X[::-1]))
    for b in range(3):
        for m in range(n_blocks):  # save to each meshblock individually
                off = blocks[:, m]
                ind_s = (meshblock*off)[::-1]
                ind_e = (meshblock*off + meshblock)[::-1]
                B_unpacked[b, ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]] = f_athdf['B'][b, m, :, :, :]
    BXcc, BYcc, BZcc = B_unpacked
    calc_and_save_B(BXcc, BYcc, BZcc, h5name, n_X, X_min, X_max, meshblock, n_blocks, blocks, dx, dy, dz)

def create_athena_alfvenspec(folder, h5name, n_X, X_min, X_max, meshblock,
                             athinput='/home/zade/masters_2021/templates_athinput/athinput.from_array',
                             spectrum=-5/3,
                             do_mode_test=0):
    h5name = folder + h5name  # eg 'ICs_template.h5'

    ath_copy = edit_athinput(athinput, folder, n_X, X_min, X_max, meshblock, h5name)
    N_HYDRO = 4  # number of hydro variables (e.g. density and momentum); assuming isothermal here
    
    # Dimension setting: 1D if only x has more than one gridpoint
    one_D = 1 if np.all(n_X[1:] == 1) else 0
    
    # Athena orders cooridinates (Z, Y, X) while n_X is in the form (X, Y, Z)
    # It's easier to start from this ordering instead of having to do array
    # manipulations at the end.
    Hy_grid = np.zeros(shape=(N_HYDRO, *n_X[::-1]))

    # Generate mean fields
    # Density
    Dnf = lambda X, Y, Z: np.ones(X.shape)
    UXf = lambda X, Y, Z: np.zeros(X.shape)
    UYf = lambda X, Y, Z: np.zeros(X.shape)
    UZf = lambda X, Y, Z: np.zeros(X.shape)
    BXf = lambda X, Y, Z: np.ones(X.shape)
    BYf = lambda X, Y, Z: np.zeros(X.shape)
    BZf = lambda X, Y, Z: np.zeros(X.shape)

    (xg, yg, zg), (dx, dy, dz) = generate_grid(X_min, X_max, n_X)
    Zg, Yg, Xg = np.meshgrid(zg, yg, xg, indexing='ij')
    rho = Dnf(Xg, Yg, Zg)
    MX, MY, MZ = rho * UXf(Xg, Yg, Zg), rho * UYf(Xg, Yg, Zg), rho * UZf(Xg, Yg, Zg)
    BXcc, BYcc, BZcc = BXf(Xg, Yg, Zg), BYf(Xg, Yg, Zg), BZf(Xg, Yg, Zg)
    B_0 = np.array([BXcc, BYcc, BZcc])
    Xg, Yg, Zg = None, None, None

    # Setting z^- waves = 0
    dB_x, dB_y, dB_z = genspec.generate_alfven(n_X, X_min, X_max, B_0, spectrum)
    du_x, du_y, du_z = dB_x / np.sqrt(rho), dB_y / np.sqrt(rho), dB_z / np.sqrt(rho)

    # adding fluctuations
    BXcc += dB_x
    BYcc += dB_y
    BZcc += dB_z
    MX += rho*du_x
    MY += rho*du_y
    MZ += rho*du_z

    Hy_grid[0] = rho
    Hy_grid[1] = MX  # using momentum for conserved values
    Hy_grid[2] = MY
    Hy_grid[3] = MZ

    # --- MESHBLOCK STRUCTURE --- #

    if one_D:
        n_blocks = n_X[0] / meshblock[0]
        blocks = np.array([np.arange(n_blocks), np.zeros(n_blocks), np.zeros(n_blocks)])
    else:
        n_blocks, blocks = generate_mesh_structure(folder, ath_copy)
    check_mesh_structure(blocks, n_X, meshblock)

    # --- SAVING VARIABLES --- #

    # - HYDRO
    Hy_h5 = np.zeros(shape=(N_HYDRO, n_blocks, *meshblock[::-1]))
    for h in range(N_HYDRO):
        for m in range(n_blocks):  # save to each meshblock individually
            off = blocks[:, m]
            ind_s = (meshblock*off)[::-1]
            ind_e = (meshblock*off + meshblock)[::-1]
            Hy_h5[h, m, :, :, :] = Hy_grid[h, ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]]

    Hy_grid = None
    if os.path.isfile(h5name):  # 'overwrite' old ICs
        os.remove(h5name)
    with h5py.File(h5name, 'a') as f:
        f['cons'] = Hy_h5
    Hy_h5 = None
    print('Hydro Saved Succesfully')

    # - MAGNETIC
    calc_and_save_B(BXcc, BYcc, BZcc, h5name, n_X, X_min, X_max, meshblock, n_blocks, blocks, dx, dy, dz)
    print('Magnetic Saved Successfully')
    print('Done!')



# save_folder = '/home/zade/masters_2021/generating_ics/'
# athinput_in_folder = '/home/zade/masters_2021/initial_linearwave_tests/'
# athinput_in = athinput_in_folder + 'athinput.linear_wave'
# h5name = 'ICs_from_linearwavetest.h5'
# athdf_input = '/home/zade/masters_2021/generating_ics/output_linearwave/linear_wave.out2.00000.athdf'

# create_athena_fromh5(save_folder, athinput_in_folder, athinput_in, h5name, athdf_input)

# folder = '/home/zade/masters_2021/alfvenspec_test/'
# h5name = 'ICs_linwave.h5'
# n_X = np.array([64, 64, 1])
# X_min = np.array([0., 0., 0.])
# X_max = np.array([1., 1., 0.125])
# meshblock = np.array([32, 32, 1])
# create_athena_fromics(folder, h5name, n_X, X_min, X_max, meshblock)

# folder = '/home/zade/masters_2021/alfvenspec_test/'
# h5name = 'ICs_alfvenspec.h5'
# n_X = np.array([128, 64, 64])
# X_min = np.array([0., 0., 0.])
# X_max = np.array([1., 0.5, 0.5])
# meshblock = np.array([64, 32, 32])
# create_athena_alfvenspec(folder, h5name, n_X, X_min, X_max, meshblock)