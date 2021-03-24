# --- IMPORTS --- #
import os
from shutil import copy

import h5py
import numpy as np
import numpy.fft as fft

import diagnostics as diag
import generate_spectrum as genspec

if diag.COMPUTER == 'local':
    from_array_path = '/home/zade/masters_2021/templates_athinput/athinput.from_array'
else:
    from_array_path = '/home/johza721/masters_2021/templates/athinput.from_array'

# --- SUPPORTING FUNCTIONS --- #

# edit_athinput - edits the corresponding athinput file with quantites input below
# ONLY WORKS FOR athinput.from_array layout
def edit_athinput(athinput, save_folder, n_X, X_min, X_max, meshblock, h5name,
                  time_lim, dt, iso_sound_speed, expand, exp_rate):
    ath_path = save_folder + athinput.split('/')[-1] + '_' + h5name.split('/')[-1].split('.')[0]
    copy(athinput, ath_path)
    ath = open(ath_path, 'r')
    list_of_lines = ath.readlines()

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
    os.system('./athena -i ' + athinput + ' -m 1 > /dev/null')  # > /dev/null supresses output, remove if need to see meshblock details
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

# --- GENERATING FUNCTIONS --- #

def create_athena_fromics(folder, h5name, n_X, X_min, X_max, meshblock,
                          energy=1., time_lim=1, dt=0.2, iso_sound_speed=1.0, expand=0, exp_rate=0., 
                          athinput=from_array_path):
    '''Function to generate an h5 file containing user set initial conditions 
       for the Athena++ from_array problem generator to start from. Generates a grid the same size as specified in the
       athinput file and calculates the quantities at each grid point, then writes to h5 file.

    Parameters
    ----------
    folder : string
        path to base folder that contains the athena binary, athinput file, and where h5 file is to be output
    h5name : string
        name of h5 file for data to be output to
    n_X : ndarray
        array containing the number of grid points in the 3 coordinate directions x1,x2,x3
    X_min : ndarray
        array containing the minimum x_n coordinate in each direction
    X_max : ndarray
        array containing the maximum x_n coordinate in each direction
    meshblock : ndarray
        array containing the meshblock dimensions in each direction
    athinput : str, optional
        name of athinput file, of the form athinput.xxx, by default '/home/zade/masters_2021/templates_athinput/athinput.from_array'
    '''
    # --- INPUTS --- #
    
    ath_copy = edit_athinput(athinput, folder, n_X, X_min, X_max, meshblock,
                             h5name, time_lim, dt, iso_sound_speed, expand, exp_rate)
    h5name = folder + h5name  # eg 'ICs_template.h5'
    N_HYDRO = 4  # number of hydro variables (e.g. density and momentum); assuming isothermal here
    # Dimension setting: 1D if only x has more than one gridpoint
    one_D = 1 if np.all(n_X[1:] == 1) else 0

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

    X_grid, (dx, dy, dz) = generate_grid(X_min, X_max, n_X)
    Hy_grid, BXcc, BYcc, BZcc = setup_hydro_grid(n_X, X_grid, N_HYDRO, Dnf, UXf, UYf, UZf, BXf, BYf, BZf)

    # initializing Alfvén wave fluctuations perpendicular to B_0 (assumed along x-axis)
    # to have same initial energy in velocity and magnetic fields.
    # dV = V / resolution = (Lx*Ly*Lz) / (Nx*Ny*Nz) = dx*dy*dz
    dV = np.prod(X_max - X_min) / np.prod(n_X)
    total_energy = 0.5*dV*np.sum(BYcc**2 + BZcc**2)
    norm_energy = np.sqrt(energy / total_energy)
    Hy_grid[1] *= norm_energy
    Hy_grid[2] *= norm_energy
    BYcc *= norm_energy
    BZcc *= norm_energy

    # --- MESHBLOCK STRUCTURE --- #

    n_blocks, blocks = make_meshblocks(folder, ath_copy, n_X, meshblock, one_D)

    # --- SAVING VARIABLES --- #

    remove_prev_h5file(h5name)

    # - HYDRO
    save_hydro_grid(h5name, Hy_grid, N_HYDRO, n_blocks, blocks, meshblock)
    print('Hydro Saved Succesfully')

    # - MAGNETIC
    # TODO: the numerical errors that arise from performing the B_cc→A→B_fc calculation
    # will cause the final cell centered B field to differ slightly in magnitude from the 
    # initial one set above. This will cause Alfvénic velocity and magnetic perturbations 
    # to not be exactly correlated, for example. Not sure how to fix.
    calc_and_save_B(BXcc, BYcc, BZcc, h5name, n_X, X_min, X_max, meshblock, n_blocks, blocks, dx, dy, dz)
    print('Magnetic Saved Successfully')
    print('Done!')

def create_athena_fromh5(save_folder, athinput_in_folder, athinput_in, h5name, athdf_input,
                         athinput_out=from_array_path):
    '''Similar function to `create_athena_fromics` but instead of user-set ICs,
    reads in an initial .athdf file and obtains the initial conditions from there.

    Parameters
    ----------
    save_folder : string
        Base folder where h5 file is to be saved.
    athinput_in_folder : string
        Base folder containing original athinput file that generated athdf.
    athinput_in : string
        Name of athinput file that generated initial athdf file.
    h5name : string
        Name of h5 file to be saved.
    athdf_input : string
        Path to initial athdf to be read.
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
    # Inverting meshblock saving method to obtain the magnetic field over the whole box
    # instead of over a meshblock
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
                             time_lim=1, dt=0.2, iso_sound_speed=1.0, expand=0, exp_rate=0.,
                             do_truncation=0, kmag_cuttoff=100., athinput=from_array_path,
                             energy=0.5, expo=-5/3, expo_prl=-2., kpeak=10., gauss_spec=0, prl_spec=0, do_mode_test=0):
    
    ath_copy = edit_athinput(athinput, folder, n_X, X_min, X_max, meshblock,
                             h5name, time_lim, dt, iso_sound_speed, expand, exp_rate)
    h5name = folder + h5name  # eg 'ICs_template.h5'                             
    N_HYDRO = 4  # number of hydro variables (e.g. density and momentum); assuming isothermal here
    # Dimension setting: 1D if only x has more than one gridpoint
    one_D = 1 if np.all(n_X[1:] == 1) else 0

    # Generate mean fields
    # Density
    Dnf = lambda X, Y, Z: np.ones(X.shape)
    UXf = lambda X, Y, Z: np.zeros(X.shape)
    UYf = lambda X, Y, Z: np.zeros(X.shape)
    UZf = lambda X, Y, Z: np.zeros(X.shape)
    BXf = lambda X, Y, Z: np.ones(X.shape)
    BYf = lambda X, Y, Z: np.zeros(X.shape)
    BZf = lambda X, Y, Z: np.zeros(X.shape)

    X_grid, (dx, dy, dz) = generate_grid(X_min, X_max, n_X)
    Hy_grid, BXcc, BYcc, BZcc = setup_hydro_grid(n_X, X_grid, N_HYDRO, Dnf, UXf, UYf, UZf, BXf, BYf, BZf)

    B_0 = np.array([BXcc, BYcc, BZcc])

    dB_y, dB_z = genspec.generate_alfven(n_X, X_min, X_max, B_0, expo,
                                         do_truncation=do_truncation, #kmag_cuttoff=kmag_cuttoff,
                                         expo_prl=expo_prl, kpeak=kpeak, gauss_spec=gauss_spec,
                                         prl_spec=prl_spec, run_test=do_mode_test)

    BYcc += dB_y
    BZcc += dB_z
    dB_y, dB_z = None, None

    # --- MESHBLOCK STRUCTURE --- #

    n_blocks, blocks = make_meshblocks(folder, ath_copy, n_X, meshblock, one_D)

    # --- SAVING VARIABLES --- #
    
    remove_prev_h5file(h5name)

    # - MAGNETIC
    
    calc_and_save_B(BXcc, BYcc, BZcc, h5name, n_X, X_min, X_max, meshblock, n_blocks, blocks, dx, dy, dz)
    print('Magnetic Saved Successfully')

    # Only looking at perturbations perpendicular to B_0, assumed to be along x-axis initially.
    # Will add perturation after t=0 corresponding to Parker spiral?
    Bcc_unpacked = np.zeros(shape=(2, *n_X[::-1]))
    with h5py.File(h5name, 'a') as f:
        for idx, b in enumerate(['bf2', 'bf3']):
            for m in range(n_blocks):  # save from each meshblock individually
                    off = blocks[:, m]
                    ind_s = (meshblock*off)[::-1]
                    ind_e = (meshblock*off + meshblock)[::-1]
                    B_fc = f[b][ m, :, :, :]

                    # linearly interpolate face centered fields to get cell-centered fields
                    # assumes evenly spaced grid points
                    # idx = 0 ⟺ y-component; idx = 1 ⟺ z-component
                    B_cc = 0.5*(B_fc[:, 1:, :] + B_fc[:, :-1, :]) if idx == 0 else 0.5*(B_fc[1:, :, :] + B_fc[:-1, :, :])

                    Bcc_unpacked[idx, ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]] = B_cc

    # Setting z^- waves = 0
    rho = Hy_grid[0]
    dB_y, dB_z = Bcc_unpacked  # no mean field along y and z axes
    du_y, du_z = dB_y / np.sqrt(rho), dB_z / np.sqrt(rho)

    # dV = V / resolution = (Lx*Ly*Lz) / (Nx*Ny*Nz) = dx*dy*dz
    dV = np.prod(X_max - X_min) / np.prod(n_X)
    # give magnetic and velocity fluctuations same initial energy
    total_energy = 0.5*dV*np.sum(dB_y**2 + dB_z**2)
    norm_energy = np.sqrt(energy / total_energy)

    Hy_grid[2] += rho*norm_energy*du_y
    Hy_grid[3] += rho*norm_energy*du_z

    dB_y, dB_z, du_y, du_z = None, None, None, None

    with h5py.File(h5name, 'a') as f:
        B_x = f['bf1'][...]
        B_y = f['bf2'][...]
        B_z = f['bf3'][...]
        # assuming B_0 is spatially homogenous ⟹ B_0 = mean(B)
        dB_x = B_x - np.mean(B_x)*np.ones_like(B_x)
        dB_y = B_y - np.mean(B_y)*np.ones_like(B_y)
        dB_z = B_z - np.mean(B_z)*np.ones_like(B_z)

        # remove all fluctuations parallel to B_0: these are not Alfvénic and are a result of numerical errors
        # otherwise rescale Alfvénic fluctations to desiered energy
        # B_y,z = B0_y,z + dB_y,z + (norm_energy - 1)*dB_y,z = B0_y,z + norm_energy*dB_y,z
        f['bf1'][...] += -dB_x
        f['bf2'][...] += dB_y*(norm_energy-1)
        f['bf3'][...] += dB_z*(norm_energy-1)


    # - HYDRO
    save_hydro_grid(h5name, Hy_grid, N_HYDRO, n_blocks, blocks, meshblock, remove_h5=0)
    print('Hydro Saved Succesfully')


def test(folder, h5name, n_X, X_min, X_max, meshblock,
         time_lim=1, dt=0.2, iso_sound_speed=1., expand=0, exp_rate=0.,
         athinput='/home/zade/masters_2021/templates_athinput/athinput.from_array',
         energy=0.5, expo=-5/3, expo_prl=-2., kpeak=10., gauss_spec=0, prl_spec=0, do_mode_test=0):

    h5name = folder + h5name  # eg 'ICs_template.h5'
    ath_copy = edit_athinput(athinput, folder, n_X, X_min, X_max, meshblock, h5name, time_lim, dt, iso_sound_speed, expand, exp_rate)
    N_HYDRO = 4  # number of hydro variables (e.g. density and momentum); assuming isothermal here
    # Dimension setting: 1D if only x has more than one gridpoint
    one_D = 1 if np.all(n_X[1:] == 1) else 0

    # Generate mean fields
    Dnf = lambda X, Y, Z: np.ones(X.shape)
    UXf = lambda X, Y, Z: np.zeros(X.shape)
    UYf = lambda X, Y, Z: np.zeros(X.shape)
    UZf = lambda X, Y, Z: np.zeros(X.shape)
    BXf = lambda X, Y, Z: np.ones(X.shape)
    BYf = lambda X, Y, Z: np.zeros(X.shape)
    BZf = lambda X, Y, Z: np.zeros(X.shape)

    X_grid, (dx, dy, dz) = generate_grid(X_min, X_max, n_X)
    Hy_grid, BXcc, BYcc, BZcc = setup_hydro_grid(n_X, X_grid, N_HYDRO, Dnf, UXf, UYf, UZf, BXf, BYf, BZf)

    B_0 = np.array([BXcc, BYcc, BZcc])
    rho = Hy_grid[0]

    # Setting z^- waves = 0
    dB_y, dB_z = genspec.generate_alfven(n_X, X_min, X_max, B_0,
                                               expo, expo_prl=expo_prl, kpeak=kpeak, gauss_spec=gauss_spec,
                                               prl_spec=prl_spec, run_test=do_mode_test)
    du_y, du_z = dB_y / np.sqrt(rho), dB_z / np.sqrt(rho)

    # dV = V / resolution = (Lx*Ly*Lz) / (Nx*Ny*Nz) = dx*dy*dz
    dV = np.prod(X_max - X_min) / np.prod(n_X)
    # give magnetic and velocity fluctuations same initial energy
    total_energy = 0.5*dV*np.sum(dB_y**2 + dB_z**2)
    norm_energy = np.sqrt(energy / total_energy)

    du_y *= norm_energy
    du_z *= norm_energy
    dB_y *= norm_energy
    dB_z *= norm_energy

    # adding fluctuations
    BYcc += dB_y
    BZcc += dB_z
    Hy_grid[2] += rho*du_y
    Hy_grid[3] += rho*du_z

    # --- MESHBLOCK STRUCTURE --- #

    n_blocks, blocks = make_meshblocks(folder, ath_copy, n_X, meshblock, one_D)

    # --- SAVING VARIABLES --- #

    # - HYDRO
    save_hydro_grid(h5name, Hy_grid, N_HYDRO, n_blocks, blocks, meshblock)
    print('Hydro Saved Succesfully')

    # - MAGNETIC
    calc_and_save_B(BXcc, BYcc, BZcc, h5name, n_X, X_min, X_max, meshblock, n_blocks, blocks, dx, dy, dz)
    print('Magnetic Saved Successfully')
    print('Done!')