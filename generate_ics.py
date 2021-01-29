# --- IMPORTS --- #
import diagnostics as diag
import numpy as np
import numpy.fft as fft
import h5py
import os
from shutil import copy

# --- SUPPORTING FUNCTIONS --- #

# edit_athinput - edits the corresponding athinput file with quantites 
# input below
def edit_athinput(athinput, n_X, X_min, X_max, meshblock, h5name):
    ath_path = athinput + h5name.split('/')[-1].split('.')[0]
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
    if component == 0:  # x-component
        A = np.concatenate((A, A[:, :, 0].reshape((*A.shape[:2], 1))), axis=2)
        A = np.concatenate((A[:, -1, :].reshape((A.shape[0], 1, A.shape[2])), A, A[:, 0, :].reshape((A.shape[0], 1, A.shape[2]))), axis=1)
        A = np.concatenate((A[-1, :, :].reshape((1, *A.shape[1:])), A, A[0, :, :].reshape((1, *A.shape[1:]))), axis=0)
    elif component == 1:  # y-component
        A = np.concatenate((A[:, :, -1].reshape((*A.shape[:2], 1)), A, A[:, :, 0].reshape((*A.shape[:2], 1))), axis=2)
        A = np.concatenate((A, A[:, 0, :].reshape((A.shape[0], 1, A.shape[2]))), axis=1)
        A = np.concatenate((A[-1, :, :].reshape((1, *A.shape[1:])), A, A[0, :, :].reshape((1, *A.shape[1:]))), axis=0)
    elif component == 2:  # z-component
        A = np.concatenate((A[:, :, -1].reshape((*A.shape[:2], 1)), A, A[:, :, 0].reshape((*A.shape[:2], 1))), axis=2)
        A = np.concatenate((A[:, -1, :].reshape((A.shape[0], 1, A.shape[2])), A, A[:, 0, :].reshape((A.shape[0], 1, A.shape[2]))), axis=1)
        A = np.concatenate((A, A[0, :, :].reshape((1, *A.shape[1:]))), axis=0)
    return A


# --- GENERATING FUNCTIONS --- #

def create_athena_fromics(folder, athinput, h5name, n_X, X_min, X_max, meshblock, run_athena=0, run_visit=0, from_h5=0, athdf_input=None):
    # --- INPUTS --- #

    # Folder and h5 filename - needed for meshblock structure
    h5name = folder + h5name  #'ICs_uBanticoscorr.h5'
    data_output = folder + 'output_' + h5name.split('/')[-1].split('.')[0]

    # Number of grid points and meshblocks - must match input
    # In order [X, Y, Z]
    # TODO: add a function that writes these to athinput? Pack into one function
    # n_X = np.array([128, 128, 1])
    # X_min = np.array([0, 0, 0])
    # X_max = np.array([1, 1, 0.125])
    # meshblock = np.array([64, 64, 1])

    ath_copy = edit_athinput(folder+athinput, n_X, X_min, X_max, meshblock, h5name)
    N_HYDRO = 4  # number of hydro variables (e.g. density and momentum)
    # Dimension setting: 1D if only x has more than one gridpoint
    one_D = 1 if np.all(n_X[1:] == 1) else 0
    Hy_grid = np.zeros(shape=(N_HYDRO, *n_X[::-1])) # expanding n_X tuple
    
    if from_h5:
        assert athdf_input is not None, 'Must include path to input HDF5 or .athdf file!'
        
        f = h5py.File(athdf_input, 'r')
        is_cons = 'cons' in list(f.keys())  # conserved variables if 1, primitive if false
        # TODO: #2 undo meshblock code
    else:
        # Variable functions. Only focus on isothermal
        Dnf = lambda X, Y, Z: np.ones(X.shape)  # density
        UXf = lambda X, Y, Z: np.zeros(X.shape)  # velocity components
        UYf = lambda X, Y, Z: np.zeros(X.shape)
        UZf = lambda X, Y, Z: -0.01*np.sin((2*np.pi/X_max[0]) * X + (2*np.pi/X_max[1]) * Y)
        BXf = lambda X, Y, Z: np.ones(X.shape) # magnetic components
        BYf = lambda X, Y, Z: np.zeros(X.shape)
        BZf = lambda X, Y, Z: -0.01*np.sin((2*np.pi/X_max[0]) * X + (2*np.pi/X_max[1]) * Y)

        # --- GRID CREATION --- #

        # edge (e) and centre (grid=g) coordinates
        xe = np.linspace(X_min[0], X_max[0], n_X[0]+1)
        ye = np.linspace(X_min[1], X_max[1], n_X[1]+1)
        ze = np.linspace(X_min[2], X_max[2], n_X[2]+1)
        xg = 0.5*(xe[:-1] + xe[1:])
        yg = 0.5*(ye[:-1] + ye[1:])
        zg = 0.5*(ze[:-1] + ze[1:])

        dx = xg[1] - xg[0]
        dy = np.inf if n_X[1] == 1 else yg[1] - yg[0]
        dz = np.inf if n_X[2] == 1 else zg[1] - zg[0]
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
        # if NHYDRO == 5:

        Xg, Yg, Zg = None, None, None

    # --- MESHBLOCK STRUCTURE --- #

    # Run shell command to obtain mesh_structure.dat file
    # Run meshblock check function
    if one_D:
        n_blocks = n_X[0] / meshblock[0]
        blocks = np.array([np.arange(n_blocks), np.zeros(n_blocks), np.zeros(n_blocks)])
    else:
        cdir = os.getcwd()
        os.chdir(folder)
        os.system('./athena -i ' + ath_copy + ' -m 1')  # > /dev/null supresses output, remove if need to see meshblock details
        blocks = read_mesh_structure(folder + 'mesh_structure.dat')
        n_blocks = blocks.shape[1]
        os.chdir(cdir)

    check_mesh_structure(blocks, n_X, meshblock)

    # --- SAVING VARIABLES --- #

    # - HYDRO
    Hy_h5 = np.zeros(shape=(N_HYDRO, n_blocks, *meshblock[::-1]))
    for h in range(N_HYDRO):
        for m in range(n_blocks):  # save to each meshblock individually
            off = blocks[:, m]
            ind_s = meshblock*off
            ind_e = meshblock*off + meshblock
            Hy_h5[h, m, :, :, :] = Hy_grid[h, ind_s[2]:ind_e[2], ind_s[1]:ind_e[1], ind_s[0]:ind_e[0]]

    Hy_grid = None
    if os.path.isfile(h5name):  # 'overwrite' old ICs
        os.remove(h5name)
    with h5py.File(h5name, 'w') as f:
        f['cons'] = Hy_h5
    Hy_h5 = None
    print('Hydro Saved Succesfully')

    # - MAGNETIC

    # Get mean of B-field (curl takes this away)
    B_mean = np.array([BXcc.mean(), BYcc.mean(), BZcc.mean()])

    # Calculate A from B using Fourier space and check this is right
    K = {}
    for k in range(3):
        if n_X[k] > 1:
            K[k] = 2j*np.pi/(X_max[k] - X_min[k])*diag.ft_array(n_X[k])
        else:
            K[k] = np.array(0j)


    K_z, K_y, K_x = np.meshgrid(K[2], K[1], K[0], indexing='ij')
    # Need to transpose x and y for it to work for some reason
    K_x, K_y, K_z = K_x.transpose((0, 2, 1)), K_y.transpose((0, 2, 1)), K_z.transpose((0, 2, 1))
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

    # A is cell-centred; shift to edges
    A_x, A_y, A_z = shift_and_extend_A(A_x, A_y, A_z) 

    # Calculate B from A using finite-difference curl (MATLAB script is how Athena does it)
    # Bx
    # TODO: check curl is working.
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
    with h5py.File(h5name, 'r+') as f:
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
    with h5py.File(h5name, 'r+') as f:
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
    with h5py.File(h5name, 'r+') as f:
        f['bf3'] = B_h5
    print('Magnetic Saved Successfully')
    print('Done!')

    # --- RUN ATHENA --- #
    if run_athena:
        os.system('mpiexec -n 2 ./athena -i ' + ath_copy + ' -d ' + data_output)
    if run_visit:
        cdir = os.getcwd()
        os.chdir('./' + data_output)
        os.system('visit')
        os.chdir(cdir)

def create_athena_fromh5(folder, athinput, h5name, n_X, X_min, X_max, meshblock, run_athena, run_visit):
    return 0
    # ath_copy = edit_athinput()
    # h5_init = h5py.File('', 'r')  # get filename

    
    # # --- SAVING VARIABLES --- #

    # # - HYDRO
    # if os.path.isfile(h5name):  # 'overwrite' old ICs
    #     os.remove(h5name)
    # with h5py.File(h5name, 'w') as f:
    #     f['cons'] = h5_init['cons']
    # print('Hydro Saved Succesfully')

    # # - MAGNETIC
    # with h5py.File(h5name, 'r+') as f:
    #     f['bf1'] = B_h5
    #     f['bf2'] = B_h5
    #     f['bf3'] = B_h5
    

    # print('Magnetic Saved Successfully')
    # print('Done!')

    # # --- RUN ATHENA --- #
    # if run_athena:
    #     os.system('mpiexec -n 2 ./athena -i ' + ath_copy + ' -d ' + data_output)
    # if run_visit:
    #     cdir = os.getcwd()
    #     os.chdir('./' + data_output)
    #     os.system('visit')
    #     os.chdir(cdir)

# folder = '/home/zade/masters_2021/generating_ics/'
# athinput = 'athinput.from_array'
# h5name = 'ICs_linwavecompare.h5'
# n_X = np.array([64, 64, 1])
# X_min = np.array([0, 0, 0])
# X_max = np.array([1, 1, 0.125])
# meshblock = np.array([32, 32, 1])

# start(folder, athinput, h5name, n_X, X_min, X_max, meshblock, run_athena=1)