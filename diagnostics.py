'''
Code to load in plasma simulations run using Athena++ and calculate diagnostics.
All copied from code written in 2020 for my Honours project with some improvements.
'''
import glob
import os
import pickle
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from pathlib import Path
from athena_read import athdf, athinput, hst
from math import pi
from matplotlib import rc
rc('text', usetex=True)  # LaTeX labels

# Path to simulation data
PATH = "/home/zade/masters_2021/" # for testing, change!
# PATH = '/media/zade/Seagate Expansion Drive/honours_project_2020/'
# PATH = '/media/zade/STRONTIUM/honours_project_2020/'
DEFAULT_PROB = 'shear_alfven'


def load_data(output_dir, n, prob=DEFAULT_PROB):
    '''Loads data from .athdf files output from Athena++.
    '''

    def f(n):
        return folder + '.out' + output_id + '.%05d' % n + '.athdf'

    # Input
    folder = PATH + output_dir + '/' + prob  # Name of output
    output_id = '2'  # Output ID (set in input file)
    filename = f(n)
    return athdf(filename)


def load_hst(output_dir, prob=DEFAULT_PROB):
    '''Loads data from .hst files output from Athena++.
    '''
    hstLoc = PATH + output_dir + '/' + prob + '.hst'
    return hst(hstLoc)


def load_athinput(athinput_path):
    '''Loads data from athinput files.
    '''
    return athinput(PATH + athinput_path)


def load_dict(output, fname=''):
    file = 'dict.pkl'
    if fname != '':
        file = fname + '_' + file

    pkl_file = open(PATH+output+file, 'rb')
    dict = pickle.load(pkl_file)
    pkl_file.close()
    return dict


def save_dict(dict, output, fname=''):
    file = 'dict.pkl'
    if fname != '':
        file = fname + '_' + file

    output = open(PATH+output+file, 'wb')
    pickle.dump(dict, output)
    output.close()


def check_dict(output, fname=''):
    file = 'dict.pkl'
    if fname != '':
        file = fname + '_' + file
    return os.path.isfile(PATH+output+file)


def make_folder(fname):
    if not os.path.exists(fname):
        path = Path(fname)
        path.mkdir(parents=True, exist_ok=True)


def get_maxn(output_dir):
    '''Gets the total number of simulation timesteps.
    '''
    return len(glob.glob(PATH+output_dir+'/*.athdf'))


# --- MATH FUNCTIONS --- #

def rms(x, do_fluc=0, axis=(2,3,4)):
    '''Returns the RMS of a given quantity.
    Default axis keyword assumes the format [time, component, x3, x2, x1].
    '''
    x_fluc = x
    if do_fluc:
        x_mean = x.mean(axis=axis)
        x_fluc -= x.mean.reshape(*x_mean.shape, 1, 1, 1)
    return np.sqrt((x_fluc**2).mean(axis=axis))


# --- VECTOR FUNCTIONS --- #


def get_mag(x):
    '''For an array of vectors with the same number of components,
    returns the magnitude of each vector in an array of the same size.'''
    return np.sqrt((x**2).sum(axis=1))


def get_unit(x):
    '''Calculates unit vector.'''
    x_mag = get_mag(x)
    return x / x_mag.reshape(*x_mag.shape, 1)


def get_vec(x, ps):
    '''Returns array of vectors of quantity x at a given array
    of 3D grid points ps.
    '''
    indices = tuple([ps[:, 0], ps[:, 1], ps[:, 2]])
    x1 = x[0][indices]
    x2 = x[1][indices]
    x3 = x[2][indices]
    return np.array((x1, x2, x3)).T


def get_lengths(load_data=1, data=None, fname=None, prob=DEFAULT_PROB, zyx=0):
    if load_data:
        assert fname is not None, 'Must have a valid filename!'
        data = load_data(fname, 0, prob)
    else:
        assert data is not None, 'Must have a valid data file!'
    X1 = data['RootGridX1'][1] - data['RootGridX1'][0]
    X2 = data['RootGridX2'][1] - data['RootGridX2'][0]
    X3 = data['RootGridX3'][1] - data['RootGridX3'][0]
    return (X3, X2, X1) if zyx else (X1, X2, X3)


def get_rootgrid(fname, prob=DEFAULT_PROB, zyx=0):
    data = load_data(fname, 0, prob)
    return data['RootGridSize'][::-1] if zyx else data['RootGridSize']


def get_vol(fname, prob=DEFAULT_PROB):
    '''Returns the volume of the simulation domain.'''
    X1, X2, X3 = get_lengths(fname, prob)
    return abs(X1*X2*X3)  # just a check to make volume positive


# --- FOURIER FUNCTIONS --- #


def ft_array(N):
    '''
    For given N, returns an array conforming to FT standard
       [0 1 2 3 ... -N/2 -N/2+1 ... -1] of length N
       If N is odd we use N-1 instead of N i.e.:
       N = 5: [0 1 2 -2 -1]
       N = 6: [0 1 2 -3 -2 -1]
    '''
    grid = np.arange(-(N//2), (N+1)//2, 1)
    grid = np.roll(grid, (N+1)//2)

    return grid


def ft_grid(input_type, data=None, fname=None, Ls=None, Ns=None, prob=DEFAULT_PROB, k_grid=0):
    '''
    Creates a grid in k-space corresponding to the real grid given in data.
    k_grid is a boolean that when True calculates a regularly spaced array
    in k-space.
    '''

    if input_type == 'data':
        assert (data is not None), 'Must have a valid data file!'
        X1 = data['RootGridX1'][1] - data['RootGridX1'][0]
        X2 = data['RootGridX2'][1] - data['RootGridX2'][0]
        X3 = data['RootGridX3'][1] - data['RootGridX3'][0]
        
        Ls = (X3, X2, X1)
        Ns = data['RootGridSize'][::-1]
    elif input_type == 'filename':
        assert (fname is not None), 'Must have valid filename path!'
        # Z, Y, X
        Ls = get_lengths(fname, prob, zyx=1)  # box side lengths
        Ns = get_rootgrid(fname, prob, zyx=1) # number of grid points
    elif input_type == 'array':
        assert (Ls is not None and Ns is not None), 'Must have valid lengths and grid information!'
    else:
        raise ValueError('Please enter a valid input type')

    K = {}
    for k in range(3):
        K[k] = 2j*pi/Ls[k]*ft_array(Ns[k])

    # Outputs Z, Y, X
    Ks = np.meshgrid(K[0], K[1], K[2], indexing='ij')
    if k_grid:
        Ks = (Ks, np.arange(0, np.max(np.imag(K[1])), 2*pi/Ls[1]))

    return Ks

# --- EXPANDING BOX CODE --- #

def a(expansion_rate, t):
    """
    Calculates the perpendicular expansion defined in Squire2020.
    """
    return 1 + expansion_rate*t


def expand_variables(a, vector):
    """
    Takes in a time series of a vector component over the whole box and
    scales by a(t).
    """
    for i in range(1, 3):
        vector[:, i, :] *= a.reshape(*a.shape, 1, 1, 1)

    return vector


def load_time_series(output_dir, prob=DEFAULT_PROB, conserved=0):
    max_n = get_maxn(output_dir)

    t, B, u, rho = [], [], [], []

    for n in range(max_n):
        data_n = load_data(output_dir, n, prob)
        t.append(data_n['Time'])
        if conserved:
            B.append((data_n['Bcc1'], data_n['Bcc2'], data_n['Bcc3']))
            u.append((data_n['mom1'], data_n['mom2'], data_n['mom3']))
            rho.append(data_n['dens'])
        else:
            B.append((data_n['Bcc1'], data_n['Bcc2'], data_n['Bcc3']))
            u.append((data_n['vel1'], data_n['vel2'], data_n['vel3']))
            rho.append(data_n['rho'])
    
    # The full-box variables B, u, rho are indexed in the following format:
    # [timestep, component (if vector quantity, x=0 etc), z_step, y_step, x_step]
    return np.array(t), np.array(B), np.array(u), np.array(rho)
