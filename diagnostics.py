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
PATH = '/media/zade/Seagate Expansion Drive/honours_project_2020/'
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


def get_maxn(fname):
    '''Gets the total number of simulation timesteps.
    '''
    return len(glob.glob(PATH+fname+'/*.athdf'))


# --- VECTOR FUNCTIONS --- #


def get_mag(X):
    '''For an array of vectors with the same number of components,
    returns the magnitude of each vector in an array of the same size.'''
    x = np.array([X[i,:].dot(X[i,:]) for i in range(len(X[:,0]))])
    return np.sqrt(x)


def get_unit(v):
    '''Calculates unit vector.'''
    v_mag = get_mag(v)
    return np.array([v[i]/v_mag[i] for i in range(len(v))])


def get_vol(fname, prob=DEFAULT_PROB):
    '''Returns the volume of the simulation domain.'''
    data = load_data(fname, 0, prob)
    X1 = data['RootGridX1'][1] - data['RootGridX1'][0]
    X2 = data['RootGridX2'][1] - data['RootGridX2'][0]
    X3 = data['RootGridX3'][1] - data['RootGridX3'][0]
    return abs(X1*X2*X3)  # just a check to make volume positive


# --- FOURIER FUNCTIONS --- #


def ft_array(N):
    '''
    For given N, returns an array conforming to FT standard:
       [0 1 2 3 ... -N/2 -N/2+1 ... -1]
    '''
    return np.concatenate((np.arange(0, N//2, 1), [-N//2],
                           np.arange(-N//2+1, 0, 1)))


def ft_grid(data, k_grid):
    '''
    Creates a grid in k-space corresponding to the real grid given in data.
    k_grid is a boolean that when True calculates a regularly spaced array
    in k-space.
    '''

    # Z, Y, X
    p = (data['x3f'], data['x2f'], data['x1f'])
    Ls = [np.max(p[0]), np.max(p[1]), np.max(p[2])]
    Ns = [len(p[0])-1, len(p[1])-1, len(p[2])-1]

    K = {}
    for k in range(3):
        K[k] = 2j*pi/Ls[k]*ft_array(Ns[k])

    # Outputs Z, Y, X
    Ks = np.meshgrid(K[0], K[1], K[2], indexing='ij')
    if k_grid:
        Ks = (Ks, np.arange(0, np.max(np.imag(K[1])), 2*pi/Ls[1]))

    return Ks
