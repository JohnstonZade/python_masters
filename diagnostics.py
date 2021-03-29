'''
Code to load in plasma simulations run using Athena++ and calculate diagnostics.
All copied from code written in 2020 for my Honours project with some improvements.
'''
import glob
import os
import pickle
import numpy as np
from pathlib import Path
from athena_read import athdf, athinput, hst
# from matplotlib import rc
# rc('text', usetex=True)  # LaTeX labels

COMPUTER = 'local'
# COMPUTER = 'nesi'

if COMPUTER == 'local':
    # Path to simulation data
    PATH = "/home/zade/masters_2021/"
    # PATH = '/media/zade/Seagate Expansion Drive/honours_project_2020/'
    # PATH = '/media/zade/STRONTIUM/honours_project_2020/'
elif COMPUTER == 'nesi':
    # add NeSI paths here
    # PATH = '/nesi/nobackup/uoo02637/zade/masters_2021/' # scratch
    PATH = '/nesi/project/uoo02637/zade/masters_2021/' # project
DEFAULT_PROB = 'from_array'


def format_path(output_dir):
    if PATH not in output_dir:
        output_dir = PATH + output_dir
    if output_dir[-1] != '/':
        output_dir += '/'
    return output_dir


def load_data(output_dir, n, prob=DEFAULT_PROB):
    '''Loads data from .athdf files output from Athena++.

    Parameters
    ----------
    output_dir : string
        path to directory containing .athdf file(s)
    n : int
        integer describing which snapshot to choose
    prob : string, optional
        the Athena++ problem generator name, by default DEFAULT_PROB

    Returns
    -------
    dict
        a dictionary containing information on all data in the simulation. Using 
        the `read_python.py` script in `/athena/vis/python/`.
    '''
    
    max_n = get_maxn(output_dir)
    assert n in range(0, max_n), 'n must be between 0 and ' + str(max_n)

    # '$folder.out$output_id.xxxxx.athdf'
    def f(n):
        return folder + '.out' + output_id + '.%05d' % n + '.athdf'

    # Input
    folder = format_path(output_dir) + prob  # Name of output
    output_id = '2'  # Output ID (set in input file)
    filename = f(n)
    return athdf(filename)


def load_hst(output_dir, prob=DEFAULT_PROB):
    '''Loads data from .hst files output from Athena++.
    '''
    hstLoc = format_path(output_dir) + prob + '.hst'
    return hst(hstLoc)


def load_athinput(athinput_path):
    '''Loads data from athinput files.
    '''
    return athinput(format_path(athinput_path))


def load_dict(output_dir, fname=''):
    file = 'dict.pkl'
    if fname != '':
        file = fname + '_' + file

    pkl_file = open(format_path(output_dir)+file, 'rb')
    dict = pickle.load(pkl_file)
    pkl_file.close()
    return dict


def save_dict(dict, output_dir, fname=''):
    file = 'dict.pkl'
    if fname != '':
        file = fname + '_' + file

    output_dir = format_path(output_dir)
    output = open(output_dir+file, 'wb')
    pickle.dump(dict, output)
    output.close()


def check_dict(output_dir, fname=''):
    file = 'dict.pkl'
    if fname != '':
        file = fname + '_' + file
    return os.path.isfile(format_path(output_dir)+file)


def make_folder(dir_name):
    if not os.path.exists(dir_name):
        path = Path(dir_name)
        path.mkdir(parents=True, exist_ok=True)


def get_maxn(output_dir):
    '''Gets the total number of simulation timesteps.
    '''
    return len(glob.glob(format_path(output_dir)+'*.athdf'))


# --- MATH FUNCTIONS --- #

def box_avg(x):
    # box indicies are always the last 3
    len_shape = len(x.shape)
    axes = tuple(np.arange(len_shape-3, len_shape))
    return np.mean(x, axis=axes)

def dot_prod(x, y, axis):
    return np.sum(x*y, axis=axis)

def rms(x):
    '''Returns the fluctuation RMS of a given quantity.
    Assumes that the spatial indicies are always the last three indices of
    an array e.g. [time, component, x3, x2, x1].
    '''
    x_mean = box_avg(x)
    dx = x - x_mean.reshape(*x_mean.shape, 1, 1, 1)
    if len(x.shape) == 5: # vector quantity
        dx2 = dot_prod(dx, dx, 1)
    else:  # scalar quantity
        dx2 = dx**2
    return np.sqrt(box_avg(dx2))


# --- VECTOR FUNCTIONS --- #


def get_mag(x):
    '''For an array of vectors with the same number of components,
    returns the magnitude of each vector in an array of the same size.
    
    E.g. x = [[1, 0, 1],\n
              [3, 4, 0],\n
              [1, 1, 1]]
         get_mag(x) = [√2, 5, √3]
    '''
    return np.sqrt(dot_prod(x, x, 1))


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


def get_lengths(load_data=1, data=None, output_dir=None, prob=DEFAULT_PROB, zyx=0):
    if load_data:
        assert (output_dir is not None), 'Must have a valid directory path!'
        data = load_data(output_dir, 0, prob)
    else:
        assert (data is not None), 'Must have a valid data file!'
    X1 = data['RootGridX1'][1] - data['RootGridX1'][0]
    X2 = data['RootGridX2'][1] - data['RootGridX2'][0]
    X3 = data['RootGridX3'][1] - data['RootGridX3'][0]
    return (X3, X2, X1) if zyx else (X1, X2, X3)


def get_rootgrid(output_dir, prob=DEFAULT_PROB, zyx=0):
    data = load_data(output_dir, 0, prob)
    return data['RootGridSize'][::-1] if zyx else data['RootGridSize']


def get_vol(output_dir, prob=DEFAULT_PROB):
    '''Returns the volume of the simulation domain.'''
    # TODO: Will need to multiply by a^2 if using in expanding box
    X1, X2, X3 = get_lengths(output_dir, prob)
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
    array = np.arange(-(N//2), (N+1)//2, 1)
    array = np.roll(array, (N+1)//2)

    return array


def ft_grid(input_type, data=None, output_dir=None, Ls=None, Ns=None, prob=DEFAULT_PROB, k_grid=0):
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
    
    elif input_type == 'output':
        assert (output_dir is not None), 'Must have a valid directory path!'
        Ls = get_lengths(output_dir, prob, zyx=1)  # box side lengths
        Ns = get_rootgrid(output_dir, prob, zyx=1) # number of grid points
    
    elif input_type == 'array':
        assert (Ls is not None and Ns is not None), 'Must have valid lengths and grid information!'
    
    else:
        raise ValueError('Please enter a valid input type')

    # Corresponds to Athena++ standard k=0 ⟺ Z, 1 ⟺ Y, 2 ⟺ X
    K = {}
    for k in range(3):
        K[k] = 2j*np.pi/Ls[k]*ft_array(Ns[k])

    Ks = np.meshgrid(K[0], K[1], K[2], indexing='ij')
    if k_grid:
        Ks = (Ks, np.arange(0, np.max(np.imag(K[1])), 2*np.pi/Ls[1]))

    return Ks


# --- MHD TURBULENCE DIAGNOSTICS --- #

def cross_helicity(rho, u_perp, B_perp): 
    udotB = dot_prod(u_perp, B_perp, 1)
    u2, B2 = dot_prod(u_perp, u_perp, 1), dot_prod(B_perp, B_perp, 1)

    return 2 * box_avg(np.sqrt(rho) * udotB) / box_avg(rho*u2 + B2)


def beta(rho, B_mag, c_s_init, expansion_rate, t):
    # rho and B_mag should be of the form rho[time, x, y, z]
    # from Squire2020, line before Eq. 4
    # assuming isothermal eos
    sound_speed = expand_sound_speed(c_s_init, expansion_rate, t)
    rhoB2_avg = box_avg(rho / B_mag**2)
    return 2 * sound_speed**2 * rhoB2_avg

def mag_compress_Squire2020(B):
    # δ(B^2) / (δB_vec)^2
    B_mag2 = dot_prod(B, B, 1)
    rms_Bmag2 = rms(B_mag2)
    sqr_rmsB = rms(B)**2
    return rms_Bmag2 / sqr_rmsB

def mag_compress_Shoda2021(B):
    # (δB)^2 / (δB_vec)^2
    B_mag = np.sqrt(dot_prod(B, B, 1))
    sqr_rmsBmag = rms(B_mag)**2
    sqr_rmsB = rms(B)**2
    return sqr_rmsBmag / sqr_rmsB

def norm_fluc_amp(fluc, background):
    # quantities of the form <B_⟂^2> / <B_x>**2
    # or <ρu_⟂^2> / <B_x>**2
    mean_fluc = box_avg(fluc)
    mean_bg_sqr = box_avg(background)**2
    return mean_fluc / mean_bg_sqr

# --- EXPANDING BOX CODE --- #

# def time_to_dist

def switchback_fraction(B_x, B_mag, B0_x):
    b_x = B_x / B_mag  # unit vector in x direction
    N_cells = b_x[0].size 
    # if B0_x is positive, switchbacks are in negative direction
    # and vice versa
    sb_frac = []
    for i in range(b_x.shape[0]):
        b_x_temp = b_x[i]
        N_flipped_b = b_x_temp[np.sign(B0_x)*b_x_temp < 0.0].size
        sb_frac.append(N_flipped_b / N_cells)
    return np.array(sb_frac)

def a(expansion_rate, t):
    """
    Calculates the perpendicular expansion defined in Squire2020.
    """
    return 1 + expansion_rate*t


def expand_sound_speed(init_c_s, expansion_rate, t):
    # tempurature evolution is adiabatic
    return init_c_s * (a(expansion_rate, t))**(-2/3)


def expand_variables(a, vector):
    """
    Takes in a time series of a vector component over the whole box and
    scales by a(t).
    """
    for i in range(1, 3): # i = 1 ⟺ y, i = 2 ⟺ z
        vector[:, i, :] *= a.reshape(*a.shape, 1, 1, 1)

    return vector


def load_time_series(output_dir, n_start=0, n_end=-1, conserved=0, just_time=0, prob=DEFAULT_PROB):
    # By default load all snapshots
    # Otherwise load all snapshots n_start...(n_end - 1)
    max_n = get_maxn(output_dir)
    if n_end >= 0:
        assert n_end > n_start, 'Please choose a valid range!'
        n_end = min(n_end, max_n)  # making sure we don't overstep max n by mistake
    else:
        n_end = max_n
    

    t, B, u, rho = [], [], [], []

    for n in range(n_start, n_end):
        data_n = load_data(output_dir, n, prob)
        t.append(data_n['Time'])
        if not just_time:
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
    if just_time:
        return np.array(t)
    else: 
        return np.array(t), np.array(B), np.array(u), np.array(rho)
