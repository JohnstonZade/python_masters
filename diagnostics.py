'''
Code to load in plasma simulations run using Athena++ and calculate diagnostics.
All copied from code written in 2020 for my Honours project with some improvements.
'''
import glob
import os
import pickle
import h5py
import numpy as np
import scipy.ndimage as ndimage
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from pathlib import Path
from itertools import permutations as perm
from athena_read import athdf, athinput, hst
from project_paths import PATH
import energy_evo as energy
# from matplotlib import rc
# rc('text', usetex=True)  # LaTeX labels

DEFAULT_PROB = 'from_array'


def format_path(output_dir, format=1):
    if format:
        if PATH not in output_dir:
            output_dir = PATH + output_dir
        if output_dir[-1] != '/':
            output_dir += '/'
    return output_dir


def load_data(output_dir, n, prob=DEFAULT_PROB, do_path_format=1, method='matt'):
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
        the `athena_read.py` script in `/athena/vis/python/`.
    '''
    
    def change_coords(data, a, method):
        # Using primed to unprimed transformation
        # from Matt Kunz's notes
        Λ = np.array([1, a, a])  # diagonal matrix
        λ = a**2  # determinant of Λ = a^2
        matts_method = method == 'matt'
        data['rho'] /= λ if matts_method else 1
        for i in range(3):
            B_string = 'Bcc' + str(i+1)
            u_string = 'vel' + str(i+1)
            data[u_string] *= Λ[i]
            data[B_string] *= Λ[i] / λ if matts_method else Λ[i]
    
    max_n = get_maxn(output_dir, do_path_format=do_path_format)
    assert n in range(0, max_n), 'n must be between 0 and ' + str(max_n)

    # '$folder.out$output_id.xxxxx.athdf'
    def f(n):
        return folder + '.out' + output_id + '.%05d' % n + '.athdf'

    # Input
    folder = format_path(output_dir, do_path_format) + prob  # Name of output
    output_id = '2'  # Output ID (set in input file)
    filename = f(n)

    data = athdf(filename)
    if method != 'nothing':
        # Rescale perpendicular components automatically
        current_a = data['a_exp']
        # use method = 'matt' for Matt's equations
        # else method = 'jono' for Jono's equations
        # where u_⟂ = a*u'_⟂ and same for B_⟂
        change_coords(data, current_a, method)
    return data


def load_and_scale_h5(output_dir, prob=DEFAULT_PROB, do_path_format=1, method='matt', undo=0):
    # Assumes output is in primitive form
    # Rescaling by factors of a to allow for correct plotting in Paraview
    # Need to check if it works properly
    max_n = get_maxn(output_dir, do_path_format=do_path_format)
    
    # '$folder.out$output_id.xxxxx.athdf'
    def filename(n):
        return folder + '.out' + output_id + '.%05d' % n + '.athdf'

    for n in range(max_n):
        # Input
        folder = format_path(output_dir, do_path_format) + prob  # Name of output
        output_id = '2'  # Output ID (set in input file)
        h5name = filename(n)
        a = athdf(h5name)['a_exp']
        Λ = np.array([1, a, a])  # diagonal matrix
        λ = a**2  # determinant of Λ = a^2
        print('n = ' + str(n))
        matts_method = method == 'matt'
        with h5py.File(h5name, 'a') as f:
            prim = f['prim'][...]
            B = f['B'][...]
            grid_x2_f = f['x2f'][...]
            grid_x2_v = f['x2v'][...]
            grid_x3_f = f['x3f'][...]
            grid_x3_v = f['x3v'][...]
            if undo:
                prim[0] *= λ if matts_method else 1
                prim[1] /= Λ[0]
                prim[2] /= Λ[1]
                prim[3] /= Λ[2]
                B[0] /= Λ[0] / λ if matts_method else Λ[0]
                B[1] /= Λ[1] / λ if matts_method else Λ[1]
                B[2] /= Λ[2] / λ if matts_method else Λ[2]
                grid_x2_f /= a
                grid_x2_v /= a
                grid_x3_f /= a
                grid_x3_v /= a
            else:
                prim[0] /= λ if matts_method else 1
                prim[1] *= Λ[0]
                prim[2] *= Λ[1]
                prim[3] *= Λ[2]
                B[0] *= Λ[0] / λ if matts_method else Λ[0]
                B[1] *= Λ[1] / λ if matts_method else Λ[1]
                B[2] *= Λ[2] / λ if matts_method else Λ[2]
                grid_x2_f *= a
                grid_x2_v *= a
                grid_x3_f *= a
                grid_x3_v *= a
            
            f['prim'][...] = prim
            f['B'][...] = B
            f['x2f'][...] = grid_x2_f
            f['x2v'][...] = grid_x2_v
            f['x3f'][...] = grid_x3_f
            f['x3v'][...] = grid_x3_v
            
            
def load_hst(output_dir, adot, prob=DEFAULT_PROB, do_path_format=1, method='matt'):
    '''Loads data from .hst files output from Athena++.
    '''
    hstLoc = format_path(output_dir, do_path_format) + prob + '.hst'
    hst_data = hst(hstLoc)

    if 'a' not in hst_data.keys():
        t_hst = hst_data['time']
        hst_data['a'] = 1.0 + adot*t_hst
    
    a_hst = hst_data['a']
    if method == 'nothing':
        return hst_data
    elif method == 'matt':
        hst_data['mass'] /= a_hst**2
        hst_data['1-mom'] /= a_hst**2
        hst_data['2-mom'] /= a_hst
        hst_data['3-mom'] /= a_hst
        hst_data['1-KE'] /= a_hst**2  # Perp kinetic energies have same scaling
        hst_data['1-ME'] /= a_hst**4
        hst_data['2-ME'] /= a_hst**2
        hst_data['3-ME'] /= a_hst**2
    elif method == 'jono':
        hst_data['2-mom'] *= a_hst  # perp momenta ~ u_perp
        hst_data['3-mom'] *= a_hst
        hst_data['2-KE'] *= a_hst**2  # perp energies ~ u_perp^2 and B_perp^2
        hst_data['3-KE'] *= a_hst**2
        hst_data['2-ME'] *= a_hst**2
        hst_data['3-ME'] *= a_hst**2

    return hst_data


def load_time_series(output_dir, n_start=0, n_end=-1, conserved=0, just_time=0, prob=DEFAULT_PROB, method='matt'):
    # By default load all snapshots
    # Otherwise load all snapshots n_start...(n_end - 1)
    max_n = get_maxn(output_dir)
    if n_end >= 0:
        assert n_end > n_start, 'Please choose a valid range!'
        n_end = min(n_end, max_n)  # making sure we don't overstep max n by mistake
    else:
        n_end = max_n
    
    t, a, B, u, rho = [], [], [], [], []

    for n in range(n_start, n_end):
        data_n = load_data(output_dir, n, prob, method=method)
        t.append(data_n['Time'])
        a.append(data_n['a_exp'])
        if not just_time:
            B.append((data_n['Bcc1'], data_n['Bcc2'], data_n['Bcc3']))
            if conserved:
                u.append((data_n['mom1'], data_n['mom2'], data_n['mom3']))
                rho.append(data_n['dens'])
            else:
                u.append((data_n['vel1'], data_n['vel2'], data_n['vel3']))
                rho.append(data_n['rho'])
    
    # The full-box variables B, u, rho are indexed in the following format:
    # [timestep, component (if vector quantity, x=0 etc), z_step, y_step, x_step]
    if just_time:
        return np.array(t), np.array(a)
    else: 
        return np.array(t), np.array(a), np.array(B), np.array(u), np.array(rho)


def load_athinput(athinput_path, do_path_format=1):
    '''Loads data from athinput files.
    '''
    return athinput(format_path(athinput_path, do_path_format))


def load_dict(output_dir, fname='', do_path_format=1):
    file = 'dict.pkl'
    if fname != '':
        file = fname + '_' + file

    with open(format_path(output_dir, do_path_format)+file, 'rb') as pkl_file:
        dict = pickle.load(pkl_file)
    return dict


def save_dict(dict, output_dir, fname='', do_path_format=1):
    file = 'dict.pkl'
    if fname != '':
        file = fname + '_' + file

    with open(format_path(output_dir, do_path_format)+file, 'wb') as output:
        pickle.dump(dict, output, pickle.HIGHEST_PROTOCOL)


def check_dict(output_dir, fname='', do_path_format=1):
    file = 'dict.pkl'
    if fname != '':
        file = fname + '_' + file
    return os.path.isfile(format_path(output_dir, do_path_format)+file)


def make_folder(dir_name):
    if not os.path.exists(dir_name):
        path = Path(dir_name)
        path.mkdir(parents=True, exist_ok=True)


def get_maxn(output_dir, do_path_format=1):
    '''Gets the total number of simulation timesteps.
    '''
    return len(glob.glob(format_path(output_dir, do_path_format)+'*.athdf'))

def get_meshblocks(n_X, n_cpus):
    def divisors(n):
        divs = [1]
        for i in range(2,int(np.sqrt(n))+1):
            if n%i == 0:
                divs.extend([i,n//i])
        divs.extend([n])
        return np.sort(np.array(list(set(divs))))[::-1]
    def get_divs(n_points, n_dir, mesh, min_divs=10):
        if n_dir == 1:
            return [1], n_points
        n = n_points // mesh
        divs = divisors(n)
        return divs[(min_divs <= divs) & (divs <= n_dir) & (n_dir % divs == 0)], n

    possible, sa_to_v = [], []
    nx, ny, nz = n_X
    MX, n = get_divs(np.prod(n_X), nx, n_cpus, min_divs=(nx//10))
    for mx in MX:
        MY, n2 = get_divs(n, ny, mx)
        for my in MY:
            MZ = get_divs(n2, nz, my)[0]
            for mz in MZ:
                tup = (mx, my, mz)
                perm_in_poss = any([p in possible for p in list(perm(tup))])
                if (np.prod(n_X) / (mx*my*mz) == n_cpus) and not perm_in_poss:
                        possible.append(tup)
                        sa_to_v.append(2*(mx*my + my*mz + mz*mx) / (mx*my*mz))
    temp, meshblocks = zip(*sorted(zip(sa_to_v, possible)))
    meshblocks = np.array(meshblocks)
    return meshblocks
    # return np.array(possible), np.array(sa_to_v)


def get_est_cputime(dx, resolution, n_cpus, a_start, a_end, a_dot, cputime=0):
    # Obtained from the six simulations in superexpand_numerical
    avg_time_per_stepres = 1e-6
    dt_cfl = 0.5*0.15*0.3*dx
    tlim = (a_end - a_start) / a_dot
    n_steps = tlim / dt_cfl
    if cputime:
        return avg_time_per_stepres * resolution * n_steps
    else:
        return avg_time_per_stepres * resolution * n_steps / n_cpus

def format_cputime(est_time, n_cpus, cputime=0, add_suffix=0, day_format=1):
    est_time_hrs = est_time / (60*60)
    est_time_days = (est_time_hrs - est_time_hrs % 24) // 24
    est_tim_mins = round((est_time_hrs % 1) * 12)*5 % 60
    prefix_str = 'Est. CPU Time: ' if cputime else 'Est. Physical Time: '
    suffix_str = ' (' + str(n_cpus) + 'x physical time)' if (cputime and add_suffix) else ''
    if day_format:
        est_time_hrs %= 24
        return prefix_str + str(int(est_time_days)) + ' days ' + str(int(est_time_hrs)) + ' hrs ' + str(int(est_tim_mins)) + ' mins' + suffix_str
    else:
        return prefix_str + str(round(est_time_hrs, 2)) + ' hrs' + suffix_str

def get_split_cputime(dx, resolution, n_cpus_list, a_list, a_dot, total_time=1, scale_prl=1):
    cpu_time, phys_time = [], []
    res = np.copy(resolution)
    for idx, a in enumerate(a_list[1:]):
        a_start = a_list[idx]
        a_end = a
        n_cpus = n_cpus_list[idx] if n_cpus_list.size > 1 else n_cpus_list[0]
        print(str(a_start) + '->' + str(a_end))
        print(res)
        cpu_time.append(get_est_cputime(dx, res.prod(), n_cpus, a_start, a_end, a_dot, cputime=1))
        phys_time.append(get_est_cputime(dx, res.prod(), n_cpus, a_start, a_end, a_dot))
        if scale_prl:
            res[0] //= (a / a_list[idx])
            dx *= (a / a_list[idx])
        else:
            res[1:] *= a_start

    cpu_time, phys_time = np.array(cpu_time), np.array(phys_time)
    if not total_time:
        return (['a = ' + str(a_list[i]) + '->' + str(a_list[i+1]) + ': ' + format_cputime(t, n_cpus, cputime=1, day_format=0) for i, t in enumerate(cpu_time)],
               ['a = ' + str(a_list[i]) + '->' + str(a_list[i+1]) + ': ' + format_cputime(t, n_cpus, day_format=0) for i, t in enumerate(phys_time)])

    tot_cpu = cpu_time.sum()
    tot_phys = phys_time.sum()
    return format_cputime(tot_cpu, n_cpus, cputime=1, day_format=0), format_cputime(tot_phys, n_cpus, day_format=0)
            

# --- MATH FUNCTIONS --- #

def box_avg(x, reshape=0):
    # box indicies are always the last 3
    len_shape = len(x.shape)
    if len_shape < 3:
        # either already averaged or not a box quantity
        return x
    axes = tuple(np.arange(len_shape-3, len_shape))
    avg = np.mean(x, axis=axes)
    if reshape:
        shape = avg.shape
        # add back in grid columns for broadcasting
        avg = avg.reshape(*shape, 1, 1, 1)
    return avg

def dot_prod(x, y, axis=1, reshape=0):
    x_dot_y = np.sum(x*y, axis=axis)
    if reshape:
        shape = x_dot_y.shape
        shape_tup = (1, *shape) if axis == 0 else (shape[0], 1, *shape[1:])
        # add back in component column for broadcasting
        x_dot_y = x_dot_y.reshape(shape_tup)
    return x_dot_y

def rms(x, dot_vector=1, do_fluc=1):
    '''Returns the fluctuation RMS of a given quantity by default.
    When do_fluc=0, just computes the RMS with no mean subtracted. If mean = 0
    (as in the case of superpositions of sinusoidal Alfvén waves), it does not
    matter whether do_fluc=0 or 1.
    Assumes that the spatial indicies are always the last three indices of
    an array e.g. [time, component, x3, x2, x1].
    '''
    x_mean = box_avg(x)
    if not do_fluc:
        x_mean *= 0.
    dx = x - x_mean.reshape(*x_mean.shape, 1, 1, 1)
    if len(x.shape) == 5 and dot_vector: # vector quantity
        # This is essentially dx**2 below summed along the component axis
        dx2 = dot_prod(dx, dx)
    else:  # scalar quantity or rms of individual vector components
        dx2 = dx**2
    return np.sqrt(box_avg(dx2))


# --- VECTOR FUNCTIONS --- #


def get_mag(x, squared=0, axis=1, reshape=0):
    '''For an array of vectors with the same number of components,
    returns the magnitude of each vector in an array of the same size.
    
    E.g. x = [[1, 0, 1],\n
              [3, 4, 0],\n
              [1, 1, 1]]
         get_mag(x) = [√2, 5, √3]
    '''
    mag2 = dot_prod(x, x, axis=axis, reshape=reshape)
    return mag2 if squared else np.sqrt(mag2)


def get_unit(x):
    '''Calculates unit vector.'''
    len_shape = len(x.shape)
    axis = len_shape - 4
    
    x_mag = get_mag(x, reshape=1, axis=axis)
    return x / x_mag


def get_vec(x, ps):
    '''Returns array of vectors of quantity x at a given array
    of 3D grid points ps.
    '''
    indices = tuple([ps[:, 0], ps[:, 1], ps[:, 2]])
    x1 = x[0][indices]
    x2 = x[1][indices]
    x3 = x[2][indices]
    return np.array((x1, x2, x3)).T


def get_lengths(data_load=1, data=None, output_dir=None, prob=DEFAULT_PROB, zyx=0):
    if data_load:
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
    X1, X2, X3 = get_lengths(output_dir=output_dir, prob=prob)
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


def ft_grid(input_type, data=None, output_dir='', Ls=None,
            Ns=None, prob=DEFAULT_PROB, make_iso_box=1):
    '''
    Creates a grid in k-space corresponding to the real grid given in data.
    k_grid is a boolean that when True calculates a regularly spaced array
    in k-space. Isotropic-to-box by default.
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
        Ls = get_lengths(output_dir=output_dir, prob=prob, zyx=1)  # box side lengths
        Ns = get_rootgrid(output_dir, prob, zyx=1) # number of grid points

    elif input_type == 'array':
        assert (Ls is not None and Ns is not None), 'Must have valid lengths and grid information!'

    else:
        raise ValueError('Please enter a valid input type')

    # k=0 ⟺ Z, 1 ⟺ Y, 2 ⟺ X
    K = {
        k: 2j * np.pi * ft_array(Ns[k])
        if make_iso_box
        else 2j * np.pi / Ls[k] * ft_array(Ns[k])
        for k in range(3)
    }

    return np.meshgrid(K[0], K[1], K[2], indexing='ij')


# --- MHD TURBULENCE DIAGNOSTICS --- #

def alfven_speed(rho, B):
    # generalizing the definition of Alfvén speed
    # for aribitraty mean fields
    B_0 = box_avg(B)  # mean field
    
    # choose the component axis
    # axis = 0 for single time entry
    # axis = 1 for multiple time entries
    axis = len(B_0.shape) - 1
    B0_mag = get_mag(B_0, axis=axis)
    return B0_mag / np.sqrt(box_avg(rho))
    

def cross_helicity(rho, u_perp, B_perp): 
    udotB = dot_prod(u_perp, B_perp)
    u2, B2 = dot_prod(u_perp, u_perp), dot_prod(B_perp, B_perp)

    return 2 * box_avg(np.sqrt(rho) * udotB) / box_avg(rho*u2 + B2)


def z_waves_evo(rho, u_perp, B_perp, v_A):
    # magnetic field in velocity units
    b_perp = B_perp / np.sqrt(rho)
    z_p = u_perp + b_perp
    z_m = u_perp - b_perp
    z_p_rms, z_m_rms = rms(z_p) / v_A, rms(z_m) / v_A
    return z_p_rms, z_m_rms

def beta(rho, B_mag2, cs_init, a):
    # rho and B_mag should be of the form rho[time, x, y, z]
    # from Squire2020, line before Eq. 4
    # assuming isothermal eos
    sound_speed = cs_init * a**(-2/3)
    rhoB2_avg = box_avg(rho / B_mag2)
    return 2 * sound_speed**2 * rhoB2_avg

def mag_compress_Squire2020(B):
    # δ(B^2) / (δB_vec)^2
    B_mag2 = get_mag(B)**2
    rms_Bmag2 = rms(B_mag2)
    sqr_rmsB = rms(B)**2
    return rms_Bmag2 / sqr_rmsB

def mag_compress_Shoda2021(B):
    # (δB)^2 / (δB_vec)^2
    B_mag = get_mag(B)
    sqr_rmsBmag = rms(B_mag)**2
    sqr_rmsB = rms(B)**2
    return sqr_rmsBmag / sqr_rmsB

def norm_fluc_amp(fluc2, background):
    # calculated from vectors
    # quantities of the form <B_⟂^2> / <B_x>**2
    # or <ρu_⟂^2> / <B_x>**2
    mean_fluc2 = box_avg(fluc2)
    sqr_mean_bg = box_avg(background)**2
    return mean_fluc2 / sqr_mean_bg

def norm_fluc_amp_hst(output_dir, adot, B_0, Lx=1., Lperp=1., method='matt', prob=DEFAULT_PROB):
    # calculated from energies
    a, Bprp_fluc, uprp_fluc = energy.get_fluc_energy(output_dir, adot, B_0, Lx=Lx, Lperp=Lperp, prob=prob, method=method)
    return a, Bprp_fluc, uprp_fluc

# --- EXPANDING BOX CODE --- #

def switchback_threshold(B, theta_threshold=30, flyby=0):
    # finding magnetic field reversals with an deviation greater
    # than theta_threshold from the mean magnetic field/Parker spiral
    # theta_threshold is input in degrees
    theta_threshold *= np.pi / 180

    if flyby:
        Bx, By, Bmag = B
        # calculate either deviation from radial or Parker
        B0x, B0y = (Bx.mean(), By.mean())
        B0 = np.sqrt(B0x**2 + B0y**2)
        B_dot_Bmean = (Bx*B0x+By*B0y) / (Bmag*B0)
    else:
        Bx = B[:, 0]
        B_0 = box_avg(B, reshape=1) # mean field in box = Parker spiral
        b, b_0 = get_unit(B), get_unit(B_0)
        B_dot_Bmean = dot_prod(b, b_0, 1)

    N_cells = Bx[0].size
    dev_from_mean = np.arccos(np.clip(B_dot_Bmean, -1., 1.))
    SB_radial_flip = Bx <= 0.
    SB_dev = dev_from_mean >= theta_threshold
    SB_dev_radial_flip = SB_dev & SB_radial_flip
    # fraction of radial flips in box: number of cells with SBs / total cells in box
    if flyby:
        dev_SB_frac = SB_dev[SB_dev].size / N_cells  # number of SBs deviating from mean field
        radial_SB_frac = SB_radial_flip[SB_radial_flip].size / N_cells  # number of SBs with flipped field
        dev_radial_SB_frac = SB_dev_radial_flip[SB_dev_radial_flip].size / N_cells  # number of SBs dev from mean with radial flip
    else:
        dev_SB_frac = np.array([SB_dev[n][SB_dev[n]].size / N_cells for n in range(B.shape[0])])
        radial_SB_frac = np.array([SB_radial_flip[n][SB_radial_flip[n]].size / N_cells for n in range(B.shape[0])])
        dev_radial_SB_frac = np.array([SB_dev_radial_flip[n][SB_dev_radial_flip[n]].size / N_cells for n in range(B.shape[0])])
    
    return SB_dev, SB_dev_radial_flip, dev_SB_frac, radial_SB_frac, dev_radial_SB_frac

def label_switchbacks(SB_mask, array3D=1):
    labels, nlabels = ndimage.label(SB_mask)
    
    if array3D:
        # ensuring switchbacks are joined if straddling
        # periodic boundaries
        x_boundary = (labels[:, :, :, 0] > 0) & (labels[:, :, :, -1] > 0)
        y_boundary = (labels[:, :, 0, :] > 0) & (labels[:, :, -1, :] > 0)
        z_boundary = (labels[:, 0, :, :] > 0) & (labels[:, -1, :, :] > 0)
        labels[:,:,:,-1][x_boundary] = labels[:,:,:,0][x_boundary]
        labels[:,:,-1,:][y_boundary] = labels[:,:,0,:][y_boundary]
        labels[:,-1,:,:][z_boundary] = labels[:,0,:,:][z_boundary]
        
        # update number of switchbacks
        label_array = np.unique(labels[labels > 0])
        nlabels = label_array.size
    return labels, nlabels, label_array

def switchback_finder(B, SB_mask, array3D=1):
    # label each individual switchback
    # and find the position of these switchbacks
    # treating the mask as a 3D "image"
    
    Bx = B[:,0] if array3D else B[0]
    By = B[:,1] if array3D else B[1]
    Bz = B[:,2] if array3D else B[2]
    
    labels, nlabels, label_array = label_switchbacks(SB_mask, array_3D=array3D)
    
    # collect all switchbacks into a dictionary
    SBs = {
        'n_SBs': nlabels
    }

    i = 0
    for label_i in label_array:
        # find all the points where switchbacks are
        # only considering a collection of points greater than 10
        points = np.where(labels == label_i)
        if points[0].shape[0] <= 10:
            SBs['n_SBs'] -= 1
            continue
        SBs[i] = np.array((Bx[points], By[points], Bz[points]))
        i += 1
    
    return SBs
    
def switchback_aspect(SB_mask, Ls, Ns):
    # label each individual switchback
    # and find the position of these switchbacks
    # treating the mask as a 3D "image"
    labels, nlabels, label_array = label_switchbacks(SB_mask)
    
    if nlabels == 0:
        return 0.
    
    pcas = {}
    dx = Ls / Ns  # dz, dy, dx
    sb_index = 0
    for label_i in label_array:
        # get the points where the switchback resides
        points = np.array(np.where(labels[0]==label_i), dtype='float').T
        if points.shape[0] <= 10:
            continue
        points *= dx  # get real coordinates, in order to calculate lengths
        # if the points cut across periodic boundaries, 
        # shift until they form a cohesive whole
        for i in range(3):
            if ((abs(points[:, i]) < Ls[i]/10).any()):
                points[:, i] = np.mod(points[:, i] + Ls[i]/4, Ls[i])
            if ((abs(Ls[i] - points[:, i]) < Ls[i]/10).any()):
                points[:, i] = np.mod(points[:, i] - Ls[i]/4, Ls[i])
        
        # perform a PCA on the switchback lengths
        # this gives three vectors along which the
        # length varies the most to the least
        pca = PCA(n_components=3, whiten=True)
        pca.fit(points)
        # convention: switchback is aligned along direction
        # with highest variance in length
        # length along vector ~ 4*sqrt(variance)
        V = pca.components_  # unit vectors sorted by variance
        V_length = 4*np.sqrt(pca.explained_variance_)
        pcas[sb_index] = {
            'unit_vectors': V[:, ::-1],  # sorting x, y, z components
            'lengths': V_length
        }
        sb_index += 1
   
    return pcas

def clock_angle(B, SB_mask, mean_switchback=1, flyby=0):
    if flyby:
        Bx, By, Bz = B
        B0x, B0y = Bx.mean(), By.mean()  # Parker spiral
        parker_angle = np.arctan(B0y/B0x)
        B0 = np.sqrt(B0x**2 + B0y**2)
        b0x, b0y = B0x/B0, B0y/B0
        Bprl = Bx*b0x + By*b0y
        Bprpx = Bx - Bprl*b0x
        Bprpy = By - Bprl*b0y
        B_prp = (Bprpx, Bprpy, Bz)
    else:
        B_0 = box_avg(B) # mean field in box = Parker spiral
        comp_index = len(B_0.shape) - 1
        B0x, B0y = np.take(B_0, 0, comp_index), np.take(B_0, 1, comp_index)
        parker_angle = np.arctan(B0y / B0x)
        b_0 = get_unit(box_avg(B, reshape=1))
        
        # get magnetic field vectors perpendicular
        # to the mean field (this is in the rotated TN-plane)
        B_prp = B - dot_prod(B, b_0, reshape=1)*b_0
    ca_bins = np.linspace(-np.pi, np.pi, 51)
    ca_grid = 0.5*(ca_bins[1:] + ca_bins[:-1])
    
    if mean_switchback:
        # find the perpendicular components of switchbacks
        # I'm assuming that projection and averaging commute
        SBs = switchback_finder(B_prp, SB_mask, array3D=(not flyby))
        
        clock_angle = []
        for n in range(SBs['n_SBs']):
            # find the mean vector over the entire switchback
            SB_n = SBs[n].mean(axis=1)
            # decompose vector into N and T components
            # T unit vector is -y_hat rotated by Parker angle
            B_N = SB_n[2] # +N <-> +z in box
            B_T = SB_n[0]*np.sin(parker_angle) - SB_n[1]*np.cos(parker_angle)
            B_prp_mag = get_mag(SB_n, axis=0)
            # clock angle is measured clockwise from N axis (z axis in box)
            # 0 = +N (+z), 90 = +T (-y), 180 = -N (-z), 270/-90 = -T (+y)
            clock_angle.append(np.arccos(B_N / B_prp_mag) if B_T >= 0. else -np.arccos(B_N / B_prp_mag))
        ca = np.histogram(clock_angle, ca_bins)[0]
    else:
        shape_tup = (B_0.shape[0],1,1,1) if comp_index == 1 else (1,1,1)
        # unit vector in T direction in TN-plane rotated to
        # be perpendicular to mean field
        t_prime_x =  np.sin(parker_angle)*np.ones(shape=shape_tup)
        t_prime_y = -np.cos(parker_angle)*np.ones(shape=shape_tup)

        B_N = np.take(B_prp, 2, comp_index)  # +N <-> +z in box
        B_T = np.take(B_prp, 0, comp_index) * t_prime_x + np.take(B_prp, 1, comp_index) * t_prime_y
        B_prp_mag = get_mag(B_prp, axis=comp_index)
        angle = np.arccos(B_N / B_prp_mag)
        # clock angle is measured clockwise from N axis (z axis in box)
        # 0 = +N (+z), 90 = +T (-y), 180 = -N (-z), 270/-90 = -T (+y)
        clock_angle = np.where(B_T >= 0., angle, -angle)
        
        ca = np.histogram(clock_angle[SB_mask], ca_bins)[0]

    return {
        'clock_angle_count': ca,
        'bins': ca_bins,
        'grid': ca_grid
    }
    
def mean_cos2(b_0, B_prp, a, output_dir):
    # part of diagnostic used in Mallet2021
    # assumes time series
    
    # load in the grid at the first snapshot
    # this won't change in the comobile frame
    KZ, KY, KX = ft_grid('output',output_dir=output_dir, make_iso_box=0)
    # add in a evolution to perpendicular lengths
    K_temp = np.array([KX, KY, KZ]).reshape(1,3,*KX.shape)
    K = K_temp / a.reshape(a.shape[0],1,1,1,1)
    K[:, 0] *= a.reshape(a.shape[0],1,1,1)

    # Decomposing k parallel and perpendicular to mean field
    Kprl = dot_prod(K, b_0)
    Kprp = K - dot_prod(K, b_0, reshape=1)*b_0
    Kprl = abs(Kprl)
    Kprp = get_mag(abs(Kprp))
    Kmag = np.maximum(np.sqrt(Kprl**2 + Kprp**2), 1e-15)
    
    # cosine squared of angle between k and B_0
    cos2_theta = (Kprl / Kmag)**2
    
    # take the FFT of B_prp over the box
    len_shape = len(B_prp.shape)
    box_axes = np.arange(len_shape-3, len_shape)
    prp_fft = np.fft.fftn(B_prp, axes=box_axes)
    
    # Energy of fluctuations is mod^2 of FFT of fluctuations
    # Have to normalize by the number of points in the box
    N_points = np.array(prp_fft.shape[-3:]).prod()
    energy_vec = 0.5*abs(prp_fft)**2 / N_points
    
    # Total energy of fluctuations is the sum 
    # of the energy in each component
    energy = energy_vec.sum(axis=1)
    
    cos2_box = box_avg(abs(K[:, 0])**2 / np.maximum((abs(K[:, 0])**2 + abs(K[:, 1])**2 + abs(K[:, 2])**2),1e-15))  # mean box-allowed wave vector rotation due to expansion
    cos2_field = box_avg(cos2_theta)  # mean box-allowed wave vector rotation relative to mean field
    not2D_mode_mask = abs(Kprl) != 0.0
    cos2_energyweight = box_avg(cos2_theta * energy) / box_avg(energy)
    cos2_energyweight_no2D = np.mean(cos2_theta[not2D_mode_mask] * energy[not2D_mode_mask]) / np.mean(energy[not2D_mode_mask])
    
    # Weight the cosine2 at each point in the box by the energy at that point
    # and then average and normalize by the mean energy
    return cos2_box, cos2_field, cos2_energyweight, cos2_energyweight_no2D

def plot_dropouts(flyby):
    # performing analysis as in Farrell
    dl = 4  # units of resolution
    dBr = -flyby['Bx'][dl:] + flyby['Bx'][:-dl]
    dur = -flyby['ux'][dl:] + flyby['ux'][:-dl]
    l = flyby['l_param'][:,0][dl//2:-dl//2]
    Br = -flyby['Bx'][dl//2:-dl//2]
    Bt = -flyby['By'][dl//2:-dl//2]
    Bn =  flyby['Bz'][dl//2:-dl//2]
    ur = -flyby['ux'][dl//2:-dl//2]
    ut = -flyby['uy'][dl//2:-dl//2]
    un =  flyby['uz'][dl//2:-dl//2]
    unonr = np.sqrt(ut**2 + un**2)
    rho = flyby['rho'][dl//2:-dl//2]
    Bmag = flyby['Bmag'][dl//2:-dl//2]

    # switchback index
    SBI = dBr*dur
    # only look at places with SBIs above sb_cut
    # this is what is used in Farrell
    sb_cut = 0.75
    # how far to look on either side of the switchback boundary
    sbsz = 30

    # find the indicies where these occur
    sbi, sbh = find_peaks(SBI, height=sb_cut)
    nsbs = sbi.size

    # step up <=> sign > 0
    # step down <=> sign < 0
    upordown = np.sign(dBr[sbi])

    sbBr, sbBt, sbBn = np.zeros(shape=(3, nsbs, 2*sbsz))
    sbur, sbut, sbun, sbunonr = np.zeros(shape=(4, nsbs, 2*sbsz))
    sbrho, sbBmag = np.zeros(shape=(2, nsbs, 2*sbsz))
    for i in range(nsbs):
        sbBr[i] = Br[sbi[i]-sbsz:sbi[i]+sbsz]
        sbBt[i] = Bt[sbi[i]-sbsz:sbi[i]+sbsz]
        sbBn[i] = Bn[sbi[i]-sbsz:sbi[i]+sbsz]
        sbur[i] = ur[sbi[i]-sbsz:sbi[i]+sbsz]
        sbut[i] = ut[sbi[i]-sbsz:sbi[i]+sbsz]
        sbun[i] = un[sbi[i]-sbsz:sbi[i]+sbsz]
        sbunonr[i] = unonr[sbi[i]-sbsz:sbi[i]+sbsz]
        sbrho[i] = rho[sbi[i]-sbsz:sbi[i]+sbsz]
        sbBmag[i] = Bmag[sbi[i]-sbsz:sbi[i]+sbsz]

    dropouts = {'step_up':{}, 'step_down': {}}
    for step in dropouts:
        mask = upordown > 0 if step == 'step_up' else upordown < 0
        dropouts[step]['Br'] = sbBr[mask].mean(axis=0)
        dropouts[step]['Bt'] = sbBt[mask].mean(axis=0)
        dropouts[step]['Bn'] = sbBn[mask].mean(axis=0)
        dropouts[step]['ur'] = sbur[mask].mean(axis=0)
        dropouts[step]['ut'] = sbut[mask].mean(axis=0)
        dropouts[step]['un'] = sbun[mask].mean(axis=0)
        dropouts[step]['unonr'] = sbunonr[mask].mean(axis=0)
        dropouts[step]['rho'] = sbrho[mask].mean(axis=0)
        dropouts[step]['Bmag'] = sbBmag[mask].mean(axis=0)

    dropouts['l_sb'] = l[round(l.size)//2 - sbsz:round(l.size)//2 + sbsz]
    return dropouts
    
    
def Bfluc_Mallet2021(adot, cos2_theta, beta, g):
    # Equation 88 from paper, assuming this is equivalent
    # to magnetic compressibility
    # assuming g is the normalized fluctuation amplitude

    return 0