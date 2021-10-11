'''
Code to load in plasma simulations run using Athena++ and calculate diagnostics.
All copied from code written in 2020 for my Honours project with some improvements.
'''
import glob
import os
import pickle
import h5py
import numpy as np
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
            # grid_x2_f = f['x2f'][...]
            # grid_x2_v = f['x2v'][...]
            # grid_x3_f = f['x3f'][...]
            # grid_x3_v = f['x3v'][...]
            if undo:
                prim[0] *= λ if matts_method else 1
                prim[1] /= Λ[0]
                prim[2] /= Λ[1]
                prim[3] /= Λ[2]
                B[0] /= Λ[0] / λ if matts_method else Λ[0]
                B[1] /= Λ[1] / λ if matts_method else Λ[1]
                B[2] /= Λ[2] / λ if matts_method else Λ[2]
                # grid_x2_f /= a
                # grid_x2_v /= a
                # grid_x3_f /= a
                # grid_x3_v /= a
            else:
                prim[0] /= λ if matts_method else 1
                prim[1] *= Λ[0]
                prim[2] *= Λ[1]
                prim[3] *= Λ[2]
                B[0] *= Λ[0] / λ if matts_method else Λ[0]
                B[1] *= Λ[1] / λ if matts_method else Λ[1]
                B[2] *= Λ[2] / λ if matts_method else Λ[2]
                # grid_x2_f *= a
                # grid_x2_v *= a
                # grid_x3_f *= a
                # grid_x3_v *= a
            
            f['prim'][...] = prim
            f['B'][...] = B
            # f['x2f'][...] = grid_x2_f
            # f['x2v'][...] = grid_x2_v
            # f['x3f'][...] = grid_x3_f
            # f['x3v'][...] = grid_x3_v
            
            
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
        shape = x.shape
        # add back in grid columns for broadcasting
        avg = avg.reshape(*shape[:2], 1, 1, 1)
    return avg

def dot_prod(x, y, axis=1, reshape=0):
    x_dot_y = np.sum(x*y, axis=axis)
    if reshape:
        shape = x.shape
        # add back in component column for broadcasting
        x_dot_y = x_dot_y.reshape(shape[0], 1, *shape[2:])
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


def get_mag(x, reshape=0):
    '''For an array of vectors with the same number of components,
    returns the magnitude of each vector in an array of the same size.
    
    E.g. x = [[1, 0, 1],\n
              [3, 4, 0],\n
              [1, 1, 1]]
         get_mag(x) = [√2, 5, √3]
    '''
    axis = 1 if len(x.shape) == 5 else 0
    return np.sqrt(dot_prod(x, x, axis=axis, reshape=reshape))


def get_unit(x):
    '''Calculates unit vector.'''
    x_mag = get_mag(x, reshape=1) if len(x.shape) == 5 else get_mag(x)
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


def ft_grid(input_type, data=None, output_dir=None, Ls=None,
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
        Ls = get_lengths(output_dir, prob, zyx=1)  # box side lengths
        Ns = get_rootgrid(output_dir, prob, zyx=1) # number of grid points

    elif input_type == 'array':
        assert (Ls is not None and Ns is not None), 'Must have valid lengths and grid information!'

    else:
        raise ValueError('Please enter a valid input type')

    # Corresponds to Athena++ standard k=0 ⟺ Z, 1 ⟺ Y, 2 ⟺ X
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
    B0_mag = get_mag(B_0)
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

def norm_fluc_amp_hst(output_dir, adot, B_0, method='matt', prob=DEFAULT_PROB):
    # calculated from energies
    a, Bprp_fluc, uprp_fluc = energy.get_fluc_energy(output_dir, adot, B_0, prob=prob, method=method)
    return a, Bprp_fluc, uprp_fluc

# --- EXPANDING BOX CODE --- #

def switchback_finder(B, theta_threshold=90):
    # finding magnetic field reversals with an deviation greater
    # than theta_threshold from the mean magnetic field/Parker spiral
    # theta_threshold is input in degrees

    theta_threshold *= np.pi / 180
    N_cells = B[0, 0].size  # number of cells in the box
    B_0 = box_avg(B, reshape=1) # mean field in box = Parker spiral
    b, b_0 = get_unit(B), get_unit(B_0)
    dev_from_mean = np.arccos(np.clip(dot_prod(b, b_0, 1), -1., 1.))
    SB_mask = dev_from_mean >= theta_threshold
    # fraction of SBs in box: number of cells with SBs / total cells in box
    SB_frac = np.array([B[n, 0][SB_mask[n]].size / N_cells for n in range(B.shape[0])])
    return SB_mask, SB_frac



