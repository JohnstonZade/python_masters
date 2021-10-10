# --- IMPORTS --- #
import h5py
import numpy as np

import diagnostics as diag
import generate_spectrum as genspec
import reinterpolate
from helper_functions import *
from project_paths import from_array_path


# --- GENERATING FUNCTIONS --- #

def create_athena_fromics(folder, h5name, n_X, X_min, X_max, meshblock,
                          const_b2=0, do_norm_energy=1, energy=1., time_lim=1, dt=0.2, iso_sound_speed=1.0, expand=0, exp_rate=0., 
                          athinput=from_array_path, set_ics_externally=0, ICs=None):
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
                             h5name, time_lim=time_lim, dt=dt, iso_sound_speed=iso_sound_speed,
                             expand=expand, exp_rate=exp_rate)
    h5name = folder + h5name  # eg 'ICs_template.h5'
    N_HYDRO = 4  # number of hydro variables (e.g. density and momentum); assuming isothermal here
    # Dimension setting: 1D if only x has more than one gridpoint
    one_D = 1 if np.all(n_X[1:] == 1) else 0

    if set_ics_externally:
        assert ICs is not None, 'Please input valid ICs!'
        Dnf, UXf, UYf, UZf, BXf, BYf, BZf = ICs
    else:
        # User set initial conditions
        # Density
        Dnf = lambda X, Y, Z: np.ones(X.shape)
        # Velocity components
        UXf = lambda X, Y, Z: np.zeros(X.shape)
        UYf = lambda X, Y, Z: np.zeros(X.shape)
        UZf = lambda X, Y, Z: np.sin((2*np.pi/X_max[0]) * X + (2*np.pi/X_max[1]) * Y)
        # Magnetic components
        BXf = lambda X, Y, Z: np.ones(X.shape)
        BYf = lambda X, Y, Z: np.zeros(X.shape)
        BZf = lambda X, Y, Z: np.sin((2*np.pi/X_max[0]) * X + (2*np.pi/X_max[1]) * Y)

    # --- GRID CREATION --- #

    X_grid, (dx, dy, dz) = generate_grid(X_min, X_max, n_X)
    Hy_grid, BXcc, BYcc, BZcc = setup_hydro_grid(n_X, X_grid, N_HYDRO, Dnf, UXf, UYf, UZf, BXf, BYf, BZf)

    if not const_b2 and do_norm_energy:
        # initializing Alfvén wave fluctuations perpendicular to B_0 (assumed along x-axis)
        # to have same initial energy in velocity and magnetic fields.
        total_energy = 0.5*np.mean(BYcc**2 + BZcc**2)
        norm_energy = np.sqrt(energy / total_energy)
        Hy_grid[2] *= norm_energy
        Hy_grid[3] *= norm_energy
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
    if const_b2:
        constB2_faceinterp(BXcc, BYcc, BZcc, h5name, n_X, X_min, X_max, meshblock, n_blocks, blocks)
    else:
        calc_and_save_B(BXcc, BYcc, BZcc, h5name, n_X, X_min, X_max, meshblock, n_blocks, blocks, dx, dy, dz)
    print('Magnetic Saved Successfully')
    print('Done!')

def create_athena_fromh5(save_folder, athinput_in_folder, athinput_in, h5name, athdf_input,
                         time_lim, dt, beta_multiplier, expand=1, exp_rate=0.5, athinput_out=from_array_path):
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
    
    save_folder = diag.format_path(save_folder)
    h5name = save_folder + h5name
    h5name += '.h5' if '.h5' not in h5name else ''
    remove_prev_h5file(h5name)
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
    
    n_X, X_min, X_max, meshblock, dt_hst, dt, expand_old, exp_rate_old, iso_sound_speed = read_athinput(athinput_in, reinterpolate=1)
    athinput_out = edit_athinput(athinput_out, save_folder, n_X, X_min, X_max, meshblock,
                                 h5name, time_lim=time_lim, dt=dt, iso_sound_speed=np.sqrt(beta_multiplier)*iso_sound_speed,
                                 expand=expand, exp_rate=exp_rate)

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

def create_athena_alfvenspec(folder, h5name, n_X, X_min, X_max, meshblock, athinput=from_array_path,
                             time_lim=1, dt=0.2, expand=0, exp_rate=0., iso_sound_speed=1.0,
                             perp_energy=0.5, spectrum='isotropic', expo=-5/3, expo_prl=-2., kpeak=(2,2), kwidth=12.0,
                             do_truncation=0, n_cutoff=None, do_mode_test=0, do_parker=0, final_bybx_ratio=1.5):
    
    folder = diag.format_path(folder)
    ath_copy = edit_athinput(athinput, folder, n_X, X_min, X_max, meshblock,
                             h5name, time_lim, dt, iso_sound_speed, expand, exp_rate)
    h5name = folder + h5name  # eg 'ICs_template.h5'                             
    N_HYDRO = 4  # number of hydro variables (e.g. density and momentum); assuming isothermal here
    # Dimension setting: 1D if only x has more than one gridpoint
    one_D = 1 if np.all(n_X[1:] == 1) else 0

    B0_x = 1.0  # mean Bx
    B0_y = 0.0  # mean By
    if do_parker:
        a_f = 1 + exp_rate*time_lim
        initial_bybx_ratio = final_bybx_ratio / a_f
        # we want B0_mag = 1.0 always
        B0_x /= np.sqrt(1.0 + initial_bybx_ratio**2)
        B0_y = initial_bybx_ratio*B0_x
    
    # Generate mean fields
    # Density
    Dnf = lambda X, Y, Z: np.ones(X.shape)
    UXf = lambda X, Y, Z: np.zeros(X.shape)
    UYf = lambda X, Y, Z: np.zeros(X.shape)
    UZf = lambda X, Y, Z: np.zeros(X.shape)
    BXf = lambda X, Y, Z: B0_x*np.ones(X.shape)
    BYf = lambda X, Y, Z: B0_y*np.ones(X.shape)
    BZf = lambda X, Y, Z: np.zeros(X.shape)

    X_grid, (dx, dy, dz) = generate_grid(X_min, X_max, n_X)
    Hy_grid, BXcc, BYcc, BZcc = setup_hydro_grid(n_X, X_grid, N_HYDRO, Dnf, UXf, UYf, UZf, BXf, BYf, BZf)
    B0 = np.array([BXcc, BYcc, BZcc])  # mean field
    
    X_grid = None

    if do_mode_test:
        # Generate a single mode for testing
        dB_x, dB_y, dB_z = genspec.generate_alfven_spectrum(n_X, X_min, X_max, B0, spectrum,
                                                      run_test=True)
    elif spectrum == 'gaussian': 
        dB_x, dB_y, dB_z = genspec.generate_alfven_spectrum(n_X, X_min, X_max, B0,
                                                      spectrum, kpeak=kpeak, kwidth=kwidth, 
                                                      do_truncation=do_truncation, n_cutoff=n_cutoff)
    else:
        # Generate isotropic or GS spectrum
        dB_x, dB_y, dB_z = genspec.generate_alfven_spectrum(n_X, X_min, X_max, B0, spectrum,
                                                      expo=expo, expo_prl=expo_prl,
                                                      do_truncation=do_truncation, n_cutoff=n_cutoff,)
    
    # Setting z^- waves = 0
    rho = Hy_grid[0] 
    # total volume weighted energy = sum(0.5*dV*B^2) = 0.5*(V/N)sum(B^2) = 0.5*V*mean(B^2)
    total_perp_energy = 0.5*np.mean(dB_x**2 + dB_y**2 + dB_z**2)
    norm_perp_energy = np.sqrt(perp_energy / total_perp_energy)
    
    du_x, du_y, du_z = dB_x / np.sqrt(rho), dB_y / np.sqrt(rho), dB_z / np.sqrt(rho)

    Hy_grid[1] += rho*norm_perp_energy*du_x
    Hy_grid[2] += rho*norm_perp_energy*du_y
    Hy_grid[3] += rho*norm_perp_energy*du_z

    BXcc += norm_perp_energy*dB_x
    BYcc += norm_perp_energy*dB_y
    BZcc += norm_perp_energy*dB_z
    dB_x, dB_y, dB_z, du_x, du_y, du_z = None, None, None, None, None, None

    # --- MESHBLOCK STRUCTURE --- #

    n_blocks, blocks = make_meshblocks(folder, ath_copy, n_X, meshblock, one_D)

    # --- SAVING VARIABLES --- #
    
    remove_prev_h5file(h5name)

    # - MAGNETIC
    calc_and_save_B(BXcc, BYcc, BZcc, h5name, n_X, X_min, X_max, meshblock, n_blocks, blocks, dx, dy, dz)
    print('Magnetic Saved Successfully')
    BXcc, BYcc, BZcc = None, None, None
    dx, dy, dz = None, None, None

    # - HYDRO
    save_hydro_grid(h5name, Hy_grid, N_HYDRO, n_blocks, blocks, meshblock, remove_h5=0)
    print('Hydro Saved Succesfully')

def reinterp_from_h5(save_folder, athinput_in_folder, athinput_in, h5name, athdf_input, athinput_out=from_array_path,
                     a_finish=8, a_re=2, new_meshblock=None, rescale_prl=1, method='matt'):
    N_HYDRO = 4
    def root_path(path):
        return ''.join(sub + '/' for sub in path.split('/')[:-1])
    
    if athinput_in_folder not in athinput_in:
        athinput_in = athinput_in_folder + athinput_in

    # From the final athdf file of the lower resolution simulation:
    # - Get final time and expansion value
    # - Get the old grid as well, as we need this for the reinterpolation
    n_f = diag.get_maxn(root_path(athdf_input), do_path_format=0) - 1
    athdf_data = diag.load_data(root_path(athdf_input), n_f, do_path_format=0, method=method)
    t_f = athdf_data['Time']
    old_Xgrid = athdf_data['x1v'], athdf_data['x2v'], athdf_data['x3v']
    athdf_data = None

    # From the initial athinput file:
    # - Get the old resolution and meshblock
    # - Get the box boundaries as well as the .hst dt (needed for next output in new athinput file)
    old_Ns, X_min, X_max, meshblock, dt_hst, dt, expand, exp_rate, iso_sound_speed = read_athinput(athinput_in, reinterpolate=1)

    # Load in the hydro data (depending on whether this is in conserved or primitive form)
    f_athdf = h5py.File(athdf_input, 'r')
    if 'cons' in list(f_athdf.keys()):  # conserved variables if 1, primitive if 0
        Hy_grid = np.array(f_athdf['cons']) 
    else:
        Hy_grid = np.copy(f_athdf['prim'])  # don't want to modify the original data set
        Hy_grid[1:] *= Hy_grid[0]  # getting momentum variables (vel * density)
    B_grid = f_athdf['B']

    # Get meshblock information from initial low resolution simulations
    n_blocks, blocks = generate_mesh_structure(athinput_in_folder, athinput_in)
    Hydro_unpacked = np.zeros(shape=(N_HYDRO, *old_Ns[::-1]))
    B_unpacked = np.zeros(shape=(3, *old_Ns[::-1]))

    # unpack from meshblocks
    for i in range(N_HYDRO):
        for m in range(n_blocks):
                off = blocks[:, m]
                ind_s = (meshblock*off)[::-1]
                ind_e = (meshblock*off + meshblock)[::-1]
                Hydro_unpacked[i, ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]] = Hy_grid[i, m, :, :, :]

    for b in range(3):
        for m in range(n_blocks):
                off = blocks[:, m]
                ind_s = (meshblock*off)[::-1]
                ind_e = (meshblock*off + meshblock)[::-1]
                B_unpacked[b, ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]] = B_grid[b, m, :, :, :]
    
    # Rescale the resolution in the ⟂ (y, z) directions
    # In (X, Y, Z) format
    new_Ns, Ls = np.copy(old_Ns), (X_max - X_min)
    if rescale_prl:
        new_Ns[0] = int(new_Ns[0] // a_re)
    else: 
        new_Ns[1:] = new_Ns[1:] * a_re
    
    
    if new_meshblock is not None:
        meshblock = new_meshblock
    else:
        if rescale_prl:
            meshblock[0] //= a_re
        else:
            meshblock[1:] *= a_re  # rescale meshblocks too (this is in X, Y, Z format)
    dx, dy, dz = generate_grid(X_min, X_max, new_Ns)[1]

    # Reinterpolate the data to the new high resolution grid
    Hydro_hires = np.zeros(shape=(N_HYDRO, *new_Ns[::-1]))
    B_hires = np.zeros(shape=(3, *new_Ns[::-1]))
    for i in range(4):
        Hydro_hires[i] = reinterpolate.reinterp_to_grid(Hydro_unpacked[i], old_Xgrid, new_Ns[::-1], Ls[::-1])
    for b in range(3):
        B_hires[b] = reinterpolate.reinterp_to_grid(B_unpacked[b], old_Xgrid, new_Ns[::-1], Ls[::-1])
    BXcc, BYcc, BZcc = B_hires

    
    h5name += '.h5' if '.h5' not in h5name else ''
    n_hst, n_hdf5 = int(np.ceil(t_f / dt_hst)), int(np.ceil(t_f / dt))
    start_time = t_f
    time_lim  = (a_finish - 1) / exp_rate 
    athinput_out = edit_athinput(athinput_out, save_folder, new_Ns, X_min, X_max, meshblock, h5name,
                                 time_lim, dt, iso_sound_speed, expand, exp_rate,
                                 start_time=start_time, n_hst=n_hst, n_hdf5=n_hdf5)

    h5name = save_folder + h5name
    remove_prev_h5file(h5name)
    
    n_blocks, blocks = generate_mesh_structure(save_folder, athinput_out)
    save_hydro_grid(h5name, Hydro_hires, N_HYDRO, n_blocks, blocks, meshblock)
    print('Hydro Saved Succesfully')

    calc_and_save_B(BXcc, BYcc, BZcc, h5name, new_Ns, X_min, X_max, meshblock, n_blocks, blocks, dx, dy, dz)
    print('Magnetic Saved Successfully')
    print('Done!')

    return athinput_out


