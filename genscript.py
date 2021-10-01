import numpy as np
import os
import diagnostics as diag
import generate_ics as genics
import spectrum as spec
import project_paths as paths

folder_root, athena_path = paths.PATH, paths.athena_path

def expand_to_a(a_final, exp_rate):
    # inverting definition of a to find time limit
    return (a_final - 1) / exp_rate

def wave_energy(wave_amp):
    return wave_amp**2 / 4

def cs_from_beta(init_norm_fluc, beta):
    return np.sqrt(0.5*(1 + init_norm_fluc) * beta)


def generate(sim_name, folder, box_aspect, cell_aspect, Nx_init, n_cpus, exp_rate, 
             dt, init_norm_fluc, beta, tlim=2., expand=1, expo=-5/3, expo_prl=-2, choose_res=0, N_prp=200,
             spectrum='isotropic', κ_prl=2, κ_prp=2, a_end=10, gen_ic=1, run_athena=0, run_spec=0):
    total_folder =  diag.format_path(folder)

    # X, Y, Z
    # box_aspect L_prl / L_prp
    L_prp = 1 / box_aspect
    # cell_aspect dx / dx_perp
    if not choose_res:
        res_aspect = box_aspect / cell_aspect  # N_prl / N_prp
        N_prp = int(Nx_init // res_aspect)
    
    n_X = np.array([Nx_init, N_prp, N_prp])
    X_min = np.array([0., 0., 0.])
    X_max = np.array([1., L_prp, L_prp])
    meshblock = diag.get_meshblocks(n_X, n_cpus)[0]

    # for editing athinput.from_array file
    time_lim = tlim if exp_rate == 0.0 else expand_to_a(a_end, exp_rate)
    expand = exp_rate != 0.0
    # time_lim = 6  # use this to manually set t_lim

    # init_norm_fluc <B^2_⟂0> / B^2_x0 = initial perp energy / initial parallel energy
    perp_energy = 0.5*init_norm_fluc    # initial energy of Alfvénic fluctuation components (assuming Bx0=1)

    iso_sound_speed = cs_from_beta(init_norm_fluc, beta)  # initial sound speed
    
    kpeak = (κ_prl, κ_prp)  # center of Gaussian spectrum peak
 
    do_truncation = 0  # cut off wave vectors above given mode numbers
    n_low, n_high = 0, 20 # modes to keep (0 <= n_low < n_high <= max(n_X)/2)

    Ls = X_max - X_min
    base_k_perp = 2*np.pi*np.sqrt((Ls[1:]**(-2)).sum())  # only worried about y and z
    kprp_mag_cutoff = (base_k_perp*n_low, base_k_perp*n_high)
    n_cutoff = (n_low, n_high)

    print('Doing ' + sim_name)
    h5name = 'ICs_' + sim_name + '.h5'
    athinput = 'athinput.from_array_' + h5name.split('.')[0]
    output = 'output_' + sim_name

    if gen_ic:
        genics.create_athena_alfvenspec(total_folder, h5name, n_X, X_min, X_max, meshblock,
                                        time_lim=time_lim, dt=dt, expand=expand, exp_rate=exp_rate,
                                        iso_sound_speed=iso_sound_speed, perp_energy=perp_energy,
                                        spectrum=spectrum, expo=expo, expo_prl=expo_prl, kpeak=kpeak,
                                        do_truncation=do_truncation, n_cutoff=n_cutoff)                      

    if run_athena:
        os.chdir(total_folder)
        command = 'mpiexec -n 4 ' + athena_path + ' -i ' + athinput + ' -d ' + output
        os.system(command)

    if run_spec:
        print(folder+output)
        inertial_range = np.array([6*10**1, 3*10**2])
        if do_truncation:
            inertial_range[0] = max(kprp_mag_cutoff[0], inertial_range[0])
            inertial_range[1] = min(kprp_mag_cutoff[1], inertial_range[1])
        spec.calc_spectrum(folder+output, folder+output, do_single_file=1, prob='from_array', return_dict=0)

    return athinput, n_X