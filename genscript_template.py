COMPUTER = 'local'
# COMPUTER = 'nesi'

if COMPUTER == 'local':
    python_masters_path = '~/python_masters/'
else:
    python_masters_path = '/home/johza721/masters_2021/python_masters/'

import sys
sys.path.insert(0, python_masters_path)
import numpy as np
import os
gen = __import__('generate_ics')
spec = __import__('spectrum')
paths = __import__('project_paths')

folder_root, athena_path = paths.PATH, paths.athena_path

def expand_to_a(a_final, exp_rate):
    # inverting definition of a to find time limit
    return (a_final - 1) / exp_rate

def wave_energy(wave_amp):
    return wave_amp**2 / 4

def cs_from_beta(init_norm_fluc, beta):
    return np.sqrt((1 + init_norm_fluc) * beta)

sim_name = ''  # put simulation name here
folder = ''    # put folder to output here, make sure Athena binary is in this folder 
total_folder =  folder_root + folder

# X, Y, Z
n_X = np.array([64, 64, 64])
X_min = np.array([0., 0., 0.])
X_max = np.array([1., 1., 1.])
meshblock = np.array([32, 32, 32])

# for editing athinput.from_array file
expand = 1      # use expanding box model
exp_rate = 0.5  # expansion rate
a_final = 8     
time_lim = expand_to_a(a_final, exp_rate)
# time_lim = 6  # use this to manually set t_lim
dt = 0.2        # time step for simulation output

# wave_amp = 0.01  # equiv to amplitude of <δB^2_⟂> / B^2_x when B_x = 1
# energy = wave_energy(wave_amp)
init_norm_fluc = 0.2  # <B^2_⟂0> / B^2_x0 = initial perp energy / initial parallel energy
perp_energy = 0.5*init_norm_fluc    # initial energy of Alfvénic fluctuation components (assuming Bx0=1)

beta = 0.2
iso_sound_speed = cs_from_beta(init_norm_fluc, beta)  # initial sound speed

expo = -5/3       # spectrum power law 
kprl = -2       # aniostropic parallel spectrum power law
prl_spec = 0    # generate anisotropic spectrum
gauss_spec = 0  # generate gaussian spectrum
do_truncation = 0  # cut off wave vectors above given mode numbers
n_low, n_high = 0, 20 # modes to keep (0 <= n_low < n_high <= max(n_X)/2)

Ls = X_max - X_min
base_k_perp = 2*np.pi*np.sqrt((Ls[1:]**(-2)).sum())  # only worried about y and z
kprp_mag_cutoff = (base_k_perp*n_low, base_k_perp*n_high)
n_cutoff = (n_low, n_high)

generate = 1
run_athena = 1
run_spectrum = 1

print('Doing ' + sim_name)
h5name = 'ICs_' + sim_name + '.h5'
athinput = 'athinput.from_array_' + h5name.split('.')[0]
output = 'output_' + sim_name

if generate:
    gen.create_athena_alfvenspec(total_folder, h5name, n_X, X_min, X_max, meshblock, 
                                time_lim=time_lim, dt=dt, iso_sound_speed=iso_sound_speed,
                                expand=expand, exp_rate=exp_rate, do_truncation=do_truncation, n_cutoff=n_cutoff,
                                energy=perp_energy, expo=expo, expo_prl=kprl, prl_spec=prl_spec, gauss_spec=gauss_spec)                      

if COMPUTER == 'local' and run_athena:
    os.chdir(total_folder)
    command = 'mpiexec -n 4 ' + athena_path + ' -i ' + athinput + ' -d ' + output
    os.system(command)

if run_spectrum:
    do_prp_spec = 1
    inertial_range = np.array([6*10**1, 3*10**2])
    if do_truncation:
        inertial_range[0] = max(kprp_mag_cutoff[0], inertial_range[0])
        inertial_range[1] = min(kprp_mag_cutoff[1], inertial_range[1])
    spec.calc_spectrum(folder+output, folder+output, sim_name, do_single_file=1, do_prp_spec=do_prp_spec, prob='from_array', inertial_range=inertial_range)
