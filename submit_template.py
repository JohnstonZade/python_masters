import os

cdir_root = os.getcwd().split('/')

if cdir_root[1] in ['Users', 'Volumes']:
    python_masters_path = '/Users/johza22p/python_masters/'
elif cdir_root[1] == 'home':
    python_masters_path = '~/python_masters/'
else:
    python_masters_path = '/home/johza721/masters_2021/python_masters/'

import sys
sys.path.insert(0, python_masters_path)
submit = __import__('generate_submit')
genics = __import__('generate_ics')
gen = __import__('genscript')
spectra = ['isotropic', 'anisotropic', 'gaussian']

reinterpolate = False
athdf_to_h5 = False

sim_name = ''    # base simulation name
folder = ''      # in project folder root
box_aspect = 10  # Lprl / Lprp
cell_aspect = 2  # dx / dx_prp
Nx_init = 64     # initial x resolution
choose_res = False  # override resolution aspect ratio
N_prp = 64       # perpendicular resolution if overriden

a_end = 10       # final expansion
exp_rate = 0.5

init_norm_amp = 0.2  # initial normalized amplitude
beta = 0.2  # initial beta
spectrum_n = 2 # 0 for isotropic, 1 for GS, 2 for Gaussian
spectrum = spectra[spectrum_n]
κ_prl, κ_prp = 2, 2  # peak of gaussian spectrum

dt = 0.2
tlim = 2 if exp_rate == 0.0 else (a_end - 1) / exp_rate
# Reinterpolation
if reinterpolate:
    a_re = 2         # factor to reinterpolate (e.g a -> 2a then reinterp)

# Athdf to H5 Generation
if athdf_to_h5:
    athinput_in_folder = ''  # path to folder containing original athinput
    athinput_in = ''  # name of athinput file
    h5name = ''  # name of h5 file to be saved
    athdf_input = ''  # full path to athdf input to be read
    beta_multiplier = 1  # set to 1 to use beta of simulation



n_nodes = 1      # number of nodes
n_cpus = 40*n_nodes  



if reinterpolate:   
    submit.generate_slurm(sim_name, folder, box_aspect, cell_aspect, Nx_init, n_nodes,
                        exp_rate, init_norm_amp, beta, spectrum=spectrum, dt=dt,
                        a_re=a_re, a_end=a_end, κ_prl=κ_prl, κ_prp=κ_prp)
elif athdf_to_h5:
    genics.create_athena_fromh5(folder, athinput_in_folder, athinput_in, h5name, athdf_input,
                                tlim, dt, beta_multiplier, exp_rate=exp_rate)
else:
    gen.generate(sim_name, folder, box_aspect, cell_aspect, Nx_init, n_cpus, exp_rate,
                 dt, init_norm_amp, beta, tlim=tlim, choose_res=choose_res, N_prp=N_prp,
                 spectrum=spectrum, κ_prl=κ_prl, κ_prp=κ_prp, a_end=a_end)