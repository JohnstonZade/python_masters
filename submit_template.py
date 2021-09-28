import os

cdir_root = os.getcwd().split('/')

if cdir_root[1] in ['Users', 'home']:
    python_masters_path = '~/python_masters/'
else:
    python_masters_path = '/home/johza721/masters_2021/python_masters/'

import sys
sys.path.insert(0, python_masters_path)
submit = __import__('generate_submit')
gen = __import__('genscript')
spectra = ['isotropic', 'anisotropic', 'gaussian']

sim_name = ''    # base simulation name
folder = ''      # in project folder root
box_aspect = 10  # Lprl / Lprp
cell_aspect = 2  # dx / dx_prp

reinterpolate = False
a_re = 2         # factor to reinterpolate (e.g a -> 2a then reinterp)
a_end = 10       # final expansion
Nx_init = 64     # initial x resolution
iso_res = False  # override resolution aspect ratio

n_nodes = 1      # number of nodes
n_cpus = 40*n_nodes    
exp_rate = 0.5
dt = 0.2
init_norm_fluc = 0.2  # amplitude squared
beta = 0.2  # initial beta
spectrum_n = 2 # 0 for isotropic, 1 for GS, 2 for Gaussian
spectrum = spectra[spectrum_n]
κ_prl, κ_prp = 2, 2  # peak of gaussian spectrum


if reinterpolate:   
    submit.generate_slurm(sim_name, folder, box_aspect, cell_aspect, Nx_init, n_nodes,
                        exp_rate, init_norm_fluc, beta, spectrum=spectrum, dt=dt,
                        a_re=a_re, a_end=a_end, κ_prl=κ_prl, κ_prp=κ_prp)
else:
    gen.generate(sim_name, folder, box_aspect, cell_aspect, Nx_init, n_cpus, exp_rate,
                 dt, init_norm_fluc, beta, spectrum=spectrum, κ_prl=κ_prl, κ_prp=κ_prp,
                 a_end=a_end)