COMPUTER = 'local'
# COMPUTER = 'nesi'

if COMPUTER == 'local':
    python_masters_path = '~/python_masters/'
else:
    python_masters_path = '/home/johza721/masters_2021/python_masters/'

import sys
sys.path.insert(0, python_masters_path)
import numpy as np
gen = __import__('generate_ics')
spec = __import__('spectrum')
paths = __import__('project_paths')

folder_root, athena_path = paths.PATH, paths.athena_path

def expand_to_a(a_final, exp_rate):
    # inverting definition of a to find time limit
    return (a_final - 1) / exp_rate


def cs_from_beta(init_norm_fluc, beta):
    return np.sqrt((1 + init_norm_fluc) * beta)

sim_name = ''
folder = ''
total_folder = folder + sim_name

save_folder = total_folder
athinput_in_folder = total_folder
athinput_in = ''
athdf_input = ''
h5name = 'ICs_' + sim_name + '.h5'
new_meshblock = None # np.array([50, 50, 50])

expand = 1
exp_rate = 0.5
a_final = 10
init_norm_fluc = 0.2
beta = 0.2
iso_sound_speed = cs_from_beta(init_norm_fluc, beta)
dt = 0.2

print('Reinterpolating ' + sim_name)

gen.reinterp_from_h5(save_folder, athinput_in_folder, athinput_in, h5name, athdf_input,
                     a_to_finish=a_final, dt=dt, iso_sound_speed=iso_sound_speed,
                     expand=expand, exp_rate=exp_rate, new_meshblock=new_meshblock)
