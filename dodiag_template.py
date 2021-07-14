import os

cdir_root = os.getcwd().split('/')

if cdir_root[1] in ['Users', 'home']:
    python_masters_path = '~/python_masters/'
else:
    python_masters_path = '/home/johza721/masters_2021/python_masters/'

import sys
sys.path.insert(0, python_masters_path)
run = __import__('run_diagnostics')
paths = __import__('project_paths')

PATH = paths.PATH

sim_name = ''  # simulation name
folder_name = ''
mult_sims = 0  # if multiple sims in folder
if mult_sims:
    folder = PATH + folder_name + '/' + sim_name + '/'
else:
    folder = PATH + sim_name + '/'

output_dir = folder + 'output/'
athinput_path = folder + 'ICs/athinput.from_array_ICs_' + sim_name
dict_name = sim_name
steps = 1
do_spectrum = 1
do_flyby = 1

run.run_loop(output_dir, athinput_path, dict_name, steps, do_spectrum, do_flyby)