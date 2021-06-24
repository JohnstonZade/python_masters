import numpy as np
import diagnostics as diag
import generate_ics as genics
import project_paths as paths
import argparse

folder_root = paths.SCRATCH

def expand_to_a(a_final, exp_rate):
    # inverting definition of a to find time limit
    return (a_final - 1) / exp_rate


def cs_from_beta(init_norm_fluc, beta):
    return np.sqrt((1 + init_norm_fluc) * beta)

parser = argparse.ArgumentParser()

parser.add_argument('sim_name')
parser.add_argument('folder')
parser.add_argument('athinput_in')
parser.add_argument('athdf_input')
parser.add_argument('resolution', help='delimited list input', 
                     type=lambda s: [int(item) for item in s.split(',')])
parser.add_argument('n_cpus', type=int)
parser.add_argument('a_finish', type=int)
parser.add_argument('a_re', type=int)

args = vars(parser.parse_args())

sim_name = args['sim_name']
folder = args['folder']
total_folder = folder # + sim_name

save_folder = total_folder
athinput_in_folder = total_folder
athinput_in = args['athinput_in']
athdf_input = args['athdf_input']  # get from slurm
h5name = 'ICs_' + sim_name + '.h5'
resolution = np.array(args['resolution'])
new_meshblock = diag.get_meshblocks(resolution, args['n_cpus'])[0]

a_finish = args['a_final']
a_re = args['cell_aspect']

print('Reinterpolating ' + sim_name)

genics.reinterp_from_h5(save_folder, athinput_in_folder, athinput_in, h5name, athdf_input,
                        a_finish=a_finish, a_re=a_re, new_meshblock=new_meshblock)
