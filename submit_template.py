computer = 'local'
# computer = 'nesi'

if computer == 'local':
    python_masters_path = '~/python_masters/'
else:
    python_masters_path = '/home/johza721/masters_2021/python_masters/'
import sys
sys.path.insert(0, python_masters_path)
submit = __import__('generate_submit')

sim_name = ''  # base simulation name
folder = ''  # in project folder root
box_aspect = 10  # Lprl / Lprp
cell_aspect = 2  # dx / dx_prp
Nx_init = 64     # initial x resolution
n_nodes = 1        # number of nodes     
exp_rate = 0.5
init_norm_fluc = 0.2
beta = 0.2
spectrum = 'gauss'
# spectrum = 'gs'

submit.generate_slurm(sim_name, folder, box_aspect, cell_aspect, Nx_init, n_nodes,
                      exp_rate, init_norm_fluc, beta, spec=spectrum)