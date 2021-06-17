import numpy as np
import diagnostics as diag
import genscript as gen
import project_paths as paths

python_masters_path, scratch_path = paths.python_masters_path, paths.SCRATCH


def make_slurm_file(job_name, n_cpus, athinput_out, reinterp=0, r_number=0, sim_name=None, folder=None,
                    athinput_in=None, athdf_input=None, n_X=None, a_final=None, cell_aspect=None):
    fname = 'submit'
    fname += '_r' + str(r_number) if reinterp else ''
    with open(fname, 'w') as f:
        f.write('#!/bin/bash -e\n\n')
        f.write('#SBATCH --job-name=' + job_name + '\n')
        f.write('#SBATCH -N ' + str(n_cpus) + '\n')
        f.write('#SBATCH --ntasks-per-node=40\n#SBATCH -t 24:00:00\n')  # update to modify time
        runout = 'run_r' + str(r_number) + '.out' if reinterp else 'run.out'
        f.write('#SBATCH --output=' + runout + '\n')
        f.write('#SBATCH --account=uoo02637\n#SBATCH --mail-type=begin\n')
        f.write('#SBATCH --mail-type=end\n#SBATCH --mail-user=johza721@student.otago.ac.nz\n\n')
        f.write('echo ${SLURM_NODELIST}\n\n')

        if reinterp:
            gen_path = 'srun --pty python ' + python_masters_path + 'genscript_reinterpolate.py '
            gen_path += sim_name + ' '
            gen_path += folder + ' '
            gen_path += athinput_in + ' '
            gen_path += athdf_input + ' '
            gen_path += '"' + str(n_X[0]) + ',' + str(n_X[1]) + ',' + str(n_X[2]) + '" '
            gen_path += str(a_final) + ' '
            gen_path += str(cell_aspect) + '\n'
            f.write(gen_path)
        
        athena_run = 'srun /home/johza721/masters_2021/athena/bin/athena_maui/athena -i '
        athena_run += athinput_out
        athena_run += ' -d output -t 23:55:00\n'
        f.write(athena_run)


def generate_slurm(sim_name, folder, box_aspect, cell_aspect, Nx_init, n_cpus,
                   exp_rate, init_norm_fluc, beta, dt=0.2, spec='iso'):

    # Generate initial athinput and ICs.h5 files in folder
    athinput_orig, n_X = gen.generate(sim_name, folder, box_aspect, cell_aspect, Nx_init, n_cpus, exp_rate,
                                      dt, init_norm_fluc, beta, reinterp=1, spec=spec)

    # Generate initial slurm file (just modify job name, Ncpus, athinput name)
    make_slurm_file(sim_name, n_cpus, athinput_orig)

    # E.g. box = 10, cell = 2
    # a:1->2 reinterp a:2->4 reinterp a:4->8 reinterp a:8->10
    # 3 reinterps = floor(log_cell(box))
    n_reinterp = int(np.floor(np.log(box_aspect)/np.log(cell_aspect)))
    n_output = (cell_aspect - 1) / (exp_rate * dt)  # file number of last athdf output

    # Generate slurm files for each additional interpolation
    # Add line to genscript_reinterpolate with cmdline arguments before athena
    for i in range(1, n_reinterp+1):
        reinterp_sim = sim_name + '_r' + str(i)

        if i == 1: 
            athinput_in = athinput_orig
        else:
            athinput_in = athinput_out
        athinput_out = athinput_orig + '_r' + str(i)

        last_athdf = folder + 'output/from_array.out2' + '.%05d' % int(n_output) + '.athdf'
        n_output += cell_aspect * n_output

        n_X[0] //= cell_aspect
        a_final = min(cell_aspect**(i+1), box_aspect)
        make_slurm_file(reinterp_sim, n_cpus, athinput_out, reinterp=1,
                        r_number=i, sim_name=reinterp_sim, folder=folder, athinput_in=athinput_in,
                        athdf_input=last_athdf, n_X=n_X, a_final=a_final, cell_aspect=cell_aspect)

