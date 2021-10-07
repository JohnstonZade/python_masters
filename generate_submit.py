import numpy as np
import diagnostics as diag
import genscript as gen
import project_paths as paths

python_masters_path, scratch_path = paths.python_masters_path, paths.SCRATCH


def make_slurm_file(job_name, n_nodes, athinput_out, reinterp=0, r_number=0, sim_name=None, folder=None,
                    athinput_in=None, athdf_input=None, n_X=None, a_final=None, a_re=None):
    
    # Generate the submit script for running Athena
    fname = 'submit'
    fname += '_r' + str(r_number) if reinterp else ''
    with open(fname, 'w') as f:
        f.write('#!/bin/bash -e\n\n')
        f.write('#SBATCH --job-name=' + job_name + '\n')
        f.write('#SBATCH -N ' + str(n_nodes) + '\n')
        f.write('#SBATCH --ntasks-per-node=40\n#SBATCH -t 24:00:00\n')  # update to modify time
        runout = 'run_r' + str(r_number) + '.out' if reinterp else 'run.out'
        f.write('#SBATCH --output=' + runout + '\n')
        f.write('#SBATCH --account=uoo02637\n#SBATCH --mail-type=begin\n')
        f.write('#SBATCH --mail-type=end\n#SBATCH --mail-user=johza721@student.otago.ac.nz\n\n')
        f.write('echo ${SLURM_NODELIST}\n\n')
        
        athena_run = 'srun /home/johza721/masters_2021/athena/bin/athena_maui/athena -i '
        athena_run += athinput_out
        athena_run += ' -d output -t 23:55:00\n'
        f.write(athena_run)
    
    # If reinterpolating, generate the submit script to first generate the reinterpolated ICs
    if reinterp:
        fname = 'submit_genr' + str(r_number)
        with open(fname, 'w') as f:
            f.write('#!/bin/bash -e\n\n')
            f.write('#SBATCH --job-name=' + job_name + '_gen\n')
            f.write('#SBATCH -N 1\n')
            f.write('#SBATCH --ntasks-per-node=1\n#SBATCH -t 00:30:00\n')
            runout = 'run_genr' + str(r_number) + '.out'
            f.write('#SBATCH --output=' + runout + '\n')
            f.write('#SBATCH --account=uoo02637\n')
            # f.write('#SBATCH --mail-type=begin\n')
            # f.write('#SBATCH --mail-type=end\n#SBATCH --mail-user=johza721@student.otago.ac.nz\n\n')
            f.write('echo ${SLURM_NODELIST}\n\n')
            
            comment = '# sim_name folder athinput_in athdf_input resolution n_cpus a_final a_re\n'
            f.write(comment)
            gen_path = 'srun python ' + python_masters_path + 'genscript_reinterpolate.py '
            gen_path += sim_name + ' '
            gen_path += folder + ' '
            gen_path += athinput_in + ' '
            gen_path += athdf_input + ' '
            gen_path += '"' + str(n_X[0]) + ',' + str(n_X[1]) + ',' + str(n_X[2]) + '" '
            n_cpus = 40*n_nodes
            gen_path += str(int(n_cpus)) + ' '
            gen_path += str(a_final) + ' '
            gen_path += str(a_re) + '\n'
            f.write(gen_path)


def generate_slurm(sim_name, folder, box_aspect, cell_aspect, Nx_init, n_nodes,
                   exp_rate, init_norm_amp, beta, gen_init_ic=1, a_re=None, a_end=4, 
                   dt=0.2, spectrum='isotropic', κ_prl=2, κ_prp=2):

    # If the reinterpolation and ending a are not specified,
    # expand the box to a cubic size by default
    if a_re is None:
        a_re = a_end

    # Generate initial athinput and ICs.h5 files in folder
    n_cpus = 40*n_nodes
    athinput_orig, n_X = gen.generate(sim_name, folder, box_aspect, cell_aspect, Nx_init, n_cpus, exp_rate,
                                      dt, init_norm_amp, beta, spectrum=spectrum, 
                                      κ_prl=κ_prl, κ_prp=κ_prp, a_end=a_re, gen_ic=gen_init_ic)

    # Generate initial slurm file (just modify job name, Ncpus, athinput name)
    make_slurm_file(sim_name, n_nodes, athinput_orig)

    # E.g. a_end = 10, a_re = 2
    # a:1->2 reinterp a:2->4 reinterp a:4->8 reinterp a:8->10
    # 3 reinterps = floor(log_a_re(a_end))
    n_reinterp = int(np.floor(np.log(a_end)/np.log(a_re)))
    if a_re**n_reinterp == a_end:
        n_reinterp -= 1  # don't need to reinterpolate at final a
    n_output = (a_re - 1) / (exp_rate * dt)  # file number of last athdf output
    n_diff = n_output

    # Generate slurm files for each additional interpolation
    # Add line to genscript_reinterpolate with cmdline arguments before athena
    folder = scratch_path + folder
    for i in range(1, n_reinterp+1):
        a_final = min(a_re**(i+1), a_end)
        print('Reinterpolating a: ' + str(a_re**i) + ' -> ' + str(a_final))
        reinterp_sim = sim_name + '_r' + str(i)

        athinput_in = athinput_orig if i == 1 else athinput_out
        athinput_out = athinput_orig + '_r' + str(i)

        last_athdf = folder + 'output/from_array.out2' + '.%05d' % int(n_output) + '.athdf'
        n_diff *= a_re
        n_output += n_diff

        n_X[0] //= a_re
        make_slurm_file(reinterp_sim, n_nodes, athinput_out, reinterp=1,
                        r_number=i, sim_name=reinterp_sim, folder=folder, athinput_in=athinput_in,
                        athdf_input=last_athdf, n_X=n_X, a_final=a_final, a_re=a_re)

