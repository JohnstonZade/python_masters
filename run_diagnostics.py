from h5py import run_tests
import numpy as np
import diagnostics as diag
import spectrum as spec
import reinterpolate


def read_athinput(athinput_path):
    ath = open(athinput_path, 'r')
    list_of_lines = ath.readlines()
    tlim = float(list_of_lines[10].split('=')[1].split('#')[0])
    dt = float(list_of_lines[22].split('=')[1].split('#')[0])
    iso_sound_speed = float(list_of_lines[53].split('=')[1].split('#')[0])
    expand = bool(list_of_lines[66].split('=')[1].split('#')[0])
    expansion_rate = float(list_of_lines[67].split('=')[1].split('#')[0]) if expand else 0.
    return expansion_rate, iso_sound_speed, tlim, dt
 

def run_loop(output_dir, athinput_path, steps=10, do_spectrum=0, do_flyby=1, flyby_a=-999):
    
    max_n = diag.get_maxn(output_dir)
    # overestimate the number of steps needed; edge case is handled when loading data
    n_steps = int(np.ceil(max_n / steps))

    # if dictionary is not already there, do full calc
    # otherwise load dictionary
    do_full_calc = not diag.check_dict(output_dir, 'data_dump')
    if not do_full_calc:
        S = diag.load_dict(output_dir, 'data_dump')
        print('Not doing full calculation')
    else:
        S = {}

    # expansion rate, sound speed from athinput
    expansion_rate, c_s_init, tlim, dt = read_athinput(athinput_path)
    S['expansion_rate'] = expansion_rate
    S['sound_speed'] = c_s_init

    if do_full_calc:
        t_full, a_full = np.array([]), np.array([])
        Bx_mean_full, beta_full, sb_frac_full = np.array([]), np.array([]), np.array([])
        cross_h_full, Bprp_fluc_full, uprp_fluc_full = np.array([]), np.array([]), np.array([])
        magcomp_sq_full, magcomp_sh_full = np.array([]), np.array([])

        print('max_n = ' + str(max_n))
        for i in range(n_steps):
            n_start = i*steps
            n_end = (i+1)*steps
            print(' Analysing n = ' + str(i*steps) + ' to ' + str((i+1)*steps - 1))
            t, a, B, u, rho = diag.load_time_series(output_dir, n_start, n_end)
            print('     Data loaded')
            
            u_prp, B_prp = u[:, 1:], B[:, 1:]
            B_x = B[:, 0]
            if i == 0:
                B0_x = B_x[0, 0, 0, 0]  # initial B0_x
            B_mag = np.sqrt(diag.dot_prod(B, B, 1))

            t_full = np.append(t_full, t)
            a_full = np.append(a_full, a)
            Bx_mean_full = np.append(Bx_mean_full, diag.box_avg(B_x))

            beta_full = np.append(beta_full, diag.beta(rho, B_mag, c_s_init, expansion_rate, t))
            sb_frac_full = np.append(sb_frac_full, diag.switchback_fraction(B_x, B_mag, B0_x))
            cross_h_full = np.append(cross_h_full, diag.cross_helicity(rho, u_prp, B_prp))
            # Bprp_fluc_full = np.append(Bprp_fluc_full, diag.norm_fluc_amp(diag.dot_prod(B_prp, B_prp, 1), B_x))
            # uprp_fluc_full = np.append(uprp_fluc_full, diag.norm_fluc_amp(rho*diag.dot_prod(u_prp, u_prp, 1), B_x))
            magcomp_sq_full = np.append(magcomp_sq_full, diag.mag_compress_Squire2020(B))
            # magcomp_sh_full = np.append(magcomp_sh_full, diag.mag_compress_Shoda2021(B))

            # clear unneeded variables to save memory, run flyby code after this
            t, a = None, None
            B, u, B_prp, u_prp, B_x, B_mag = None, None, None, None, None, None
            print('     Data cleared')

        S['time'] = t_full
        S['perp_expand'] = a_full
        S['Bx_mean'] = Bx_mean_full
        S['beta'] = beta_full
        S['sb_frac'] = sb_frac_full
        S['cross_helicity'] = cross_h_full
        S['C_B2_Squire'] = magcomp_sq_full
        # S['C_B2_Shoda'] = magcomp_sh_full
        
        a_normfluc, Bprp_fluc, uprp_fluc = diag.norm_fluc_amp_hst(output_dir)
        S['a_normfluc'] = a_normfluc
        S['norm_fluc_Bprp'] = Bprp_fluc
        S['norm_fluc_uprp'] = uprp_fluc

    if do_spectrum:
        spec_hik_energy, spec_hik_a = np.array([]), np.array([])
        for n in range(max_n):
            if n % 5 == 0:  # don't want to run too often
                print('Spectrum calculation started at n = ' + str(n))
                spec_a = round(S['perp_expand'][n], 2)
                spec_name = 'mhd_spec_a' + str(spec_a)
                S[spec_name] = spec.calc_spectrum(output_dir, output_dir, prob='from_array', dict_name=spec_name,
                                                return_dict=1, do_single_file=1, n=n, a=spec_a)
                spec_hik_a = np.append(spec_hik_a, spec_a)
                spec_hik_energy = np.append(spec_hik_energy, spec_hik_energy_frac(S[spec_name]))
        T = {}
        T['hi_energy_frac'], T['hi_energy_a'] = spec_hik_energy, spec_hik_a
        S['energy_in_hi_k'] = T

    if do_flyby:
        try:
            t = np.arange(0, tlim+dt, dt)
            a = diag.a(expansion_rate, t)

            # set to do last flyby by default
            if flyby_a == -999:
                flyby_a = a[-1]
            assert a[0] <= flyby_a <= a[-1], 'Please choose a valid a!'
            # index of the given a value we want
            flyby_n = int(np.round((flyby_a - 1) / (expansion_rate * dt)))

            if max_n < int((tlim+dt)/dt) and max_n <= flyby_n:
                raise ValueError('Athena has not output the expected number of .athdf files, and we are trying to access a file that does not exist',
                                'max_n = ' + str(max_n), 'expected max_n = ' + str(int((tlim+dt)/dt)), 'flyby_n = ' + str(flyby_n))

            flyby_string = 'flyby_a' + str(flyby_a)
            S[flyby_string] = reinterpolate.flyby(output_dir, flyby_a, flyby_n)
            print(flyby_string + ' done')
        except ValueError as ve:
            print(ve.args)

    diag.save_dict(S, output_dir, 'data_dump')


def spec_hik_energy_frac(S):
    k = S['kgrid']
    kmax = max(k)
    EM = S['EM_prp']
    hik_EM = EM[k > (kmax/6)].sum()
    tot_EM = EM.sum()
    return hik_EM / tot_EM