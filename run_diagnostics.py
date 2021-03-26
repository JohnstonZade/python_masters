from h5py import run_tests
import numpy as np
import diagnostics as diag
import spectrum as spec
import spacecraft_flyby as spacecraft


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
            t, B, u, rho = diag.load_time_series(output_dir, n_start, n_end)
            print('     Data loaded')
            a = diag.a(expansion_rate, t)

            B, u = diag.expand_variables(a, B), diag.expand_variables(a, u)
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
            Bprp_fluc_full = np.append(Bprp_fluc_full, diag.norm_fluc_amp(diag.dot_prod(B_prp, B_prp, 1), B_x))
            uprp_fluc_full = np.append(uprp_fluc_full, diag.norm_fluc_amp(rho*diag.dot_prod(u_prp, u_prp, 1), B_x))
            magcomp_sq_full = np.append(magcomp_sq_full, diag.mag_compress_Squire2020(B))
            magcomp_sh_full = np.append(magcomp_sh_full, diag.mag_compress_Shoda2021(B))

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
        S['norm_fluc_Bprp'] = Bprp_fluc_full
        S['norm_fluc_uprp'] = uprp_fluc_full
        S['C_B2_Squire'] = magcomp_sq_full
        S['C_B2_Shoda'] = magcomp_sh_full
    
    t = np.arange(0, tlim+dt, dt)
    a = diag.a(expansion_rate, t)

    # set to do last flyby by default
    if flyby_a == -999:
        flyby_a = a[-1]
    assert a[0] <= flyby_a <= a[-1], 'Please choose a valid a!'
    # index of the given a value we want
    flyby_n = int(np.round((flyby_a - 1) / (expansion_rate * dt)))

    if do_spectrum:
        spec_name = 'mhd_spec_a' + str(flyby_a)
        S[spec_name] = spec.calc_spectrum(output_dir, output_dir, prob='from_array', dict_name=spec_name,
                                          return_dict=1, do_single_file=1, n=flyby_n, a=flyby_a)
        print(spec_name + ' done')

    if do_flyby:
        flyby_string = 'flyby_a' + str(flyby_a)
        S[flyby_string] = spacecraft.flyby(output_dir, flyby_a, flyby_n)
        print(flyby_string + ' done')

    diag.save_dict(S, output_dir, 'data_dump')

def run(output_dir, athinput_path, do_flyby=1, flyby_do_last=1,  flyby_n=0):
    
    # if dictionary is not already there, do full calc
    # otherwise load dictionary
    do_full_calc = not diag.check_dict(output_dir, 'data_dump')
    if not do_full_calc:
        S = diag.load_dict(output_dir, 'data_dump')
    else:
        S = {}

    # expansion rate, sound speed from athinput
    expansion_rate, c_s_init, tlim, dt = read_athinput(athinput_path)
    S['expansion_rate'] = expansion_rate
    S['sound_speed'] = c_s_init
    
    if do_full_calc:
        t, B, u, rho = diag.load_time_series(output_dir)
        a = diag.a(expansion_rate, t)
        S['time'] = t
        S['perp_expand'] = a

        B, u = diag.expand_variables(a, B), diag.expand_variables(a, u)
        u_prp, B_prp = u[:, 1:], B[:, 1:]
        B_x = B[:, 0]
        S['Bx_mean'] = diag.box_avg(B_x)
        B0_x = B_x[0, 0, 0, 0]
        B_mag = np.sqrt(diag.dot_prod(B, B, 1))

        S['beta'] = diag.beta(rho, B_mag, c_s_init, expansion_rate, t)
        print('beta done')
        S['sb_frac'] = diag.switchback_fraction(B_x, B_mag, B0_x)
        print('sb frac done')
        S['cross_helicity'] = diag.cross_helicity(rho, u_prp, B_prp)
        print('cross helicity done')
        S['norm_fluc_Bprp'] = diag.norm_fluc_amp(diag.dot_prod(B_prp, B_prp, 1), B_x)
        S['norm_fluc_uprp'] = diag.norm_fluc_amp(rho*diag.dot_prod(u_prp, u_prp, 1), B_x)
        print('fluctuations done')
        S['C_B2_Squire'] = diag.mag_compress_Squire2020(B)
        S['C_B2_Shoda'] = diag.mag_compress_Shoda2021(B)
        print('compressibility done')

        # clear unneeded variables to save memory, run flyby code after this
        B, u, B_prp, u_prp, B_x, B_mag = None, None, None, None, None, None 

    if do_flyby:
        N_t = int(tlim / dt)
        t = np.arange(0, tlim+dt, N_t)
        a = diag.a(expansion_rate, t)
        # automatically starts at a random point
        # in the last snapshot.
        if 'flyby' in S.keys():
            S.pop('flyby')
        S['flyby'] = spacecraft.flyby(output_dir, a, do_last=flyby_do_last, n=flyby_n)
        print('flyby done')

    diag.save_dict(S, output_dir, 'data_dump')