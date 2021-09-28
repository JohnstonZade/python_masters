import numpy as np
import diagnostics as diag
import spectrum as spec
import reinterpolate
from athena_read import athinput as athinput_dict


def read_athinput(athinput_path):
    ath_dict = athinput_dict(athinput_path)
    
    tlim = ath_dict['time']['tlim']
    dt = ath_dict['output2']['dt']
    iso_sound_speed = ath_dict['hydro']['iso_sound_speed']
    expand = bool(ath_dict['problem']['expanding'])
    expansion_rate = ath_dict['problem']['expand_rate'] if expand else 0.
    
    return expansion_rate, iso_sound_speed, tlim, dt
 

def run_loop(output_dir, athinput_path, dict_name='data_dump', steps=10, do_spectrum=0, do_flyby=1, override_not_full_calc=0,
             method='matt'):
    
    max_n = diag.get_maxn(output_dir)
    # overestimate the number of steps needed; edge case is handled when loading data
    n_steps = int(np.ceil(max_n / steps))

    # if dictionary is not already there, do full calc
    # otherwise load dictionary
    do_full_calc = (not diag.check_dict(output_dir, dict_name)) or override_not_full_calc
    if not do_full_calc:
        S = diag.load_dict(output_dir, dict_name)
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
        cross_h_full, z_p_full, z_m_full = np.array([]), np.array([]), np.array([])
        Bprp_fluc_full, uprp_fluc_full, kinetic_fluc_full = np.array([]), np.array([]), np.array([])
        magcomp_sq_full = np.array([])
        # magcomp_sh_full = np.array([])

        print('max_n = ' + str(max_n))
        for i in range(n_steps):
            n_start = i*steps
            n_end = (i+1)*steps
            print(' Analysing n = ' + str(i*steps) + ' to ' + str((i+1)*steps - 1))
            t, a, B, u, rho = diag.load_time_series(output_dir, n_start, n_end, method=method)
            print('     Data loaded')
            
            u_prp, B_prp = u[:, 1:], B[:, 1:]
            B_x = B[:, 0]
            if i == 0:
                B0_x = B_x[0, 0, 0, 0]  # initial B0_x
            B_mag = np.sqrt(diag.dot_prod(B, B, 1))
            rho_avg = diag.box_avg(rho)
            z_p_rms, z_m_rms = diag.z_waves_evo(rho, u_prp, B_prp, a)

            t_full = np.append(t_full, t)
            a_full = np.append(a_full, a)
            Bx_mean_full = np.append(Bx_mean_full, diag.box_avg(B_x))

            beta_full = np.append(beta_full, diag.beta(rho, B_mag, c_s_init, expansion_rate, t))
            sb_frac_full = np.append(sb_frac_full, diag.switchback_fraction(B_x, B_mag, B0_x))
            cross_h_full = np.append(cross_h_full, diag.cross_helicity(rho, u_prp, B_prp))
            z_p_full = np.append(z_p_full, z_p_rms)
            z_m_full = np.append(z_m_full, z_m_rms)
            Bprp_fluc_full = np.append(Bprp_fluc_full, diag.norm_fluc_amp(diag.dot_prod(B_prp, B_prp, 1), B_x))
            uprp_fluc_full = np.append(uprp_fluc_full, diag.norm_fluc_amp(diag.dot_prod(u_prp, u_prp, 1), B_x / np.sqrt(rho_avg)))
            kinetic_fluc_full = np.append(kinetic_fluc_full, diag.norm_fluc_amp(rho*diag.dot_prod(u_prp, u_prp, 1), B_x))
            magcomp_sq_full = np.append(magcomp_sq_full, diag.mag_compress_Squire2020(B))
            # magcomp_sh_full = np.append(magcomp_sh_full, diag.mag_compress_Shoda2021(B))

            # clear unneeded variables to save memory, run flyby code after this
            t, a = None, None
            B, u, B_prp, u_prp, B_x, B_mag, rho_avg = None, None, None, None, None, None, None
            print('     Data cleared')

        S['time'] = t_full
        S['perp_expand'] = a_full
        S['Bx_mean'] = Bx_mean_full
        S['beta'] = beta_full
        S['sb_frac'] = sb_frac_full
        S['Bprp_norm_fluc'] = Bprp_fluc_full
        S['uprp_norm_fluc'] = uprp_fluc_full
        S['kinetic_norm_fluc'] = kinetic_fluc_full
        S['cross_helicity'] = cross_h_full
        S['z_plus'] = z_p_full
        S['z_minus'] = z_m_full
        S['C_B2_Squire'] = magcomp_sq_full
        # S['C_B2_Shoda'] = magcomp_sh_full
        
        a_normfluc, Bprp_fluc, kinetic_fluc = diag.norm_fluc_amp_hst(output_dir, expansion_rate, method)
        S['a_norm_fluc_hst'] = a_normfluc
        S['Bprp_norm_fluc_hst'] = Bprp_fluc
        S['kinetic_norm_fluc_hst'] = kinetic_fluc

        diag.save_dict(S, output_dir, dict_name)

    a_step = 1 if (1 + expansion_rate*(max_n-1)*dt > 2) else 0.1
    t_step = dt
    if expansion_rate != 0:
        t_step *= expansion_rate
    spec_step = int(a_step / t_step)  # eg if delta_a = 1, adot=0.5, dt=0.2 then spec_step = 10
    if do_spectrum:
        spec_hik_mag, spec_hik_kin, spec_hik_a = np.array([]), np.array([]), np.array([])
        for n in range(max_n):
            if n % spec_step == 0:  # don't want to run too often
                print('Spectrum calculation started at n = ' + str(n))
                if expansion_rate != 0.0:
                    spec_a = round(S['perp_expand'][n], 1)
                    spec_name = 'mhd_spec_a' + str(spec_a)
                else:
                    spec_a = 1.0
                    spec_name = 'mhd_spec_t' + str(round(S['time'][n], 1))
                S[spec_name] = spec.calc_spectrum(output_dir, output_dir, prob='from_array', dict_name=spec_name,
                                                  do_single_file=1, n=n, a=spec_a, method=method)
                spec_hik_a = np.append(spec_hik_a, spec_a)
                spec_hik_mag = np.append(spec_hik_mag, spec_hik_energy_frac(S[spec_name]))
                spec_hik_kin = np.append(spec_hik_kin, spec_hik_energy_frac(S[spec_name], do_magnetic=0))
        T = {}
        T['hi_mag_energy_frac'], T['hi_kin_energy_frac'], T['hi_energy_a'] = spec_hik_mag, spec_hik_kin, spec_hik_a
        S['energy_in_hi_k'] = T
        diag.save_dict(S, output_dir, dict_name)

    if do_flyby:
        for n in range(max_n):
            if n % spec_step == 0:
                if expansion_rate != 0:
                    flyby_a = round(S['perp_expand'][n], 1)
                    flyby_string = 'flyby_a' + str(flyby_a)
                else:
                    flyby_a = 1.0
                    flyby_string = 'flyby_t' + str(round(S['time'][n], 1))
                S[flyby_string] = reinterpolate.flyby(output_dir, flyby_a, n, method=method)
                diag.save_dict(S, output_dir, dict_name)
                print(flyby_string + ' done')

    


def spec_hik_energy_frac(S, do_magnetic=1):
    k = S['grids']['Kmag']
    kmax = max(k)
    E = S['EM'] if do_magnetic else S['EK']
    hik_E = E[k > (kmax/6)].sum()
    tot_E = E.sum()
    return hik_E / tot_E
