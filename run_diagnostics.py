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
 

def run_loop(output_dir, athinput_path, dict_name='data_dump', steps=10, do_spectrum=0, do_flyby=1, 
             method='matt', theta_threshold=90):
    
    max_n = diag.get_maxn(output_dir)
    n_done = 0
    S = {}
    do_full_calc = True
    
    if diag.check_dict(output_dir, dict_name):
        S = diag.load_dict(output_dir, dict_name)
        n_done = S['a'].size
        if n_done == max_n:
            do_full_calc = False
            print('Not doing full calculation')

    # overestimate the number of steps needed; edge case is handled when loading data
    n_steps = int(np.ceil((max_n - n_done) / steps))
    print('n done = ' + str(n_done))
    
    # expansion rate, sound speed from athinput
    expansion_rate, c_s_init, tlim, dt = read_athinput(athinput_path)
    S['expansion_rate'] = expansion_rate
    S['sound_speed'] = c_s_init

    L_x, L_prp = diag.get_lengths(output_dir=output_dir)[:2]
    
    if do_full_calc:
        if n_done == 0:
            S['time'], S['a'] = np.array([]), np.array([])
            S['Bx_mean'], S['By_mean'], S['beta'] = np.array([]), np.array([]), np.array([])
            S['sb_frac'], S['alfven_speed'] = np.array([]), np.array([])
            S['sb_clock_angle'] = {}
            S['cross_helicity'], S['z_plus'], S['z_minus'] = np.array([]), np.array([]), np.array([])
            S['C_B2_Squire'] = np.array([])

        print('max_n = ' + str(max_n))
        for i in range(n_steps):
            n_start = n_done + i*steps
            n_end = n_done + (i+1)*steps
            print(' Analysing n = ' + str(n_start) + ' to ' + str(n_end-1))
            t, a, B, u, rho = diag.load_time_series(output_dir, n_start, n_end, method=method)
            print('     Data loaded')
            
            S['time'] = np.append(S['time'], t)
            S['a'] = np.append(S['a'], a)
            
            print('         - Calculating beta')
            B_mag2 = diag.get_mag(B)**2  # full field magnitude
            S['beta'] = np.append(S['beta'], diag.beta(rho, B_mag2, c_s_init, a))
            B_mag2 = None
            
            print('         - Calculating mean field')
            B_x, B_y = B[:, 0], B[:, 1]
            S['Bx_mean'] = np.append(S['Bx_mean'], diag.box_avg(B_x))
            S['By_mean'] = np.append(S['By_mean'], diag.box_avg(B_y))
            B_x, B_y = None, None
            
            v_A = diag.alfven_speed(rho, B)
            S['alfven_speed'] = np.append(S['alfven_speed'], v_A)
            b_0 = diag.get_unit(diag.box_avg(B, reshape=1)) # mean field
            
            # Finding perpendicular fluctuations
            B_prp = B - diag.dot_prod(B, b_0)*b_0
            u_prp = u - diag.dot_prod(u, b_0)*b_0
            b_0 = None
            
            print('         - Calculating z+/-')
            z_p_rms, z_m_rms = diag.z_waves_evo(rho, u_prp, B_prp, v_A)
            S['z_plus'] = np.append(S['z_plus'], z_p_rms)
            S['z_minus'] = np.append(S['z_minus'], z_m_rms)
            z_p_rms, z_m_rms = None, None
            
            print('         - Calculating cross helicity')
            S['cross_helicity'] = np.append(S['cross_helicity'], diag.cross_helicity(rho, u_prp, B_prp))
            rho, B_prp, u_prp, v_A = None, None, None, None
            
            print('         - Calculating SB fraction')
            sb_mask, sb_frac = diag.switchback_finder(B, theta_threshold=theta_threshold)
            sb_ca_temp = diag.clock_angle(B, sb_mask)
            if i == 0:
                # set up bins and grid
                # these will be the same for all runs
                S['sb_clock_angle'] = sb_ca_temp
            else:
                # increment count otherwise
                S['sb_clock_angle']['clock_angle_count'] += sb_ca_temp['clock_angle_count']
            S['sb_frac'] = np.append(S['sb_frac'], sb_frac)
            sb_frac, sb_mask, sb_ca_temp = None, None, None
            
            print('         - Calculating magnetic compressibility')
            S['C_B2_Squire'] = np.append(S['C_B2_Squire'], diag.mag_compress_Squire2020(B))
            
            # clear unneeded variables to save memory, run flyby code after this
            t, a, B, u = None, None, None, None 
            print('     Data cleared')
            diag.save_dict(S, output_dir, dict_name)
            print('     Dictionary saved')
        
        print('Calculating amplitude evolution')
        B = diag.load_time_series(output_dir, 0, 1, method=method)[2]
        B_0 = diag.box_avg(B)[0, :2]
        B = None
        a_normfluc, Bprp_fluc, kinetic_fluc = diag.norm_fluc_amp_hst(output_dir, expansion_rate, B_0,
                                                                     Lx=L_x, Lperp=L_prp, method=method)
        S['a_norm_fluc'] = a_normfluc
        S['Bprp_norm_fluc'] = Bprp_fluc
        S['kinetic_norm_fluc'] = kinetic_fluc

        diag.save_dict(S, output_dir, dict_name)

    a_max = 1 + expansion_rate*(max_n-1)*dt
    a_step = 1 if (a_max > 2) else 0.1
    if expansion_rate == 0:
        spec_step = int(1 / dt) if max_n > 10 else 1
    else:
        spec_step = int(a_step / (expansion_rate*dt))  # eg if delta_a = 1, adot=0.5, dt=0.2 then spec_step = 10
    if do_spectrum:
        for n in range(max_n):
            if n % spec_step == 0:  # don't want to run too often
                print('Spectrum calculation started at n = ' + str(n))
                if expansion_rate != 0.0:
                    spec_a = round(S['a'][n], 1)
                    spec_name = 'mhd_spec_a' + str(spec_a)
                else:
                    spec_a = 1.0
                    spec_name = 'mhd_spec_t' + str(round(S['time'][n], 1))
                S[spec_name] = spec.calc_spectrum(output_dir, output_dir, prob='from_array', dict_name=spec_name,
                                                  do_single_file=1, n=n, a=spec_a, method=method)
        diag.save_dict(S, output_dir, dict_name)

    if do_flyby:
        for n in range(max_n):
            if n % spec_step == 0:
                print('Flyby started at n = ' + str(n))
                if expansion_rate != 0.0:
                    flyby_a = round(S['a'][n], 1)
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
