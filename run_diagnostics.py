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
 

def run_loop(output_dir, athinput_path, dict_name='data_dump', steps=1, do_spectrum=0, 
             override_do_full_calc=0, method='matt'):
    
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

    if override_do_full_calc:
        do_full_calc = not override_do_full_calc
        print('Overriding full calculation')
    

    # overestimate the number of steps needed; edge case is handled when loading data
    n_steps = int(np.ceil((max_n - n_done) / steps))
    print('n done = ' + str(n_done))
    
    # expansion rate, sound speed from athinput
    expansion_rate, c_s_init, tlim, dt = read_athinput(athinput_path)
    S['expansion_rate'] = expansion_rate
    S['sound_speed'] = c_s_init
    
    # step for non-continuous calculations such as cos^2 theta
    a_max = 1 + expansion_rate*(max_n-1)*dt
    a_step = 1 if (a_max > 2) else 0.1
    if expansion_rate == 0:
        spec_step = int(1 / dt) if max_n > 10 else 1
    else:
        spec_step = int(a_step / (expansion_rate*dt))  # eg if delta_a = 1, adot=0.5, dt=0.2 then spec_step = 10
        
    S['spec_step'] = spec_step
    
    if do_full_calc:
        if n_done == 0:
            S['time'], S['a'] = np.array([]), np.array([])
            S['Bx_mean'], S['By_mean'], S['beta'] = np.array([]), np.array([]), np.array([])
            S['alfven_speed'] = np.array([])
            S['mean_cos2_theta'] = {'box': np.array([]), 'field': np.array([]), 'energy_weight': np.array([]), 'no2D_energy_weight': np.array([])}
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
            B_mag2 = diag.get_mag(B, squared=1)  # full field magnitude
            S['beta'] = np.append(S['beta'], diag.beta(rho, B_mag2, c_s_init, a))
            t, B_mag2 = None, None
            
            print('         - Calculating mean field')
            B_x, B_y = B[:, 0], B[:, 1]
            S['Bx_mean'] = np.append(S['Bx_mean'], diag.box_avg(B_x))
            S['By_mean'] = np.append(S['By_mean'], diag.box_avg(B_y))
            B_x, B_y = None, None
            
            v_A = diag.alfven_speed(rho, B)
            S['alfven_speed'] = np.append(S['alfven_speed'], v_A)
            B_0 = diag.box_avg(B, reshape=1) # mean field
            b_0 = diag.get_unit(B_0)
            
            # Fluctuations = field - mean
            δB = B - B_0
            δu = u
            # only do every now and then as FFT is computaionally intensive
            if n_start % spec_step == 0:
                print('         - Calculating <cos^2 θ> ')
                cos2_box, cos2_field, cos2_energy_weight, cos2_energyweight_no2D = diag.mean_cos2(b_0, δB, a, output_dir)
                S['mean_cos2_theta']['box'] = np.append(S['mean_cos2_theta']['box'], cos2_box)
                S['mean_cos2_theta']['field'] = np.append(S['mean_cos2_theta']['field'], cos2_field)
                S['mean_cos2_theta']['energy_weight'] = np.append(S['mean_cos2_theta']['energy_weight'], cos2_energy_weight)
                S['mean_cos2_theta']['no2D_energy_weight'] = np.append(S['mean_cos2_theta']['no2D_energy_weight'], cos2_energyweight_no2D)
            b_0, u = None, None
            
            print('         - Calculating z+/-')
            z_p_rms, z_m_rms = diag.z_waves_evo(rho, δu, δB, v_A)
            S['z_plus'] = np.append(S['z_plus'], z_p_rms)
            S['z_minus'] = np.append(S['z_minus'], z_m_rms)
            z_p_rms, z_m_rms = None, None
            
            print('         - Calculating cross helicity')
            S['cross_helicity'] = np.append(S['cross_helicity'], diag.cross_helicity(rho, δu, δB))
            rho, δB, δu, v_A = None, None, None, None
            a = None
            
            print('         - Calculating magnetic compressibility')
            S['C_B2_Squire'] = np.append(S['C_B2_Squire'], diag.mag_compress_Squire2020(B))
            
            # clear unneeded variables to save memory, run flyby code after this
            B = None 
            print('     Data cleared')
            diag.save_dict(S, output_dir, dict_name)
            print('     Dictionary saved')
        
        print('Calculating amplitude evolution')
        B = diag.load_time_series(output_dir, 0, 1, method=method)[2]
        B_0 = diag.box_avg(B)[0, :2]  # initial mean field (always in xy-plane)
        B = None
        L_x, L_prp = diag.get_lengths(output_dir=output_dir)[:2]
        a_normfluc, Bprp_fluc, kinetic_fluc = diag.norm_fluc_amp_hst(output_dir, expansion_rate, B_0,
                                                                     Lx=L_x, Lperp=L_prp, method=method)
        S['a_norm_fluc'] = a_normfluc
        S['Bprp_norm_fluc'] = Bprp_fluc
        S['kinetic_norm_fluc'] = kinetic_fluc

        diag.save_dict(S, output_dir, dict_name)

    
    if do_spectrum:
        S['spectra'] = {}
        for n in range(max_n):
            if n % spec_step == 0:  # don't want to run too often
                print('Spectrum calculation started at n = ' + str(n))
                if expansion_rate != 0.0:
                    spec_name = 'mhd_spec_a' + str(round(S['a'][n], 1))
                else:
                    spec_name = 'mhd_spec_t' + str(round(S['time'][n], 1))
                S['spectra'][spec_name] = spec.calc_spectrum(output_dir, output_dir, prob='from_array', dict_name=spec_name,
                                                  do_single_file=1, n=n, method=method)
            diag.save_dict(S, output_dir, dict_name)


def run_switchback_loop(output_dir, athinput_path, dict_name='data_dump', steps=1, method='matt'):
    max_n = diag.get_maxn(output_dir)
    n_done = 0
    S = {}
    do_full_calc = True
    
    dict_name += '_sbdata'
    
    # expansion rate, sound speed from athinput
    expansion_rate, c_s_init, tlim, dt = read_athinput(athinput_path)
    S['expansion_rate'] = expansion_rate
    
    # step for non-continuous calculations
    a_step = 0.5  # analyse in steps of da = 0.5
    spec_step = int(a_step / (expansion_rate*dt))
    
    
    
    if diag.check_dict(output_dir, dict_name):
        S = diag.load_dict(output_dir, dict_name)
        n_done = S['a'].size
        if n_done == (1 + max_n // spec_step):
            do_full_calc = False
            print('Not doing full calculation')
            
    # overestimate the number of steps needed; edge case is handled when loading data
    n_steps = int(np.ceil((max_n - n_done) / steps))
    print('n done = ' + str(n_done))
    
    

    L_x, L_prp = diag.get_lengths(output_dir=output_dir)[:2]
    
        
    S['spec_step'] = spec_step
    
    if do_full_calc:
        if n_done == 0:
            S['time'], S['a'] = np.array([]), np.array([])
            S['sb_data'] = {60: {}, 90: {}, 120: {}, 150:{}, 'sb_frac_radial': np.array([])}

        print('max_n = ' + str(max_n))
        for i in range(n_steps):
            n_start = n_done + i*steps
            n_end = n_done + (i+1)*steps
            if n_start % spec_step == 0:
                print(' Analysing n = ' + str(n_start) + ' to ' + str(n_end-1))
                t, a, B, u, rho = diag.load_time_series(output_dir, n_start, n_end, method=method)
                u, rho = None, None
                print('     Data loaded')
                
                S['time'] = np.append(S['time'], t)
                S['a'] = np.append(S['a'], a)
                
                # --- SWITCHBACK DIAGNOSTICS --- #
                
                Ns = np.array(B.shape[2:])  # Nz, Ny, Nx
                Ls = np.array([L_prp, L_prp, L_x])  # Lz, Ly, Lx
                Ls[:2] *= a[0]

                print('         - Calculating SB data')
                # loop over threshold angles
                for theta_threshold in [60, 90, 120]:
                    print('             - θ_thresh = ' + str(theta_threshold) + '∘')
                    sb_mask_dev, sb_mask_devflip, sb_frac_dev, sb_frac_radial, sb_frac_devflip = diag.switchback_threshold(B, theta_threshold=theta_threshold)
                    theta_dict = S['sb_data'][theta_threshold]
                    for n in range(n_start,n_end):
                        t_index = n - n_start
                        sb_ca_temp = diag.clock_angle(B[t_index:t_index+1], sb_mask_dev[t_index:t_index+1])
                        if n == 0:
                            # --- SETUP --- #
                            # -- CLOCK ANGLE -- #
                            theta_dict['clock_angle'] = {}
                            theta_dict['clock_angle']['grid'] = sb_ca_temp['grid']
                            theta_dict['clock_angle']['bins'] = sb_ca_temp['bins']
                            theta_dict['clock_angle']['clock_angle_count'] = np.zeros_like(sb_ca_temp['grid'])
                            
                            # -- SB FRACTION -- #
                            theta_dict['full_sb_frac'] = np.array([])
                            theta_dict['radial_sb_frac'] = np.array([])
                            theta_dict['aspect_full'] = {}
                            theta_dict['aspect_radial_flip'] = {}

                        # increment total count
                        theta_dict['clock_angle']['clock_angle_count'] += sb_ca_temp['clock_angle_count']
                        
                        # add individual count
                        s_name = str(n)
                        theta_dict['clock_angle'][s_name] = sb_ca_temp['clock_angle_count']
                        
                        # add individual switchback fraction data
                        theta_dict['full_sb_frac'] = np.append(theta_dict['full_sb_frac'], sb_frac_dev)
                        theta_dict['radial_sb_frac'] = np.append(theta_dict['radial_sb_frac'], sb_frac_devflip)
                        
                        # do PCA analysis
                        theta_dict['aspect_full'][s_name] = diag.switchback_aspect(sb_mask_dev, Ls, Ns)
                        theta_dict['aspect_radial_flip'][s_name] = diag.switchback_aspect(sb_mask_devflip, Ls, Ns)
                        
                        S['sb_data'][theta_threshold] = theta_dict
                        
                # total radial switchback fraction (f_{bx < 0})
                S['sb_data']['sb_frac_radial'] = np.append(S['sb_data']['sb_frac_radial'], sb_frac_radial)
                sb_frac_radial, sb_frac_dev, sb_mask_dev, sb_mask_devflip, sb_ca_temp = None, None, None, None, None
                
                # clear unneeded variables to save memory, run flyby code after this
                B, t, a = None, None, None
                print('     Data cleared')
                diag.save_dict(S, output_dir, dict_name)
                print('     Dictionary saved')
    
    S['flyby'] = {}
    S['flyby']['sb_clock_angle'], S['flyby']['dropouts'] = {30: {}, 60: {}, 90: {}}, {}
    flyby_n = np.arange(0, max_n, step=spec_step)
    for i, n in enumerate(flyby_n):
        if n % spec_step == 0:
            print('Flyby started at n = ' + str(n))
            if expansion_rate != 0.0:
                flyby_a = round(S['a'][i], 1)
                flyby_string = 'flyby_a' + str(flyby_a)
            else:
                flyby_a = 1.0
                flyby_string = 'flyby_t' + str(round(S['time'][i], 1))
            S['flyby'][flyby_string] = reinterpolate.flyby(output_dir, flyby_a, n, method=method)
            diag.save_dict(S, output_dir, dict_name)
            print(flyby_string + ' done')
            
            print(' - Calculating flyby SBs')
            
            flyby = reinterpolate.flyby(output_dir, flyby_a, n, method=method, output_plot=0)
            Bx, By, Bz, Bmag = flyby['Bx'], flyby['By'], flyby['Bz'], flyby['Bmag']
            
            for theta_threshold in [60, 90, 120]:
                theta_dict = S['flyby']['sb_clock_angle'][theta_threshold]
                # switchback finder
                SB_mask = diag.switchback_threshold((Bx, By, Bmag), flyby=1, theta_threshold=theta_threshold)[0]
                # flyby clock angle
                clock_angle_dict = diag.clock_angle((Bx, By, Bz), SB_mask, flyby=1)
                if n == 0:
                    # set up bins and grid
                    # these will be the same for all runs
                    theta_dict['grid'] = clock_angle_dict['grid']
                    theta_dict['bins'] = clock_angle_dict['bins']
                    theta_dict['clock_angle_count'] = np.zeros_like(clock_angle_dict['grid'])
                # increment count otherwise
                theta_dict['clock_angle_count'] += clock_angle_dict['clock_angle_count']
                # add individual count
                theta_dict[flyby_string] = clock_angle_dict['clock_angle_count']
                S['flyby']['sb_clock_angle'][theta_threshold] = theta_dict
                clock_angle_dict = None
            # farrell analysis
            S['flyby']['dropouts'][flyby_string] = diag.plot_dropouts(flyby)
            diag.save_dict(S, output_dir, dict_name)
            print(' - Dictionary saved')