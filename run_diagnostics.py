import numpy as np
from scipy.ndimage.measurements import label
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
             override_do_full_calc=0, do_cos=1, method='matt'):
    
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
            S['mean_cos2_theta'] = {'mean_field': np.array([]), 'radial': np.array([])}
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
            u_0 = diag.box_avg(u, reshape=1)
            b_0 = diag.get_unit(B_0)

            # Fluctuations = field - mean
            δB = B - B_0
            δu = u - u_0  # shouldn't be any mean velocity field but feel better doing this
            # only do every now and then as FFT is computaionally intensive
            if do_cos and n_start % spec_step == 0:
                print('         - Calculating <cos^2 θ> ')
                cos2_meanfield, cos2_radial = diag.mean_cos2(b_0, δB, a, output_dir)
                S['mean_cos2_theta']['mean_field'] = np.append(S['mean_cos2_theta']['mean_field'], cos2_meanfield)
                S['mean_cos2_theta']['radial'] = np.append(S['mean_cos2_theta']['radial'], cos2_radial)
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
        B_0 = (S['Bx_mean'][0], S['By_mean'][0]) # initial mean field (always in xy-plane)
        mesh_data = athinput_dict(athinput_path)['mesh']
        L_x, L_prp = mesh_data['x1max'], mesh_data['x2max']
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
                if spec_name not in S['spectra'].keys():
                    S['spectra'][spec_name] = spec.calc_spectrum(output_dir, output_dir, prob='from_array', dict_name=spec_name,
                                                    do_single_file=1, n=n, method=method)
            diag.save_dict(S, output_dir, dict_name)


def run_switchback_loop(output_dir, athinput_path, dict_name='data_dump', steps=1, method='matt', start_at=0, n_startat=0,
                        do_full_calc=True, do_flyby=True, do_flyby_clock=True):
    max_n = diag.get_maxn(output_dir)
    n_done = 0
    S = {}
    
    dict_name += '_sbdata'
    
    # expansion rate, sound speed from athinput
    expansion_rate, c_s_init, tlim, dt = read_athinput(athinput_path)
    S['expansion_rate'] = expansion_rate
    
    # step for non-continuous calculations
    a_step = 0.5  # analyse in steps of da = 0.5
    spec_step = int(a_step / (expansion_rate*dt))
    
    
    
    if diag.check_dict(output_dir, dict_name):
        S = diag.load_dict(output_dir, dict_name)
        n_done =  S['a'].size
        if n_done == (1 + (max_n // spec_step)):
            do_full_calc = False
            print('Not doing full calculation')
    
    if start_at:
        n_done = n_startat
    # overestimate the number of steps needed; edge case is handled when loading data
    n_steps = int(np.ceil((max_n - n_done) / steps))
    print('n done = ' + str(n_done))
    
    

    mesh_data = athinput_dict(athinput_path)['mesh']
    L_x, L_prp = mesh_data['x1max'], mesh_data['x2max']
    
        
    S['spec_step'] = spec_step
    
    # do over z = 0.125, 0.25, 0.375, 0.5, 0.625, 0.75
    # corresponds to theta_thresh = 41, 60, 75, 90, 104, 120 degrees (ish for z = 0.125, 0.375, 0.625; exact otherwise)
    # z = 0.5*(1-cos(theta))
    
    z_list = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75])
    
    if do_full_calc:
        if n_done == 0:
            S['time'], S['a'] = np.array([]), np.array([])
            # S['sb_data']['sb_frac_radial'] = np.array([])
            S['sb_data'] = {}
            for z in z_list:
                S['sb_data'][z] = {'pol_frac': {}}
            S['pol_frac_tot'] = {}

        print('max_n = ' + str(max_n))
        for i in range(n_steps):
            n_start = n_done + i*steps
            n_end = n_done + (i+1)*steps
            if n_start % spec_step == 0:
                print(' Analysing n = ' + str(n_start) + ' to ' + str(n_end-1))
                t, a, B, u, rho = diag.load_time_series(output_dir, n_start, n_end, method=method)
                u = None
                print('     Data loaded')
                
                S['time'] = np.append(S['time'], t)
                S['a'] = np.append(S['a'], a)
                s_name = 'a' + str(round(a[0],1))
                
                # --- SWITCHBACK DIAGNOSTICS --- #
                
                Ns = np.array(B.shape[2:])  # Nz, Ny, Nx
                N_cells = Ns.prod()
                Ls = np.array([L_prp, L_prp, L_x])  # Lz, Ly, Lx
                Ls[:2] *= a[0]
                
                B_0 = diag.box_avg(B, reshape=1) # mean field in box = Parker spiral
                b, b_0 = diag.get_unit(B), diag.get_unit(B_0)
                cos_theta = diag.dot_prod(b, b_0, 1)
                B0x, B0y = B_0[0, 0, 0, 0, 0], B_0[0, 1, 0, 0, 0]
                
                xi, xi_bins = diag.polarisation_fraction(B, rho)
                S['pol_frac_tot'][s_name] = {'xi_count': xi, 'xi_bins': xi_bins}  
                

                print('         - Calculating SB data')
                # loop over threshold normlised deflection
                for z_threshold in z_list:
                    print('             - z_thresh = ' + str(z_threshold))
                    print('                 - Doing switchback threshold')
                    sb_mask_dev, sb_frac_dev = diag.switchback_threshold(cos_theta, N_cells, z_threshold=z_threshold)
                    
                    # if n_start % (2*spec_step) == 0:  # do clock angle/labeling every da=1
                    #     print('                 - Doing switchback labels')
                    #     labels, nlabels, label_array, pos = diag.label_switchbacks(sb_mask_dev)
                    #     labels = labels[0]
                    #     # label_array = diag.sort_sb_by_size(labels, label_array)
                    theta_dict = S['sb_data'][z_threshold]
                    
                    xi, xi_bins = diag.polarisation_fraction(B, rho, inside_SB=1, SB_mask=sb_mask_dev)
                    theta_dict['pol_frac'][s_name] = {'xi_count': xi, 'xi_bins': xi_bins}
                    
                    for n in range(n_start,n_end):
                        if n == 0:
                            # --- SETUP --- #
                            # -- CLOCK ANGLE -- #
                            theta_dict['clock_angle'] = {}
                            theta_dict['clock_angle']['grid'] = np.linspace(-np.pi, np.pi, 51)
                            theta_dict['clock_angle']['bins'] = 0.5*(theta_dict['clock_angle']['grid'][1:] + theta_dict['clock_angle']['grid'][:-1])
                            theta_dict['clock_angle']['all'] = {}
                            # theta_dict['clock_angle']['all'], theta_dict['clock_angle']['mean'], theta_dict['clock_angle']['SB_info'] = {}, {}, {}
                            
                            # -- SB FRACTION -- #
                            theta_dict['full_sb_frac'] = np.array([])
                            # theta_dict['aspect'] = {}

                        print('                 - Doing clock angle')
                        sb_ca_temp = diag.clock_angle(B[0], (B0x, B0y), SB_mask=sb_mask_dev[0])
                        theta_dict['clock_angle']['all'][s_name] = sb_ca_temp['all_clock_angle_count']
                        
                        # if n_start % (2*spec_step) == 0:
                        #     print('                 - Doing clock angle')
                        #     sb_ca_temp = diag.clock_angle(B[0], (B0x, B0y), SB_mask=sb_mask_dev[0])
                        #     # sb_ca_temp = diag.clock_angle(B[0], (B0x, B0y), SB_mask=sb_mask_dev[0], label_tuple=(labels, nlabels, label_array, pos))
                            
                        #     # add individual count
                        #     theta_dict['clock_angle']['all'][s_name] = sb_ca_temp['all_clock_angle_count']
                        #     # theta_dict['clock_angle']['mean'][s_name] = sb_ca_temp['mean_clock_angle_count']
                        #     # theta_dict['clock_angle']['SB_info'][s_name] = sb_ca_temp['SB_info']

                        #     # # do PCA analysis
                        #     # print('                 - Doing PCA')
                        #     # theta_dict['aspect'][s_name] = diag.switchback_aspect((labels, nlabels, label_array), Ls, Ns)
                        
                        # add individual switchback fraction data
                        theta_dict['full_sb_frac'] = np.append(theta_dict['full_sb_frac'], sb_frac_dev)
                        
                        S['sb_data'][z_threshold] = theta_dict
                        diag.save_dict(S, output_dir, dict_name)
                        
                # total radial switchback fraction (f_{bx < 0})
                # SB_radial_flip = B[:, 0] <= 0.
                # sb_frac_radial = SB_radial_flip[0][SB_radial_flip[0]].size / N_cells
                # S['sb_data']['sb_frac_radial'] = np.append(S['sb_data']['sb_frac_radial'], sb_frac_radial)
                # SB_radial_flip = None
                # sb_frac_radial, sb_frac_dev, sb_mask_dev, sb_ca_temp = None, None, None, None
                sb_frac_dev, sb_mask_dev, sb_ca_temp = None, None, None
                
                # clear unneeded variables to save memory, run flyby code after this
                B, t, a = None, None, None
                print('     Data cleared')
                diag.save_dict(S, output_dir, dict_name)
                print('     Dictionary saved')
    
    if do_flyby:
        if not do_full_calc:
            S_sbdict = diag.load_dict(output_dir, dict_name.split('fb_and_dropouts_sbdata')[0]+'sbdata')
        S['flyby'] = {}
        S['flyby']['dropouts'], S['flyby']['sb_clock_angle'] = {}, {}
        
        for z in z_list:
            S['flyby']['sb_clock_angle'][z] = {}
        flyby_n = np.arange(0, max_n, step=spec_step)
        for i, n in enumerate(flyby_n):
            if n % spec_step == 0:
                print('Flyby started at n = ' + str(n))
                if expansion_rate != 0.0:
                    flyby_a = round(S_sbdict['a'][i], 1) if not do_full_calc else round(S['a'][i], 1)
                    flyby_string = 'flyby_a' + str(flyby_a)
                else:
                    flyby_a = 1.0
                    flyby_string = 'flyby_t' + str(round(S['time'][i], 1))
                S['flyby'][flyby_string] = reinterpolate.flyby(output_dir, flyby_a, n, method=method)
                diag.save_dict(S, output_dir, dict_name)
                print(flyby_string + ' done')
                print(' - Calculating flyby SBs')
                flyby = reinterpolate.flyby(output_dir, flyby_a, n, method=method, output_plot=0)
                
                if do_flyby_clock:
                    Bx, By, Bz, Bmag = flyby['Bx'], flyby['By'], flyby['Bz'], flyby['Bmag']
                    
                    # calculate either deviation from radial or Parker
                    B0x, B0y = (Bx.mean(), By.mean())
                    B0 = np.sqrt(B0x**2 + B0y**2)
                    cos_theta = (Bx*B0x+By*B0y) / (Bmag*B0)
                    N_cells = Bx.size
                    
                    for z_threshold in z_list:
                        theta_dict = S['flyby']['sb_clock_angle'][z_threshold]
                        # switchback finder
                        SB_mask = diag.switchback_threshold(cos_theta, N_cells, flyby=1, z_threshold=z_threshold)[0]
                        # flyby clock angle
                        clock_angle_dict = diag.clock_angle((Bx, By, Bz), (B0x, B0y), SB_mask=SB_mask, flyby=1)
                        if n == 0:
                            # set up bins and grid
                            # these will be the same for all runs
                            theta_dict['grid'] = np.linspace(-np.pi, np.pi, 51)
                            theta_dict['bins'] = 0.5*(theta_dict['grid'][1:] + theta_dict['grid'][:-1])
                            theta_dict['all'] = {}
                            # theta_dict['all'], theta_dict['mean'], theta_dict['SB_info'] = {}, {}, {}
                            
                        theta_dict['all'][flyby_string] = clock_angle_dict['all_clock_angle_count']
                        # theta_dict['mean'][flyby_string] = clock_angle_dict['mean_clock_angle_count']
                        # theta_dict['SB_info'][flyby_string] = clock_angle_dict['SB_info']
                        S['flyby']['sb_clock_angle'][z_threshold] = theta_dict
                        clock_angle_dict = None
                # farrell analysis
                c_s = c_s_init*flyby_a**(-2/3)
                if flyby_a > 1.0:
                    S['flyby']['dropouts'][flyby_string] = diag.dropouts(flyby, c_s)
                # S['flyby']['dropouts'][flyby_string] = diag.plot_dropouts(flyby, c_s)
                diag.save_dict(S, output_dir, dict_name)
                print(' - Dictionary saved')