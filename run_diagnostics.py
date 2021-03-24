from h5py import run_tests
import numpy as np
import diagnostics as diag
import spacecraft_flyby as spacecraft


tests = ['beta', 'sb_frac', 'cross_helicity', 'norm_fluc_Bprp',
         'C_B2_Squire', 'flyby']

def read_athinput(athinput_path):
    ath = open(athinput_path, 'r')
    list_of_lines = ath.readlines()
    tlim = float(list_of_lines[10].split('=')[1].split('#')[0])
    dt = float(list_of_lines[22].split('=')[1].split('#')[0])
    iso_sound_speed = float(list_of_lines[53].split('=')[1].split('#')[0])
    expand = bool(list_of_lines[66].split('=')[1].split('#')[0])
    expansion_rate = float(list_of_lines[67].split('=')[1].split('#')[0]) if expand else 0.
    return expansion_rate, iso_sound_speed, tlim, dt
 

def run(output_dir, athinput_path, flyby_do_last=1,  flyby_n=0):
    
    dict_is_there = diag.check_dict(output_dir, 'data_dump')
    if dict_is_there:
        S = diag.load_dict(output_dir, 'data_dump')
    else:
        S = {}
    run_tests = [test not in list(S.keys()) for test in tests]

    # expansion rate, sound speed from athinput
    expansion_rate, c_s_init, tlim, dt = read_athinput(athinput_path)
    S['expansion_rate'] = expansion_rate
    S['sound_speed'] = c_s_init
    
    if not dict_is_there:
        t, B, u, rho = diag.load_time_series(output_dir)
    else:
        N_t = int(tlim / dt)
        t = np.arange(0, tlim+dt, N_t)
    a = diag.a(expansion_rate, t)
    S['time'] = t
    S['perp_expand'] = a
    
    if not dict_is_there:
        B, u = diag.expand_variables(a, B), diag.expand_variables(a, u)
        u_prp, B_prp = u[:, 1:], B[:, 1:]
        B_x = B[:, 0]
        S['Bx_mean'] = diag.box_avg(B_x)
        B0_x = B_x[0, 0, 0, 0]
        B_mag = np.sqrt(diag.dot_prod(B, B, 1))

    # a for loop would be nice here but I don't know how to
    # change functions in a loop 
    if run_tests[0]:
        S['beta'] = diag.beta(rho, B_mag, c_s_init, expansion_rate, t)
        print('beta done')
    if run_tests[1]:
        S['sb_frac'] = diag.switchback_fraction(B_x, B_mag, B0_x)
        print('sb frac done')
    if run_tests[2]:
        S['cross_helicity'] = diag.cross_helicity(rho, u_prp, B_prp)
        print('cross helicity done')
    if run_tests[3]:
        S['norm_fluc_Bprp'] = diag.norm_fluc_amp(diag.dot_prod(B_prp, B_prp, 1), B_x)
        S['norm_fluc_uprp'] = diag.norm_fluc_amp(rho*diag.dot_prod(u_prp, u_prp, 1), B_x)
        print('fluctuations done')
    if run_tests[4]:
        S['C_B2_Squire'] = diag.mag_compress_Squire2020(B)
        S['C_B2_Shoda'] = diag.mag_compress_Shoda2021(B)
        print('compressibility done')

    if not dict_is_there:
        # clear unneeded variables to save memory, run flyby code after this
        B, u, B_prp, u_prp, B_x, B_mag = None, None, None, None, None

    # automatically starts at a random point
    # in the last snapshot.
    if 'flyby' in S.keys():
        S.pop('flyby')
    S['flyby'] = spacecraft.flyby(output_dir, a, do_last=flyby_do_last, n=flyby_n)
    print('flyby done')

    diag.save_dict(S, output_dir, 'data_dump')