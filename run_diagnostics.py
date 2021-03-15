import numpy as np
import diagnostics as diag


def read_athinput(athinput_path):
    ath = open(athinput_path, 'r')
    list_of_lines = ath.readlines()
    iso_sound_speed = float(list_of_lines[53].split('=')[1].split('#')[0])
    expand = bool(list_of_lines[66].split('=')[1].split('#')[0])
    expansion_rate = float(list_of_lines[67].split('=')[1].split('#')[0]) if expand else 0.
    return expansion_rate, iso_sound_speed



def run(output_dir, athinput_path):
    S = {}

    # expansion rate, sound speed from athinput
    expansion_rate, c_s_init = read_athinput(athinput_path)
    S['expansion_rate'] = expansion_rate

    t, B, u, rho = diag.load_time_series(output_dir)
    a = diag.a(expansion_rate, t)
    S['time'] = t
    S['perp_expand'] = a
    B, u = diag.expand_variables(a, B), diag.expand_variables(a, u)
    
    u_prp, B_prp = u[:, 1:], B[:, 1:]
    u_prp2, B_prp2 = diag.dot_prod(u_prp, u_prp, 1), diag.dot_prod(B_prp, B_prp, 1)
    B_x = B[:, 0]
    B0_x = B_x[0, 0, 0, 0]
    B_mag = np.sqrt(diag.dot_prod(B, B, 1))

    S['beta'] = diag.beta(rho, B_mag, c_s_init, expansion_rate, t)
    S['sb_frac'] = diag.switchback_fraction(B_x, B_mag, B0_x)
    S['cross_helicity'] = diag.cross_helicity(rho, u_prp, B_prp)
    S['norm_fluc_Bprp'] = diag.norm_fluc_amp(B_prp2, B_x)
    S['norm_fluc_uprp'] = diag.norm_fluc_amp(rho*u_prp2, B_x)
    S['C_B2_Squire'] = diag.mag_compress_Squire2020(B)
    S['C_B2_Shoda'] = diag.mag_compress_Shoda2021(B)

    diag.save_dict(S, output_dir, 'data_dump')