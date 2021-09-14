from athena_read import hst
import diagnostics as diag
import numpy as np


def get_energy_data(output_dir, adot,
                    vol_norm=0,
                    B0=1.0,
                    Lx=1.0,
                    prob='',
                    method='matt'):
    hstData = diag.load_hst(output_dir, adot, prob, method=method)

    # Volume to calculate energy density
    vol = diag.get_vol(output_dir, prob) if vol_norm else 1
    t_A = Lx  # true when v_A = 1 and B0 along the x axis
    t = hstData['time'] / t_A  # scale by Alfven period
    a = hstData['a']
    EKprp = np.array([hstData['2-KE'], hstData['3-KE']]) / vol
    EBprp = np.array([hstData['2-ME'], hstData['3-ME']]) / vol

    return t, a, EKprp.sum(axis=0), EBprp.sum(axis=0), hstData['1-ME'][0]
