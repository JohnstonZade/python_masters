from athena_read import hst
import diagnostics as diag
import numpy as np


def get_fluc_energy(output_dir, adot, B_0, 
                    vol_norm=0, rho_0=1.0, Lx=1.0, prob='', method='matt'):
    hstData = diag.load_hst(output_dir, adot, prob, method=method)
    t = hstData['time'] 
    a = hstData['a']
    # Mean field and Alfvén velocity
    Bx0, By0 = B_0
    γ = By0 / Bx0  # initial By0/Bx0 fraction (for Parker spiral)
    v_A = (Bx0 / np.sqrt(rho_0)) * a**-1 * np.sqrt(1. + (γ*a)**2)
    t_A = Lx / v_A  # Lx = 1 always
    hstData['time_alfven_units'] = t / t_A
    
    # Volume to calculate energy density
    vol = diag.get_vol(output_dir, prob) if vol_norm else 1
    
    # Mean magnetic field evolution
    Bx_mean, By_mean = Bx0*a**-2, By0*a**-1
    mean_ME_x, mean_ME_y = 0.5*Bx_mean**2, 0.5*By_mean**2
    
    # no mean flows as of yet, removing mean field contribution
    KEprp_comp = np.array([hstData['1-KE'], hstData['2-KE'], hstData['3-KE']]) / vol
    MEprp_comp = np.array([(hstData['1-ME'] - mean_ME_x), (hstData['2-ME'] - mean_ME_y), hstData['3-ME']]) / vol

    # Total fluc energy is the sum of the individual components
    KEprp = KEprp_comp.sum(axis=0)
    MEprp = MEprp_comp.sum(axis=0)
    
    # scale by B_0**2
    mean_ME = mean_ME_x + mean_ME_y
    KEprp /= mean_ME
    MEprp /= mean_ME
    
    return a, MEprp, KEprp
