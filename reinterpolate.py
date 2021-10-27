import numpy as np
from numpy.random import uniform
import diagnostics as diag
from helper_functions import pad_array, pad_grid, reinterp_generate_grid, get_grid_info
from scipy.interpolate import RegularGridInterpolator as rgi


def reinterp_to_grid(data, old_Xgrid, new_Ns, Ls):
    xg, yg, zg = old_Xgrid
    xg, yg, zg = pad_grid(xg), pad_grid(yg), pad_grid(zg)
    data_interp = rgi((zg, yg, xg), pad_array(data))

    Zg_hires, Yg_hires, Xg_hires = reinterp_generate_grid(new_Ns, Ls, return_mesh=1, pad=0)
    pts = np.array([Zg_hires.ravel(), Yg_hires.ravel(), Xg_hires.ravel()]).T
    return data_interp(pts).reshape(*new_Ns)

def flyby(output_dir, flyby_a, flyby_n, do_rand_start=1, l_start=None,
          l_dir=np.array([np.pi/8, np.sqrt(0.5), 1.]), norm_B0=1, method='matt',
          output_plot=1):
    # Loading data with the correct method scales the data to the physical variables
    data = diag.load_data(output_dir, flyby_n, prob='from_array', method=method)
    Ns, Ls = get_grid_info(data)
    
    zg, yg, xg = reinterp_generate_grid(Ns, Ls)
    # Zg, Yg, Xg = np.meshgrid(zg, yg, xg, indexing='ij')  # only needed if defining a function on grid
    

    # extending to general mean fields
    rho_data = data['rho']
    B = np.array((data['Bcc1'], data['Bcc2'], data['Bcc3']))
    B_0 = diag.get_mag(diag.box_avg(B), axis=0)  # single time entry
    v_A = diag.alfven_speed(rho_data, B)
    scale_B_mean = 1 / B_0 if norm_B0 else 1  # 1 / <Bx> = a^2
    scale_v_A = 1 / v_A if norm_B0 else 1  # rho^1/2 / <Bx> = rho^1/2 * a^2

    Bx = pad_array(data['Bcc1'] * scale_B_mean)
    By = pad_array(data['Bcc2'] * scale_B_mean)
    Bz = pad_array(data['Bcc3'] * scale_B_mean)
    ux = pad_array(data['vel1'] * scale_v_A)
    uy = pad_array(data['vel2'] * scale_v_A)
    uz = pad_array(data['vel3'] * scale_v_A)
    rho = pad_array(rho_data) *flyby_a**2
    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

    # interpolaters
    Bx_i = rgi((zg, yg, xg), Bx)
    By_i = rgi((zg, yg, xg), By)
    Bz_i = rgi((zg, yg, xg), Bz)    
    ux_i = rgi((zg, yg, xg), ux)
    uy_i = rgi((zg, yg, xg), uy)
    uz_i = rgi((zg, yg, xg), uz)
    Bmag_i = rgi((zg, yg, xg), Bmag)
    rho_i = rgi((zg, yg, xg), rho)

    # N_linepts = 4*np.max(Ns)
    # N_loop = 25  # number of times along an axis?
    # lvec = np.linspace(0, N_loop, N_linepts).reshape(N_linepts, 1)    
    N_y = Ns[1]
    # if plotting, do a short flyby to cut down on space
    # otherwise flythrough most of the box and analyse
    total_length = N_y if output_plot else 100*N_y
    dl = flyby_a*(yg[1] - yg[0])
    N_dl = int(total_length / dl)
    lvec = np.linspace(-total_length/2, total_length/2, N_dl).reshape(N_dl, 1)

    if do_rand_start:
        # start at random point in box
        l_start = uniform(high=Ls)
    else:
        assert l_start is not None, 'l_start must be a valid numpy array!'
    # direction biased in x direction (z, y, x)
    l_dir /= np.sqrt(np.sum(l_dir**2))
    pts = np.mod(l_start + lvec*l_dir, Ls)

    # Interpolate data along line running through box
    FB = {}
    FB['Bx'], FB['By'], FB['Bz'] = Bx_i(pts), By_i(pts), Bz_i(pts)
    FB['ux'], FB['uy'], FB['uz'] = ux_i(pts), uy_i(pts), uz_i(pts)
    FB['Bmag'], FB['rho'] = Bmag_i(pts), rho_i(pts)
    FB['start_point'], FB['direction'] = l_start, l_dir
    FB['l_param'], FB['points'] = lvec, pts
    FB['a'], FB['snapshot_number'] = flyby_a, flyby_n
    FB['normed_to_Bx'] = 'true' if norm_B0 else 'false'

    return FB

