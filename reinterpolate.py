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
          l_dir=np.array([1/np.sqrt(950.6), 1/np.sqrt(89.5), 1.]), norm_B0=1, method='matt',
          output_plot=1, set_direction=True, number_along_y=np.sqrt(90.234), number_along_z=100, total_length_y=100):
    # Loading data with the correct method scales the data to the physical variables
    data = diag.load_data(output_dir, flyby_n, prob='from_array', method=method)
    Ns, Ls = get_grid_info(data)
    Ls[:2] *= flyby_a  # stretch perpendicular lengths
    zg, yg, xg = reinterp_generate_grid(Ns, Ls)
    # Zg, Yg, Xg = np.meshgrid(zg, yg, xg, indexing='ij')  # only needed if defining a function on grid
    

    # extending to general mean fields
    rho_data = data['rho']
    B = np.array((data['Bcc1'], data['Bcc2'], data['Bcc3']))
    B_0_vector = diag.box_avg(B)
    B_0 = diag.get_mag(diag.box_avg(B), axis=0)  # single time entry
    v_A = diag.alfven_speed(rho_data, B)
    scale_B_mean = 1 / B_0 if norm_B0 else 1  # 1 / <Bx> = a^2
    scale_v_A = 1 / v_A if norm_B0 else 1  # rho^1/2 / <Bx> = rho^1/2 * a^2
    scale_rho = flyby_a**2 if norm_B0 else 1
    
    Bx = pad_array(data['Bcc1'] * scale_B_mean)
    By = pad_array(data['Bcc2'] * scale_B_mean)
    Bz = pad_array(data['Bcc3'] * scale_B_mean)
    ux = pad_array(data['vel1'] * scale_v_A)
    uy = pad_array(data['vel2'] * scale_v_A)
    uz = pad_array(data['vel3'] * scale_v_A)
    rho = pad_array(rho_data * scale_rho)
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
  
    # get unit vector of direction
  
    # if plotting, do a short flyby to cut down on space
    # otherwise fly through either 1/10 of the box 
    # or 5 million points (for large resolutions) for analysis
    N_points = Ns.prod()
    dl = yg[1] - yg[0] # walk in steps of dy = a*Ly / Ny
    if not set_direction:
        N_dl = 50000 if output_plot else min(int(5e6), N_points//10)
    else:
        alpha, beta = flyby_a/number_along_y, flyby_a/number_along_z
        dx = dl / np.sqrt(1 + alpha**2 + beta**2)
        l_dir = np.array([beta*dx, alpha*dx, dx])
        total_length_y *= Ls[1]  # number of times along y axis of box
        N_dl = int(total_length_y / l_dir[1])
    total_length = N_dl * dl
    lvec = np.linspace(0, total_length, N_dl, endpoint=False).reshape(N_dl, 1)

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
    FB['l_param'], FB['points'], FB['dl'] = lvec[:, 0], pts, dl
    FB['a'], FB['snapshot_number'] = flyby_a, flyby_n
    FB['normed_to_Bx'] = 'true' if norm_B0 else 'false'
    FB['norms'] = {'B': scale_B_mean, 'u': scale_v_A, 'rho': scale_rho, 'B_0_vec': B_0_vector}
    
    return FB

