'''Code to calculate structure function of a fluid simulation.'''
import numpy as np
from numpy.random import randint, random
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import rc
import diagnostics as diag
rc('text', usetex=True)  # LaTeX labels
default_prob = diag.DEFAULT_PROB

# --- HELPER FUNCTIONS --- #

def generate_points(grid, N):
    '''Generate N pairs of random points on the grid. Biased towards generating
    points that are closer together.
    '''
    N_p = int(N)
    L1 = randint(0, grid, size=(N_p, len(grid)))
    # Generate second set of random points to bias closer to points in L1
    # Higher values of r_pow give a smaller bias
    r_pow = 6
    L2 = L1 + np.ceil(grid*random(size=(N_p, len(grid)))**r_pow).astype(int)
    # Ensure points are in the grid
    L2 = np.mod(L2, grid)
    return L1, L2


def select_y(x, x_bin, y, i, mask=[], use_mask=0, return_mask=0):
    '''Selects the y vector values with the condition that
    the corresponding x values satisfy x_bin[i] <= x < x_bin[i+1].

    If a mask is provided, it is used on both x and y so that the order of
    the elements is unchanged when selecting based on x_bin.
    '''

    if use_mask:
        x = x[mask]
        y = y[mask]
    assert i < len(x_bin)-1, 'Can\'t access element at len(x_bin)'
    x_mask = np.logical_and(x_bin[i] <= x, x < x_bin[i+1])

    return x_mask if return_mask else y[x_mask]


def get_mean(x, x_bin, y, i, mask=[], use_mask=0):
    '''Calculate the mean of the selection. Mask parameters are only used
    for the select_y() call.
    '''
    y_sel = select_y(x, x_bin, y, i, mask, use_mask)

    # Stops error for mean of empty slice
    # Might be a better way to handle this
    if len(y_sel) == 0:
        return float('nan')

    return np.mean(y_sel)


def get_l_perp(L1, L2, l, B):
    '''Finds the lengths of vectors whose angle to the magnetic field is
    within a specified range. Essentially does the following:
    For a distribution of points L1, L2
        Gets the magnetic field vectors at L1 and L2
        Takes the average of the two to get the local mean field.
    Gets l and B unit vectors and then calculates cos(theta) = l dot b
    Bins l vectors with theta = [0, 15, 30, 45, 60, 75, 90]
        0-15 is l_parallel, 45-90 is l_perp
    Outputs angle list and mask for the l vector in order for the
    structure function to be calculated.
    '''
    # Calculate average B field between point pairs
    B1_vec = diag.get_vec(B, L1)
    B2_vec = diag.get_vec(B, L2)
    B_mean = 0.5*(B1_vec + B2_vec)

    # Dot product of unit vectors to get cos(θ)
    # cθ = np.sum(diag.get_unit(B_mean)*diag.get_unit(l), axis=1).clip(-1., 1.)
    # θ_data = np.arccos(cθ)
    # θ_mask = θ_data > np.pi / 2
    # θ_data[θ_mask] = np.pi - θ_data[θ_mask]

    cθ = abs(np.sum(diag.get_unit(B_mean)*diag.get_unit(l), axis=1)).clip(0., 1.)
    θ_data = np.arccos(cθ)

    θ = np.array([0, 15, 75, 90])
    θlen = len(θ) - 1
    θ_rad = (np.pi/180)*θ

    # Create l_mask depending on θ
    l_mask = [select_y(θ_data, θ_rad, diag.get_mag(l), i, return_mask=1)
              for i in range(θlen)]
    return θ, l_mask


def calc_struct(L1, L2, v, l_mag, L_min, mask=[], use_mask=0):
    # For each pair of position vectors x1 ∈ L1, x2 ∈ L2
    # Get vectors v2, v1 at each point
    # Calculate Δv2 = abs(v2 - v1)**2
    # We now have a mapping of l to Δv2 <- structure function
    v1_vec = diag.get_vec(v, L1)
    v2_vec = diag.get_vec(v, L2)
    Δv_vec = v2_vec - v1_vec
    Δv_mag2 = diag.get_mag(Δv_vec)**2

    # Bin and plot structure function
    # Plot in the middle of the bin points otherwise the size of arrays
    # won't match up.
    N_l = 30 # 15
    l_bin = np.logspace(np.log10(2*L_min/N_l), np.log10(L_min), N_l+1)
    l_grid = 0.5*(l_bin[:-1] + l_bin[1:])
    Δv_avg = np.array([get_mean(l_mag, l_bin, Δv_mag2, i, mask, use_mask)
                       for i in range(N_l)])

    return l_grid, Δv_avg

def get_spectral_slope(kgrid, spectrum, inertial_range):

    mask = (inertial_range[0] <= kgrid) & (kgrid <= inertial_range[1])
    # while np.all(np.logical_not(mask)):  # if all false, returns true
    #     inertial_range *= 2
    #     mask = (inertial_range[0] <= kgrid) & (kgrid <= inertial_range[1])
    kgrid, spectrum = kgrid[mask], spectrum[mask]
    
    if len(kgrid.shape) == 1:
        kgrid = kgrid.reshape((-1,1))  # have to make kgrid 2D
    
    log_k, log_spec = np.log(kgrid), np.log(spectrum)
    model = LinearRegression().fit(log_k, log_spec)
    slope = model.coef_
    return slope[0]

# --- PLOTTING --- #

def plot_MHD(l, t, titles, vels, Bs, fname, inertial_range=(2*10**-2, 4*10**-2)):
    filename = diag.PATH + fname
    inertial_range = np.array(inertial_range)

    inertial_range[0] = l[0]
    inertial_range[1] = inertial_range[0]*2

    l_mask = (inertial_range[0] <= l) & (l < inertial_range[1])
    l_inertial = l[l_mask]

    B_sf = []

    # for i in range(len(titles)):
    # gets parallel and perp components
    for i in [0, len(titles)-1]:

        slope = get_spectral_slope(l, Bs[i], inertial_range)
        slope_label = "{:+.2f}".format(slope)

        fit_start = Bs[i][l_mask][0]
        l_1 = fit_start * (l_inertial/inertial_range[0])
        l_23 = fit_start * (l_inertial/inertial_range[0])**(2/3)
        l_fit = fit_start * (l_inertial/inertial_range[0])**(slope)

        B_sf.append(Bs[i])  # for comparison of parallel and perp SF
        plt.loglog(l, vels[i], l, Bs[i])
        if i == 0:
            plt.loglog(l_inertial, l_1, ':', l_inertial, l_fit, ':')
            plt.title(r'$S_2(l_\|)$ with ' + titles[i])
            plt.xlabel(r'$l_\|$')
            plt.ylabel(r'$S_2(l_\|)$)')
            plt.legend(['Velocity', 'Magnetic',
                        r'$l_\|$', r'$l_{\|}^{' + slope_label + '}$'])
        else:
            plt.loglog(l_inertial, l_23, ':', l_inertial, l_fit, ':')
            plt.title(r'$S_2(l_\perp)$ with ' + titles[i])
            plt.xlabel(r'$l_\perp$')
            plt.ylabel(r'$S_2(l_\perp)$)')
            plt.legend(['Velocity', 'Magnetic',
                        r'$l_{\perp}^{2/3}$', r'$l_{\perp}^{' + slope_label + '}$'])
        plt.savefig(filename + '/t' + t + '_' + str(i) + '.png')
        plt.clf()

    B_sf_prl, B_sf_prp = B_sf[0], B_sf[1]
    plt.loglog(l, B_sf_prl, l, B_sf_prp)
    plt.title('Magnetic Structure Functions Comparisons')
    plt.xlabel(r'$l_{\|, \perp}$')
    plt.ylabel(r'$S_2$')
    plt.legend([r'$S_2(l_\|)$', r'$S_2(l_\perp)$'])
    plt.savefig(filename + '/magnetic_comparison' + '.png')
    plt.clf()


def plot_struct(l_grid, v_avg, t, fname, inertial_range=(3*10**-2, 6*10**-2)):
    filename = diag.PATH + fname

    l_mask = (inertial_range[0] <= l_grid) & (l_grid < inertial_range[1])
    l_inertial = l_grid[l_mask]
    fit_start = v_avg[l_mask][0]
    l_23 = fit_start * (l_inertial/inertial_range[0])**(2/3)

    plt.loglog(l_grid, v_avg, l_inertial, l_23, ':')
    plt.title(r'$S_2(l)$ at $t=$ ' + t)
    plt.xlabel(r'$l$')
    plt.ylabel(r'$S_2(l)$)')
    plt.legend(['Velocity Structure Function', r'$l^{2/3}$'])
    plt.savefig(filename + '/struct_t' + t + '.png')
    plt.clf()

# --- CALL FUNCTION --- #

def structure_function(fname, n, do_mhd=1, N=1e7, do_ldist=0, prob=default_prob):
    '''Calculates and plots structure function.

    Parameters
    ----------
    fname : string
        name of folder containing data of interest
    n : int
        number corresponding to a given time snapshot within the data
    do_mhd : bool, optional
        runs computation of anisotropic structure function in MHD turbulence, by default 1
    N : float, optional
        the number of points to calculate the structure function over, by default 1e6
    do_ldist : bool, optional
        boolean for running a test on the distribution of distances between points, by default 0
    prob : string, optional
        name of Athena++ problem, by default default_prob
    '''

    def get_length(do_diff=0):
        names = ['RootGridX3', 'RootGridX2', 'RootGridX1']
        lengths = np.array([data[name][:2] for name in names])
        if do_diff:
            return lengths[:, 1] - lengths[:, 0]
        return lengths

    def get_points(grid_points):
        Ngrid = data['RootGridSize'][::-1]
        Lgrid = get_length()
        return grid_points*(Lgrid[:, 1] - Lgrid[:, 0]) / Ngrid + Lgrid[:, 0]

    # Read in data and set up grid
    data = diag.load_data(fname, n, prob)
    # Following (z, y, x) convention from athena_read
    grid = data['RootGridSize'][::-1]
    t = '{:.1f}'.format(data['Time']) + ' s'

    vel_data = np.array((data['vel1'], data['vel2'], data['vel3']))
    if do_mhd:
        B_data = np.array((data['Bcc1'], data['Bcc2'], data['Bcc3']))
        # B_mean = np.mean(B_data, axis=(1,2,3)).reshape((3, 1, 1, 1))
        # B_data -= B_mean
    L1, L2 = generate_points(grid, N)  # L2 = L1 + l
    print('Generated points')

    # Get actual position vector components for each pair of grid points
    # and difference vector between them
    x1_vec = get_points(L1)
    x2_vec = get_points(L2)
    l_vec = x2_vec - x1_vec
    # Find distance between each pair of points
    l_mag = diag.get_mag(l_vec)
    # Minimum box side length for making l_grid in calc_struct()
    L_min = np.min(get_length(do_diff=1))
    print('Lengths calculated')

    # Output distribution of l vector lengths
    if do_ldist:
        n, bins, patches = plt.hist(l_mag, 100)
        plt.title(r'Distribution of $l$ at $t =$ ' + t)
        plt.xlabel('Distance between points')
        plt.ylabel('Counts')
        plt.show()

    if do_mhd:
        θ, l_mask = get_l_perp(L1, L2, l_vec, B_data)
        titles, vels, Bs = [], [], []
        l_grid = calc_struct(L1, L2, vel_data, l_mag, L_min)[0]

        for i, l_m in enumerate(l_mask):
            titles.append(str(θ[i]) + r'$^\circ$ $\leq \theta <$ '
                          + str(θ[i+1]) + r'$^\circ$' + ' at t = ' + t)
            vels.append(calc_struct(L1, L2, vel_data, l_mag, L_min, l_m, 1)[1])
            Bs.append(calc_struct(L1, L2, B_data, l_mag, L_min, l_m, 1)[1])

        plot_MHD(l_grid, t, titles, vels, Bs, fname)
    else:
        l_grid, Δv_avg = calc_struct(L1, L2, vel_data, l_mag, L_min)
        plot_struct(l_grid, Δv_avg, t, fname)
