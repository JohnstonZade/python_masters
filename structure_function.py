'''Code to calculate structure function of a fluid simulation.'''
import numpy as np
from numpy.random import randint, random
import matplotlib.pyplot as plt
from matplotlib import rc
import diagnostics as diag
rc('text', usetex=True)  # LaTeX labels
default_prob = diag.DEFAULT_PROB


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
    the corresponding x values satisfy x_bin[i] <= x <= x_bin[i+1].

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
        Takes the average of the two to get the mean field.
    Gets l and B unit vectors and then calculates cos(theta) = l dot b
    Bins l vectors with theta = [0, 15, 30, 45, 60, 75, 90]
        0-15 is l_parallel, 45-90 is l_perp
    Outputs angle list and mask for the l vector in order for the
    structure function to be calculated.
    '''
    # Calculate average B field between point pairs
    B1_vec = np.array([diag.get_vec(B, p) for p in L1])
    B2_vec = np.array([diag.get_vec(B, p) for p in L2])
    B_mean = 0.5*(B1_vec + B2_vec)

    # Dot product of unit vectors to get cos(θ)
    cθ = abs(np.sum(diag.get_unit(B_mean)*diag.get_unit(l), axis=1))
    θ_data = np.arccos(cθ)
    θ = np.linspace(0, 90, 10, endpoint=True)  # 7 default
    θlen = len(θ) - 1
    θ_rad = (np.pi/180)*θ

    # Create l_mask depending on θ
    l_mask = [select_y(θ_data, θ_rad, diag.get_mag(l), i, return_mask=1)
              for i in range(θlen)]
    return θ, l_mask


def calc_struct(L1, L2, v, l_mag, L_max, mask=[], use_mask=0):
    # For each pair of position vectors x1 ∈ L1, x2 ∈ L2
    # Get vectors v1, v2 at each point
    # Calculate Δv2 = abs(v1 - v2)**2
    # We now have a mapping of l to Δv2 <- structure function
    v1_vec = np.array([diag.get_vec(v, p) for p in L1])
    v2_vec = np.array([diag.get_vec(v, p) for p in L2])
    Δv_vec = v1_vec - v2_vec
    Δv_mag2 = diag.get_mag(Δv_vec)**2

    # Bin and plot structure function
    # Plot in the middle of the bin points otherwise the size of arrays
    # won't match up.
    N_l = 30
    l_bin = np.logspace(np.log10(2*L_max/N_l), np.log10(L_max/2), N_l+1)
    l_grid = 0.5*(l_bin[:-1] + l_bin[1:])
    Δv_avg = np.array([get_mean(l_mag, l_bin, Δv_mag2, i, mask, use_mask)
                       for i in range(N_l)])

    return l_grid, Δv_avg


def plot_MHD(l, t, titles, vels, Bs, fname):
    filename = diag.PATH + fname

    # for i in range(len(titles)):
    # gets parallel and perp components
    for i in [0, len(titles)-1]:
        plt.loglog(l, vels[i], l, Bs[i])
        plt.loglog(l, l**(2/3), ':', l, l, ':')
        plt.title(r'$S_2(l)$ with ' + titles[i])
        plt.xlabel(r'log($l$)')
        plt.ylabel(r'log($S_2(l)$))')
        plt.legend(['Vel Structure Function', 'B-field Structure Function',
                    r'$l^{2/3}$', r'$l$'])
        plt.savefig(filename + '/t' + t + '_' + str(i) + '.png')
        plt.clf()


def plot_struct(l_grid, v_avg, t, fname):
    filename = diag.PATH + fname

    plt.loglog(l_grid, v_avg, l_grid, l_grid**(2/3), ':')
    plt.title(r'$S_2(l)$ at $t=$ ' + t)
    plt.xlabel(r'log($l$)')
    plt.ylabel(r'log($S_2(l)$))')
    plt.legend(['Structure Function', r'$l^{2/3}$'])
    plt.savefig(filename + '/struct_t' + t + '.png')
    plt.clf()


def structure_function(fname, n, do_mhd=1, N=1e6, do_ldist=0, prob=default_prob):
    '''Calculates and plots structure function.'''

    def get_length():
        '''Returns the dimensions of the simulation.'''
        names = ['RootGridX3', 'RootGridX2', 'RootGridX1']
        return np.array([data[name][1]-data[name][0] for name in names])

    def get_point(p):
        '''Returns coordinate of given grid point.'''
        tp = tuple(p)
        return np.array([zz[tp], yy[tp], xx[tp]])

    # Read in data and set up grid
    data = diag.load_data(fname, n, prob)
    # Following (z, y, x) convention from athena_read
    grid = data['RootGridSize'][::-1]
    t = '{:.1f}'.format(data['Time']) + ' s'

    zz, yy, xx = np.meshgrid(data['x3f'], data['x2f'], data['x1f'],
                             indexing='ij')
    vel_data = (data['vel1'], data['vel2'], data['vel3'])
    if do_mhd:
        B_data = (data['Bcc1'], data['Bcc2'], data['Bcc3'])
    L1, L2 = generate_points(grid, N)
    print('Generated points')

    # Get actual position vector components for each pair of grid points
    # and difference vector between them
    x1_vec = np.array([get_point(p) for p in L1])
    x2_vec = np.array([get_point(p) for p in L2])
    l_vec = x1_vec - x2_vec
    # Find distance between each pair of points
    l_mag = diag.get_mag(l_vec)
    # Maximum box side length for making l_grid in calc_struct()
    L = np.max(get_length())
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
        l_grid = calc_struct(L1, L2, vel_data, l_mag, L)[0]

        for i, l_m in enumerate(l_mask):
            titles.append(str(θ[i]) + r'$^\circ$ $\leq \theta <$ '
                          + str(θ[i+1]) + r'$^\circ$' + ' at t = ' + t)
            vels.append(calc_struct(L1, L2, vel_data, l_mag, L, l_m, 1)[1])
            Bs.append(calc_struct(L1, L2, B_data, l_mag, L, l_m, 1)[1])

        plot_MHD(l_grid, t, titles, vels, Bs, fname)

    l_grid, Δv_avg = calc_struct(L1, L2, vel_data, l_mag, L)
    plot_struct(l_grid, Δv_avg, t, fname)
