'''Code to calculate the energy spectrum of turbulence
   in a fluid simulation run in Athena++.
'''
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import diagnostics as diag
from matplotlib import rc
rc('text', usetex=True)  # LaTeX labels
default_prob = diag.DEFAULT_PROB


def calc_spectrum(output_dir, save_dir, fname, prob=default_prob,
                  plot_title='test', do_mhd=1, do_title=1):

    # Getting turnover time and converting to file number
    max_n = diag.get_maxn(output_dir)
    tau_file = int(max_n/2)
    nums = range(0, tau_file)  # average over first 2 alfven periods
    # nums = range(tau_file, max_n) # average over last 2 alfven periods

    do_full_calc = not diag.check_dict(save_dir)
    if do_full_calc:
        # create grid of K from first time step
        data = diag.load_data(output_dir, 0, prob=prob)
        (KX, KY, KZ), kgrid = diag.ft_grid(data, 1)
        Kprl = np.abs(KX)
        Kperp = np.sqrt(np.abs(KY)**2 + np.abs(KZ)**2)
        Kmag = np.sqrt(Kprl**2+Kperp**2)
        Kspec = Kmag

        # Dictionary to hold spectrum information
        S = {}
        S['Nk'] = len(kgrid) - 1  # number of elements in kgrid
        S['kgrid'] = 0.5*(kgrid[:-1] + kgrid[1:])  # average of neighbours

        # Count the number of modes in each bin to normalize later -- this
        # gives a smoother result, as we want the average energy in each bin.
        # This isn't usually used, but will keep it in case it is needed.
        oneGrid = np.ones(KX.shape)
        S['nbin'] = spect1D(oneGrid, oneGrid, Kspec, kgrid)*np.size(oneGrid)**2
        S['nnorm'] = S['nbin']/S['kgrid']**2
        S['nnorm'] /= np.mean(S['nnorm'])

        # average over snapshots in nums
        def m3(a):
            return np.mean(np.mean(np.mean(a)))

        ns = 0  # counter
        fields = ['vel1', 'vel2', 'vel3', 'Bcc1', 'Bcc2', 'Bcc3',
                  'EK', 'EM', 'B', 'rho']

        # Initializing variable fields in spectrum dictionary
        for var in fields:
            S[var] = 0

        for n in nums:
            try:
                data = diag.load_data(output_dir, n, prob=prob)
            except IOError:
                print('Could not load file', n)
                break

            if ns % 10 == 0:
                print('Doing n =', n)

            # Take the Fourier transform of the individual components
            # Find their energy (ia Parseal's theorem)
            # Add to total energy spectrum
            for vel in fields[:3]:
                ft = fft.fftn(data[vel])
                S[vel] += spect1D(ft, ft, Kspec, kgrid)
                S['EK'] += S[vel]  # Total spectrum is sum of each component

            if do_mhd:
                Bmag = 0
                for Bcc in fields[3:6]:
                    ft = fft.fftn(data[Bcc])
                    S[Bcc] += spect1D(ft, ft, Kspec, kgrid)
                    S['EM'] += S[Bcc]
                    Bmag += data[Bcc]**2

                Bmag = np.sqrt(Bmag)
                ft_Bmag = fft.fftn(Bmag)
                S['B'] += spect1D(ft_Bmag, ft_Bmag, Kspec, kgrid)

            ft_rho = fft.fftn(data['rho'] - m3(data['rho']))
            S['rho'] += spect1D(ft_rho, ft_rho, Kspec, kgrid)

            ns += 1

        # Average over number of times done
        for var in fields:
            S[var] /= ns
        S['nums'] = nums

        diag.save_dict(S, save_dir)
    else:
        S = diag.load_dict(save_dir, 'mhd')

    plot_spectrum(S, save_dir, fname, plot_title, do_mhd, do_title=do_title)


def plot_spectrum(S, save_dir, fname, plot_title, do_mhd=1, do_title=1,
                  plot_show=0, k_line_mod=0, k_params=None):
    # plot spectrum
    if do_mhd:
        x = S['kgrid'][1:]
        if k_line_mod:
            x_mask = np.logical_and(k_params[0] < x, x < k_params[1])
            x_mod = x[x_mask]
            x53 = k_params[2] * (x_mod/k_params[0])**(-5/3)
            x3 = k_params[3] * (x_mod/k_params[0])**(-3)
            plt.loglog(x, S['EK'][1:], x, S['EM'][1:],
                       x_mod, x53, ':', x_mod, x3, ':')
        else:
            x53 = x**(-5/3) * 10**(2)
            x3 = x**(-3) * 10**(4.7)
            plt.loglog(x, S['EK'][1:], x, S['EM'][1:],
                       x, x53, ':',
                       x, x3, ':')
        plt.legend([r'$E_K$', r'$E_B$', r'$k^{-5/3}$', r'$k^{-3}$'])
    else:
        plt.loglog(S['kgrid'], S['EK'], S['kgrid'], S['kgrid']**(-5/3), ':')
        plt.legend([r'$E_K$', r'$k^{-5/3}$'])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$E(k)$')
    if do_title:
        plt.title('Energy  Spectrum: ' + plot_title)

    if diag.PATH not in save_dir:
        save_dir = diag.PATH + save_dir

    if plot_show:
        plt.show()
    else:
        plt.savefig(save_dir + fname + '_spec.pdf')
        plt.savefig(save_dir + fname + '_spec.png')
        plt.close()


def spect1D(v1, v2, K, kgrid):
    '''Function to find the spectrum < v1 v2 >,
    K is the kgrid associated with v1 and v2
    kgrid is the grid for spectral shell binning
    '''
    nk = len(kgrid) - 1
    out = np.zeros((nk, 1))
    NT2 = np.size(K)**2
    for k in range(nk):
        mask = np.logical_and(K < kgrid[k+1], K > kgrid[k])
        sum = np.sum(np.real(v1[mask])*np.conj(v2[mask]))
        out[k] = np.real(sum) / NT2
    return out
