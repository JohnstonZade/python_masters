'''Code to calculate the energy spectrum of turbulence
   in a fluid simulation run in Athena++.
'''
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import diagnostics as diag
from sklearn.linear_model import LinearRegression
from matplotlib import rc
rc('text', usetex=True)  # LaTeX labels
default_prob = diag.DEFAULT_PROB


def calc_timeind_spectrum(output_dir, save_dir, fname, prob=default_prob,
                          plot_title='test', inertial_range=(10**1.5, 10**2),
                          do_mhd=1, do_title=1):
    # create grid of K from first time step
        data = diag.load_data(output_dir, 0, prob=prob)
        (KZ, KY, KX), kgrid = diag.ft_grid('data', data=data, prob=prob, k_grid=1)
        Kprl = np.abs(KX)
        Kperp = np.sqrt(np.abs(KY)**2 + np.abs(KZ)**2)
        Kmag = np.sqrt(Kprl**2+Kperp**2)
        Kspec = Kmag

        # Dictionary to hold spectrum information
        S = {}
        S['Nk'] = len(kgrid) - 1  # number of elements in kgrid
        S['kgrid'] = 0.5*(kgrid[:-1] + kgrid[1:])  # average of neighbours

        fields = ['vel1', 'vel2', 'vel3', 'Bcc1', 'Bcc2', 'Bcc3',
                  'EK', 'EK_prl', 'EK_prp', 'EM', 'EM_prl', 'EM_prp', 'B', 'rho']

        # Initializing variable fields in spectrum dictionary
        for var in fields:
            S[var] = 0

        # Take the Fourier transform of the individual components
        # Find their energy (ie Parseval's theorem)
        # Add to total energy spectrum
        for vel in fields[:3]:
            ft = fft.fftn(data[vel])
            S[vel] += spect1D(ft, ft, Kspec, kgrid)
            S['EK'] += S[vel]  # Total spectrum is sum of each component
            S['EK_prl'] = spect1D(ft, ft, Kprl, kgrid)
            S['EK_prp'] += spect1D(ft, ft, Kperp, kgrid)

        if do_mhd:
            Bmag = 0
            for Bcc in fields[3:6]:
                ft = fft.fftn(data[Bcc])
                S[Bcc] += spect1D(ft, ft, Kspec, kgrid)
                S['EM'] += S[Bcc]
                S['EM_prl'] = spect1D(ft, ft, Kprl, kgrid)
                S['EM_prp'] += spect1D(ft, ft, Kperp, kgrid)
                Bmag += data[Bcc]**2

            Bmag = np.sqrt(Bmag)
            ft_Bmag = fft.fftn(Bmag)
            S['B'] += spect1D(ft_Bmag, ft_Bmag, Kspec, kgrid)

        ft_rho = fft.fftn(data['rho'] - np.mean(data['rho']))
        S['rho'] += spect1D(ft_rho, ft_rho, Kspec, kgrid)

        plot_spectrum(S, save_dir, fname, plot_title, inertial_range, do_mhd, do_title=do_title)


def calc_spectrum(output_dir, save_dir, fname, inertial_range=(10**1.5, 10**2), prob=default_prob,
                  plot_title='test', dict_name='mhd_spec', do_mhd=1, do_title=1):

    # Getting turnover time and converting to file number
    max_n = diag.get_maxn(output_dir)
    tau_file = int(max_n/2)
    nums = range(0, tau_file)  # average over first 2 alfven periods
    # nums = range(tau_file, max_n) # average over last 2 alfven periods

    do_full_calc = not diag.check_dict(save_dir, dict_name)
    if do_full_calc:
        # create grid of K from first time step
        data = diag.load_data(output_dir, 0, prob=prob)
        (KZ, KY, KX), kgrid = diag.ft_grid('data', data=data, prob=prob, k_grid=1)
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
                  'EK', 'EK_prl', 'EK_prp', 'EM', 'EM_prl', 'EM_prp', 'B', 'rho']

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
            # Find their energy (ie Parseval's theorem)
            # Add to total energy spectrum
            for vel in fields[:3]:
                ft = fft.fftn(data[vel])
                S[vel] += spect1D(ft, ft, Kspec, kgrid)
                S['EK'] += S[vel]  # Total spectrum is sum of each component
                S['EK_prl'] = spect1D(ft, ft, Kprl, kgrid)
                S['EK_prp'] += spect1D(ft, ft, Kperp, kgrid)

            if do_mhd:
                Bmag = 0
                for Bcc in fields[3:6]:
                    ft = fft.fftn(data[Bcc])
                    S[Bcc] += spect1D(ft, ft, Kspec, kgrid)
                    S['EM'] += S[Bcc]
                    S['EM_prl'] = spect1D(ft, ft, Kprl, kgrid)
                    S['EM_prp'] += spect1D(ft, ft, Kperp, kgrid)
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

        diag.save_dict(S, save_dir, dict_name)
    else:
        S = diag.load_dict(save_dir, dict_name)

    plot_spectrum(S, save_dir, fname, plot_title, inertial_range, do_mhd, do_title=do_title)


def plot_spectrum(S, save_dir, fname, plot_title, inertial_range, do_mhd=1, do_title=1,
                  do_pdf=0):
    # plot spectrum
    if do_mhd:
        k = S['kgrid'][1:]
        EK = S['EK_prp'][1:]
        EM = S['EM_prp'][1:]
        slope = get_spectral_slope(k, EM, inertial_range)
        slope_label = "{:+.2f}".format(slope)

        plt.loglog(k, EK, k, EM)
        plt.xlabel(r'$k_\perp$')
        plt.ylabel(r'$E(k_\perp)$')
        legend = [r'$E_{K,\perp}$', r'$E_{B,\perp}$', r'$k_{\perp}^{-5/3}$', r'$k_{\perp}^{' + slope_label + '}$']
        
        # generating fitting line
        k_mask = np.logical_and(inertial_range[0] <= k, k < inertial_range[1])
        k_inertial = k[k_mask]
        fit_start = EM[k_mask][0] * 10**0.3
        x_53 = fit_start * (k_inertial/inertial_range[0])**(-5/3)
        x_slope = fit_start * (k_inertial/inertial_range[0])**(slope)
        plt.loglog(k_inertial, x_53, ':', k_inertial, x_slope, ':')
        
        plt.legend(legend)
    else:
        plt.loglog(S['kgrid'], S['EK'], S['kgrid'], S['kgrid']**(-5/3), ':')
        plt.legend([r'$E_K$', r'$k^{-5/3}$'])
        plt.xlabel(r'$k$')
        plt.ylabel(r'$E(k)$')
    
    if do_title:
        plt.title('Energy Spectrum: ' + plot_title)

    if diag.PATH not in save_dir:
        save_dir = diag.PATH + save_dir

    if do_pdf:
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
        # For k between kgrid[k] and kgrid[k+1]
        mask = np.logical_and(K < kgrid[k+1], K > kgrid[k])
        # Find the total energy within that k range
        # This is the specturm <v1 v2>(k) ~ integral(v1 v2* dk) with kgrid[k] < k < kgrid[k+1]
        # which is equivalent to the total energy in that range via Parseval's theorem
        # essentially the mean of v1 v2* within this frequency range and thus the mean energy
        spec_sum = np.sum(np.real(v1[mask])*np.conj(v2[mask]))
        out[k] = np.real(spec_sum) / NT2
    return out


def get_spectral_slope(kgrid, spectrum, inertial_range):
    mask = (inertial_range[0] <= kgrid) & (kgrid <= inertial_range[1])
    kgrid, spectrum = kgrid[mask], spectrum[mask]
    
    if len(kgrid.shape) == 1:
        kgrid = kgrid.reshape((-1,1))  # have to make kgrid 2D
    
    log_k, log_spec = np.log(kgrid), np.log(spectrum)
    model = LinearRegression().fit(log_k, log_spec)
    slope = model.coef_
    return slope[0, 0]