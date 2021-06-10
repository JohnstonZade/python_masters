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


def calc_spectrum(output_dir, save_dir, fname='', return_dict=0, inertial_range=(10**1.5, 10**2), prob=default_prob,
                  plot_title='test', dict_name='mhd_spec', do_single_file=0, n=0, a=1, do_isotropic=1,
                  normalize_energy=1, do_mhd=1, gaussian=0, do_prp_spec=1, do_prl_spec=0, do_title=1):

    # Getting turnover time and converting to file number
    if do_single_file:
        nums = range(n, n+1)
    else:
        max_n = diag.get_maxn(output_dir) // 2
        nums = range(0, max_n)  # average over first 2 alfven periods
    # nums = range(tau_file, max_n) # average over last 2 alfven periods

    do_full_calc = not diag.check_dict(save_dir, dict_name)
    if do_full_calc:
        # create grid of K from first time step
        data = diag.load_data(output_dir, 0, prob=prob)
        (KZ, KY, KX), kgrid = diag.ft_grid('data', data=data, prob=prob, k_grid=1, make_iso_box=do_isotropic)
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

        ns = 0  # counter
        fields = ['vel1', 'vel2', 'vel3', 'Bcc1', 'Bcc2', 'Bcc3',
                  'EK', 'EK_prp', 'EK_prl', 'EM', 'EM_prp', 'EM_prl',
                  'EK_2D', 'EM_2D', 'B', 'rho']

        # Initializing variable fields in spectrum dictionary
        for var in fields:
            S[var] = 0

        for n in nums:
            try:
                data = diag.load_data(output_dir, n, prob=prob)
            except IOError:
                print('Could not load file', n)
                break

            # Take the Fourier transform of the individual components
            # Find their energy (i.e. Parseval's theorem)
            # Add to total energy spectrum
            for vel in fields[:3]:
                v = data[vel]
                ft = fft.fftn(v)
                S[vel] += spect1D(ft, ft, Kspec, kgrid)
                S['EK'] += S[vel]  # Total spectrum is sum of each component
                S['EK_prl'] += spect1D(ft, ft, Kprl, kgrid)
                S['EK_prp'] += spect1D(ft, ft, Kperp, kgrid)
                S['EK_2D']  += spect2D(ft, ft, Kprl, Kperp, kgrid)
            if normalize_energy:
                # v_A ~ a^(-1) ⟹ (v_A)^2 ∼ a^(-2), assuming v_A0 = 1
                S['EK'] /= a**(-2)
                S['EK_prp'] /= a**(-2)
                S['EK_2D'] /= a**(-2)

            if do_mhd:
                Bmag = 0
                for Bcc in fields[3:6]:
                    B = data[Bcc]
                    ft = fft.fftn(B)
                    S[Bcc] += spect1D(ft, ft, Kspec, kgrid)
                    S['EM'] += S[Bcc]
                    S['EM_prl'] += spect1D(ft, ft, Kprl, kgrid)
                    S['EM_prp'] += spect1D(ft, ft, Kperp, kgrid)
                    S['EM_2D']  += spect2D(ft, ft, Kprl, Kperp, kgrid)
                    Bmag += B**2
                if normalize_energy:
                    # B_x ∼ a^(-2) ⟹ (B_x)^2 ∼ a^(-4), assuming ⟨B_x0⟩=1
                    S['EM'] /= a**(-4)
                    S['EM_prp'] /= a**(-4)
                    S['EM_2D'] /= a**(-4)
            
                Bmag = np.sqrt(Bmag)
                ft_Bmag = fft.fftn(Bmag)
                S['B'] += spect1D(ft_Bmag, ft_Bmag, Kspec, kgrid)

            ft_rho = fft.fftn(data['rho'] - np.mean(data['rho']))
            S['rho'] += spect1D(ft_rho, ft_rho, Kspec, kgrid)

            ns += 1

        if not do_single_file:
            # Average over number of times done
            for var in fields:
                S[var] /= ns
            S['nums'] = nums

        diag.save_dict(S, save_dir, dict_name)
    else:
        S = diag.load_dict(save_dir, dict_name)

    if return_dict:
        return S
    else:
        plot_spectrum(S, save_dir, fname, plot_title, inertial_range, do_isotropic=do_isotropic, gaussian=gaussian, do_mhd=do_mhd, do_prp_spec=do_prp_spec,
                      do_prl_spec=do_prl_spec, do_title=do_title)


def plot_spectrum(S, save_dir, fname, plot_title, inertial_range, do_mhd=1, do_isotropic=1, gaussian=0, do_prp_spec=1, do_prl_spec=0, do_title=1, normalized=1,
                  do_pdf=0):
    # plot spectrum
    if do_mhd:
        inertial_range = np.array(inertial_range)

        k = S['kgrid'][1:]
        if do_prp_spec:
            EK = S['EK_prp'][1:]
            EM = S['EM_prp'][1:]
        elif do_prl_spec:
            EK = S['EK_prl'][1:]
            EM = S['EM_prl'][1:]
        else:
            EK = S['EK'][1:]
            EM = S['EM'][1:]
        plt.loglog(k, EK, k, EM)
        
        
        # generating fitting line
        # get closest k values to desired inertial range
        inertial_range[0] = k[np.abs(k - inertial_range[0]).argmin()]
        inertial_range[1] = k[np.abs(k - inertial_range[1]).argmin()]
        slope = get_spectral_slope(k, EK, inertial_range)
        slope_label = "{:+.2f}".format(slope)

        if do_prp_spec:
            plt.xlabel(r'$k_\perp L_\perp$')
            if normalized:
                plt.ylabel(r'$E_{K}(k_\perp L_\perp) / v^2_A, \ E_{B}(k_\perp L_\perp) / B^2_x$')
            else:
                plt.ylabel(r'$E(k_\perp L\perp)$')
            if gaussian:
                legend = [r'$E_{K,\perp}$', r'$E_{B,\perp}$']
            else:
                legend = [r'$E_{K,\perp}$', r'$E_{B,\perp}$', r'$(k_{\perp}L_{\perp})^{-5/3}$',
                          r'$(k_{\perp}L_{\perp})^{' + slope_label + '}$']
        elif do_prl_spec:
            plt.xlabel(r'$k_\| L_\|$')
            if normalized:
                plt.ylabel(r'$E_{K}(k_\| L\|) / v^2_A, \ E_{B}(k_\| L\|) / B^2_x$')
            else:
                plt.ylabel(r'$E(k_\| L\|)$')
            if gaussian:
                legend = [r'$E_{K,\|}$', r'$E_{B,\|}$']
            else:
                legend = [r'$E_{K,\|}$', r'$E_{B,\|}$', r'$(k_{\|}L_{\|})^{-5/3}$', r'$(k_{\|}L_{\|})^{' + slope_label + '}$']
        else:
            plt.xlabel(r'$k$')
            plt.ylabel(r'$E(k)$')
            legend = [r'$E_{K}$', r'$E_{B}$', r'$k^{-5/3}$', r'$k^{' + slope_label + '}$']

        k_mask = np.logical_and(inertial_range[0] <= k, k <= inertial_range[1])
        # while np.all(np.logical_not(k_mask)):  # if all false, returns true
        #     inertial_range *= 2
        #     k_mask = (inertial_range[0] <= k) & (k <= inertial_range[1])
        k_inertial = k[k_mask]
        fit_start = EK[k_mask][0]

        x_theory_slope = -5/3
        x_theory = fit_start * (k_inertial/inertial_range[0])**(x_theory_slope)
        x_slope = fit_start * (k_inertial/inertial_range[0])**(slope)
        if not gaussian:
            plt.loglog(k_inertial, x_theory, ':', k_inertial, x_slope, ':')
        
        plt.legend(legend)
    else:
        plt.loglog(S['kgrid'], S['EK'], S['kgrid'], S['kgrid']**(-5/3), ':')
        plt.legend([r'$E_K$', r'$k^{-5/3}$'])
        plt.xlabel(r'$k$')
        plt.ylabel(r'$E(k)$')
    
    if do_title:
        plt.title('Energy Spectrum: ' + plot_title)

    save_dir = diag.format_path(save_dir)

    if do_prp_spec:
        fig_suffix = '_prpspec'
    elif do_prl_spec:
        fig_suffix = '_prlspec'
    else:
        fig_suffix = '_spec'
    
    if do_pdf:
        plt.savefig(save_dir + fname + fig_suffix + '.pdf')
    plt.savefig(save_dir + fname + fig_suffix + '.png')
    plt.close()


def spect1D(v1, v2, K, kgrid):
    '''Function to find the spectrum < v1 v2 >,
    K is the kgrid associated with v1 and v2
    kgrid is the grid for spectral shell binning
    '''
    nk = len(kgrid) - 1  # number of bins
    out = np.zeros((nk, 1))
    NT2 = np.size(K)**2  # total number of elements summed over, used as normalization
    for k in range(nk):
        # For k between kgrid[k] and kgrid[k+1]
        mask = (kgrid[k] < K) & (K <= kgrid[k+1])
        # Find the total energy within that k range
        # This is the specturm <v1 v2>(k) ~ integral(v1 v2* dk) with kgrid[k] < k < kgrid[k+1]
        # which is equivalent to the total energy in that range via Parseval's theorem
        # essentially the mean of v1 v2* within this frequency range and thus the mean energy
        spec_sum = np.sum(np.real(v1[mask])*np.conj(v2[mask]))
        out[k] = np.real(spec_sum) / NT2
    return out


def spect1D_test(v1, v2, K, kgrid):
    nk = len(kgrid) - 1
    NT2 = (K.size)**2
    k = np.copy(kgrid)
    k.reshape(kgrid.shape, 1, 1, 1)
    Kb = np.broadcast_to(K, (nk, *K.shape))
    v1b = np.broadcast_to(v1, (nk, *v1.shape))
    v2b = np.broadcast_to(v2, (nk, *v2.shape))
    mask = np.logical_not( (k[:-1] < Kb) & (Kb <= k[1:]) )
    v1m = np.ma.array(v1b, mask=mask)
    v2m = np.ma.array(v2b, mask=mask)
    prod = np.real(v1m)*np.conj(v2m)
    out = np.real(prod.sum(axis=(1,2,3))) / NT2
    return out


def spect2D(v1, v2, Kprl, Kprp, kgrid):
    '''Function to find the spectrum < v1 v2 >,
    K is the kgrid associated with v1 and v2
    kgrid is the grid for spectral shell binning
    '''
    nk = len(kgrid) - 1
    out = np.zeros((nk, nk))
    NT2 = np.size(Kprp)**2
    for kprl in range(nk):
        mask_prl = (kgrid[kprl] < Kprl) & (Kprl <= kgrid[kprl+1])
        for kprp in range(nk):
            # For k between kgrid[k] and kgrid[k+1]
            mask_prp = (kgrid[kprp] < Kprp) & (Kprp <= kgrid[kprp+1])
            mask = mask_prl & mask_prp
            # Find the total energy within that k range
            # This is the specturm <v1 v2>(k) ~ integral(v1 v2* dk) with kgrid[k] < k < kgrid[k+1]
            # which is equivalent to the total energy in that range via Parseval's theorem
            # essentially the mean of v1 v2* within this frequency range and thus the mean energy
            spec_sum = np.sum(np.real(v1[mask])*np.conj(v2[mask]))
            out[kprp, kprl] = np.real(spec_sum) / NT2
    return out


def get_spectral_slope(kgrid, spectrum, inertial_range):
    mask = (inertial_range[0] <= kgrid) & (kgrid <= inertial_range[1])
    while np.all(np.logical_not(mask)):  # if all false, returns true
        inertial_range *= 2
        mask = (inertial_range[0] <= kgrid) & (kgrid <= inertial_range[1])
    kgrid, spectrum = kgrid[mask], spectrum[mask]
    
    if len(kgrid.shape) == 1:
        kgrid = kgrid.reshape((-1,1))  # have to make kgrid 2D
    
    log_k, log_spec = np.log(kgrid), np.log(spectrum)
    model = LinearRegression().fit(log_k, log_spec)
    slope = model.coef_
    return slope[0, 0]