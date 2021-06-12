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
        data = diag.load_data(output_dir, n, prob=prob)
        KZ, KY, KX = diag.ft_grid('data', data=data, prob=prob, make_iso_box=do_isotropic)
        Kprl = np.maximum(np.abs(KX), 1e-4)
        Kprp = np.maximum(np.sqrt(abs(KY)**2 + abs(KZ)**2), 1e-4)
        Kmag = np.sqrt(Kprl**2+Kprp**2)
        
        Kmag_mult, Kmag_bins = get_k_bins(Kmag)
        Kprl_mult, Kprl_bins = get_k_bins(Kprl)
        Kprp_mult, Kprp_bins = get_k_bins(Kprp)

        def grid_from_bins(bins):
            return 0.5*(bins[1:] + bins[:-1])

        # Dictionary to hold spectrum information
        S = {}
        grids, bins = {}, {}
        grids['Kmag'], grids['Kprl'], grids['Kprp'] = grid_from_bins(Kmag_bins), grid_from_bins(Kprl_bins), grid_from_bins(Kprp_bins)
        bins['Kmag'], bins['Kprl'], bins['Kprp'] = Kmag_bins, Kprl_bins, Kprp_bins
        S['grids'], S['bins'] = grids, bins

        ns = 0  # counter
        fields = ['vel1', 'vel2', 'vel3', 'Bcc1', 'Bcc2', 'Bcc3',
                  'EK', 'EK_box_prl', 'EK_box_prp', 'EK_fluc_prp', 'EK_fluc_prp_box_prp', 'EK_2D',
                  'EM', 'EM_box_prl', 'EM_box_prp', 'EM_fluc_prp', 'EM_fluc_prp_box_prp', 'EM_2D',
                  'B', 'rho']

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
                # Isotropic spectra
                S[vel] += spec1D(ft, ft, Kmag, Kmag_bins, Kmag_mult, kmode_norm=1)
                S['EK'] += S[vel]  # Total spectrum is sum of each component
                # Box-parallel (along x-axis) spectrum
                S['EK_box_prl'] += spec1D(ft, ft, Kprl, Kprl_bins, Kprl_mult)
                # Box-perpendicular spectrum
                S['EK_box_prp'] += spec1D(ft, ft, Kprp, Kprp_bins, Kprp_mult, kmode_norm=1)
                if vel != 'vel1':
                    # Perpendicular fluctuation spectra
                    S['EK_fluc_prp'] += spec1D(ft, ft, Kmag, Kmag_bins, Kmag_mult, kmode_norm=1)
                    S['EK_fluc_prp_box_prp'] += spec1D(ft, ft, Kprp, Kprp_bins, Kprp_mult, kmode_norm=1)
                # 2D (k_prl and k_prl) spectrum
                S['EK_2D']  += spec2D(ft, ft, Kprp, Kprl, Kprp_bins, Kprl_bins, Kprp_mult)
            if normalize_energy:
                # v_A ~ a^(-1) ⟹ (v_A)^2 ∼ a^(-2), assuming v_A0 = 1
                for key in S.keys():
                    if 'EK' in key:
                        S[key] /= a**(-2)

            if do_mhd:
                Bmag = 0
                for Bcc in fields[3:6]:
                    B = data[Bcc]
                    ft = fft.fftn(B)
                    # Isotropic spectra
                    S[Bcc] += spec1D(ft, ft, Kmag, Kmag_bins, Kmag_mult, kmode_norm=1)
                    S['EM'] += S[Bcc]  # Total spectrum is sum of each component
                    # Box-parallel (along x-axis) spectrum
                    S['EM_box_prl'] += spec1D(ft, ft, Kprl, Kprl_bins, Kprl_mult)
                    # Box-perpendicular spectrum
                    S['EM_box_prp'] += spec1D(ft, ft, Kprp, Kprp_bins, Kprp_mult, kmode_norm=1)
                    if Bcc != 'Bcc1':
                        # Perpendicular fluctuation spectra
                        S['EM_fluc_prp'] += spec1D(ft, ft, Kmag, Kmag_bins, Kmag_mult, kmode_norm=1)
                        S['EM_fluc_prp_box_prp'] += spec1D(ft, ft, Kprp, Kprp_bins, Kprp_mult, kmode_norm=1)
                    # 2D (k_prl and k_prl) spectrum
                    S['EM_2D']  += spec2D(ft, ft, Kprp, Kprl, Kprp_bins, Kprl_bins, Kprp_mult)
                    Bmag += B**2
                if normalize_energy:
                    # B_x ∼ a^(-2) ⟹ (B_x)^2 ∼ a^(-4), assuming ⟨B_x0⟩=1
                    for key in S.keys():
                        if 'EM' in key:
                            S[key] /= a**(-4)
            
                Bmag = np.sqrt(Bmag)
                ft_Bmag = fft.fftn(Bmag)
                S['B'] += spec1D(ft_Bmag, ft_Bmag, Kmag, Kmag_bins, Kmag_mult, kmode_norm=1)

            ft_rho = fft.fftn(data['rho'] - np.mean(data['rho']))
            S['rho'] += spec1D(ft_rho, ft_rho, Kmag, Kmag_bins, Kmag_mult, kmode_norm=1)

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

def get_k_bins(k):
    k_flat = k.reshape(-1)
    k_min, k_max = 1e-5, k.max()
    # k_min, k_max = k[k > 0].min(), k.max()
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), 2000)
    # multiplicity of a mode (number of times we see that wavenumber)
    k_hist = np.histogram(k_flat, k_bins)[0]
    
    # Removing bins that have no modes in them (essentially widening the bins)
    mode_mult = np.delete(k_hist, k_hist == 0)
    k_bins = np.hstack((np.delete(k_bins[:-1], k_hist == 0), k_bins[-1]))
    return mode_mult, k_bins

def spec1D(v1, v2, k, k_bins, mode_mult, kmode_norm=0):
    # 1-to-1 correspondence in flattening grid
    # (i.e. FT gets mapped to the same index as its corresponding k-point)
    k_flat = k.reshape(-1)
    energy = (0.5*v1*np.conj(v2)).reshape(-1)  # Parseval's theorem
    # Bin energies in a given k_range
    e_hist = np.histogram(k_flat, k_bins, weights=np.real(energy))[0]
    
    if kmode_norm:
        # accounting for increase in modes in kspace
        # as we move further out
        n_modes = mode_mult.sum()
        e_hist /= n_modes**2
    return e_hist

def spec2D(v1, v2, kprp, kprl, kprp_bins, kprl_bins, mode_mult):
    # 1-to-1 correspondence in flattening grid
    # (i.e. FT gets mapped to the same index as its corresponding k-point)
    kprp_flat = kprp.reshape(-1)
    kprl_flat = kprl.reshape(-1)
    energy = (0.5*v1*np.conj(v2)).reshape(-1)  # Parseval's theorem
    # Bin energies in a given k_range
    e_hist = np.histogram2d(kprp_flat, kprl_flat, [kprp_bins, kprl_bins], weights=np.real(energy))[0]
    
    # accounting for increase in modes in kspace
    # as we move further out
    n_modes = mode_mult.sum()
    e_hist /= n_modes**2
    return e_hist

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



    
    

    