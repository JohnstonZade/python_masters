'''Code to calculate the energy spectrum of turbulence
   in a fluid simulation run in Athena++.
'''
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import diagnostics as diag
from sklearn.linear_model import LinearRegression
import scipy.integrate as integrate
from matplotlib import rc
rc('text', usetex=True)  # LaTeX labels
default_prob = diag.DEFAULT_PROB


def array_avg(arr):
    return 0.5*(arr[1:] + arr[:-1])


def calc_spectrum(output_dir, save_dir, return_dict=1, prob=default_prob,
                  dict_name='mhd_spec', do_single_file=0, n=0,
                  normalize_energy=1, bmag_and_rho=0, method='matt'):

    # Getting turnover time and converting to file number
    if do_single_file:
        nums = range(n, n+1)
    else:
        max_n = diag.get_maxn(output_dir) // 2
        nums = range(max_n)
    # nums = range(tau_file, max_n) # average over last 2 alfven periods

    do_full_calc = not diag.check_dict(save_dir, dict_name)
    if do_full_calc:
        # create grid of K from first time step
        data = diag.load_data(output_dir, n, prob=prob, method=method)
        KZ, KY, KX = diag.ft_grid('data', data=data, prob=prob)
        Kprl = np.maximum(np.abs(KX), 1e-4)
        Kprp = np.maximum(np.sqrt(abs(KY)**2 + abs(KZ)**2), 1e-4)
        Kmag = np.sqrt(Kprl**2+Kprp**2)
        Npoints = Kmag.size  # needed for FFT normalization

        abs_KY, abs_KZ = abs(KY), abs(KZ)
        Kmag_mult, Kmag_grid, Kmag_bins = get_k_bins(Kmag, abs_KY, abs_KZ, 3)
        Kprp_mult, Kprp_grid, Kprp_bins = get_k_bins(Kprp, abs_KY, abs_KZ, 2)
        Kprl_mult, Kprl_grid, Kprl_bins = get_k_bins(Kprl, abs_KY, abs_KZ, 1)

        # Dictionary to hold spectrum information
        S = {}
        grids, bins, norms = {}, {}, {}
        grids['Kmag'], grids['Kprl'], grids['Kprp'] = Kmag_grid, Kprl_grid, Kprp_grid
        bins['Kmag'], bins['Kprl'], bins['Kprp'] = Kmag_bins, Kprl_bins, Kprp_bins
        norms['Kmag'], norms['Kprl'], norms['Kprp'] = Kmag_mult, Kprl_mult, Kprp_mult
        S['grids'], S['bins'], S['norms'] = grids, bins, norms

        ns = 0  # counter
        fields = ['vel1', 'vel2', 'vel3', 'Bcc1', 'Bcc2', 'Bcc3',
                  'EK', 'EK_prlbox', 'EK_prpbox', 'EK_prpfluc', 'EK_prpfluc_prlbox', 'EK_prpfluc_prpbox', 'EK_prpfluc_2D',
                  'EM', 'EM_prlbox', 'EM_prpbox', 'EM_prpfluc', 'EM_prpfluc_prlbox', 'EM_prpfluc_prpbox', 'EM_prpfluc_2D',
                  'E+_prpfluc_prlbox', 'E+_prpfluc_prpbox', 'E+_prpfluc_2D', 'E-_prpfluc_prlbox', 'E-_prpfluc_prpbox', 'E-_prpfluc_2D']
        if bmag_and_rho:
            fields.append('B')
            fields.append('rho')
        # Initializing variable fields in spectrum dictionary
        for var in fields:
            S[var] = 0

        for n in nums:
            # data is already loaded for a single file
            if not do_single_file:
                try:
                    data = diag.load_data(output_dir, n, prob=prob, method=method)
                except IOError:
                    print('Could not load file', n)
                    break

            # Take the Fourier transform of the individual components
            # Find their energy (i.e. Parseval's theorem)
            # Add to total energy spectrum
            for vel in fields[:3]:
                print('Doing ' + vel + ' spectrum')
                v = data[vel]
                ft = fft.fftn(v) / Npoints
                iso = spec1D(ft, Kmag, Kmag_bins, Kmag_mult)
                prlbox = spec1D(ft, Kprl, Kprl_bins, Kprl_mult)
                prpbox = spec1D(ft, Kprp, Kprp_bins, Kprp_mult)
                # Perpendicular fluctuation spectra
                S['EK_prpfluc'] += iso
                S['EK_prpfluc_prlbox'] += prlbox
                S['EK_prpfluc_prpbox'] += prpbox
                # 2D (k_prl and k_prl) spectrum
                S['EK_prpfluc_2D']  += spec2D(ft, Kprp, Kprl, Kprp_bins, Kprl_bins, Kprp_mult)

            Bmag = 0
            for Bcc in fields[3:6]:
                print('Doing ' + Bcc + ' spectrum')
                B = data[Bcc]
                ft = fft.fftn(B - np.mean(B)) / Npoints
                iso = spec1D(ft, Kmag, Kmag_bins, Kmag_mult)
                prlbox = spec1D(ft, Kprl, Kprl_bins, Kprl_mult)
                prpbox = spec1D(ft, Kprp, Kprp_bins, Kprp_mult)
                # Isotropic spectra
                # S[Bcc] += iso
                # S['EM'] += iso  # Total spectrum is sum of each component
                # # Box-parallel (along x-axis) spectrum
                # S['EM_prlbox'] += prlbox
                # # Box-perpendicular spectrum
                # S['EM_prpbox'] += prpbox
                # Perpendicular fluctuation spectra
                S['EM_prpfluc'] += iso
                S['EM_prpfluc_prlbox'] += prlbox
                S['EM_prpfluc_prpbox'] += prpbox
                # 2D (k_prl and k_prl) spectrum
                S['EM_prpfluc_2D']  += spec2D(ft, Kprp, Kprl, Kprp_bins, Kprl_bins, Kprp_mult)
                Bmag += B**2
                
                        
            # Elsasser spectra
            rho = data['rho']
            for i in range(1,4):
                u = data['vel'+str(i)]
                B = data['Bcc'+str(i)]
                b = (B - np.mean(B)) / np.sqrt(rho)  # perpendicular mag fluc in velocity units
                for idx, name in enumerate(['E+_', 'E-_']):
                    z = u + ((-1)**idx)*b # z+ for idx = 0, z- otherwise
                    ft_z = fft.fftn(z) / Npoints
                    iso = spec1D(ft_z, Kmag, Kmag_bins, Kmag_mult)
                    prlbox = spec1D(ft_z, Kprl, Kprl_bins, Kprl_mult)
                    prpbox = spec1D(ft_z, Kprp, Kprp_bins, Kprp_mult)
                    S[name+'prpfluc_prlbox'] += prlbox
                    S[name+'prpfluc_prpbox'] += prpbox
                    # 2D (k_prl and k_prl) spectrum
                    S[name+'prpfluc_2D']  += spec2D(ft, Kprp, Kprl, Kprp_bins, Kprl_bins, Kprp_mult)

            if normalize_energy:
                # v_A ~ a^(-1) ⟹ (v_A)^2 ∼ a^(-2), assuming v_A0 = 1 
                # (above only for purely radial fields, this is more general)
                rho = data['rho']
                B = np.array((data['Bcc1'], data['Bcc2'], data['Bcc3']))
                v_A = diag.alfven_speed(rho, B)
                for key in S:
                    if ('EK' in key) or ('E+' in key) or ('E-' in key):
                        S[key] /= v_A**2

                # B_x ∼ a^(-2) ⟹ (B_x)^2 ∼ a^(-4), assuming ⟨B_x0⟩=1
                # (above only for purely radial fields, this is more general)
                B_0 = diag.get_mag(diag.box_avg(B), axis=0)  # single time entry
                for key in S:
                    if 'EM' in key:
                        S[key] /= B_0**2
            
            if bmag_and_rho:
                Bmag = np.sqrt(Bmag)
                ft_Bmag = fft.fftn(Bmag)
                S['B'] += spec1D(ft_Bmag, ft_Bmag, Kmag, Kmag_bins, Kmag_mult) 
                ft_rho = fft.fftn(data['rho'] - np.mean(data['rho']))
                S['rho'] += spec1D(ft_rho, ft_rho, Kmag, Kmag_bins, Kmag_mult)

            ns += 1

        if not do_single_file:
            # Average over number of times done
            for var in fields:
                S[var] /= ns
            S['nums'] = nums

        if return_dict:
            return S
        else:
            diag.save_dict(S, save_dir, dict_name)
    else:
        S = diag.load_dict(save_dir, dict_name)

    


def get_k_bins(k, ky, kz, dim):
    k_flat, ky_flat, kz_flat = k.reshape(-1), ky.reshape(-1), kz.reshape(-1)
    k_flat = k_flat[(ky_flat != 0.) & (kz_flat != 0.)]
    k_min, k_max = 0.5*k[k > 0].min(), k.max()
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), 2000)
    # multiplicity of a mode (number of times we see that wavenumber)
    k_hist = np.histogram(k_flat, k_bins)[0]
    
    # Removing bins that have no modes in them (essentially widening the bins)
    zero_mask = np.where((k_hist == 0))
    mode_mult = np.delete(k_hist, zero_mask)
    k_bins = np.hstack((np.delete(k_bins[:-1], zero_mask), k_bins[-1]))
    k_grid = array_avg(k_bins)
    
    # Normalization
    mode_mult = mode_mult / k_grid**(dim - 1)  # accounting for number of modes in a shell in kspace
    mode_mult /= mode_mult[0]
    return mode_mult, k_grid, k_bins

def spec1D(v, k, k_bins, mode_norm):
    # Note: Only worried about autocorrelation
    # need to change v -> v1,v2 for more general spectra
    # 1-to-1 correspondence in flattening grid
    # (i.e. FT gets mapped to the same index as its corresponding k-point)
    k_flat = k.reshape(-1)
    # v has already been fft normalized
    energy = 0.5*(np.abs(v)**2).reshape(-1)  # v*conj(v) = |v|^2
    # Bin energies in a given k_range
    e_hist = np.histogram(k_flat, k_bins, weights=energy)[0]

    # mode per bin normalization
    e_hist /= mode_norm
    
    return e_hist

def spec2D(v, kprp, kprl, kprp_bins, kprl_bins, mode_norm):
    # 1-to-1 correspondence in flattening grid
    # (i.e. FT gets mapped to the same index as its corresponding k-point)
    kprp_flat = kprp.reshape(-1)
    kprl_flat = kprl.reshape(-1)
    energy = 0.5*(np.abs(v)**2).reshape(-1)  # v*conj(v) = |v|^2
    # Bin energies in a given k_range
    e_hist = np.histogram2d(kprp_flat, kprl_flat, [kprp_bins, kprl_bins], weights=energy)[0]

    # mode per bin normalization
    # Only need to normalize by kprp modes as number of kprl modes constant
    e_hist /= mode_norm.reshape(mode_norm.size, 1)
    
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
    
    

    