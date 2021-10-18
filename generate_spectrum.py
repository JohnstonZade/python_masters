import numpy as np
import numpy.random as random
import numpy.fft as fft

import diagnostics as diag


def decompose_k(KX, KY, KZ, B0_x, B0_y, B0_z):
    # want to write k = k_prl * b_0 + k_prp
    # where b_0 = B_0 / |B_0|, k_prl = k ⋅ b_0, k_prp = k - k_prl b_0
    B0_mag = np.sqrt(B0_x**2 + B0_y**2 + B0_z**2)
    b0_x   = B0_x / B0_mag
    b0_y   = B0_y / B0_mag
    b0_z   = B0_z / B0_mag

    Kprl   = KX*b0_x + KY*b0_y + KZ*b0_z
    Kprp_x = KX - Kprl*b0_x
    Kprp_y = KY - Kprl*b0_y
    Kprp_z = KZ - Kprl*b0_z
    Kprl   = abs(Kprl)
    Kprp   = np.sqrt(abs(Kprp_x)**2 + abs(Kprp_y)**2 + abs(Kprp_z)**2)

    return Kprl, Kprp


def get_kpow(expo, expo_prl):
    # accounting for how volume in k space changes
    # for expo_prl = -2 (i.e. k_prl ∝ (k_prp)^(2/3)) second term is equivalent to expo + 1 + 2/3
    # for isotropic expo_prl = expo (k_prl ∝ k_prp) second term is zero
    kpow = expo + 2.0
    kpow += (expo - expo_prl) / (expo_prl - 1.) if expo_prl != 1. else 0.
    kpow /= 2  # initialising B not B^2
    return kpow

def gaussian_spec(Kprl, Kprp, kprl0, kprp0, kwidth):
    # A Gaussian spectrum of the form
    # exp[(-(kprp - kprp0)^2 - (kprl - kprl0)^2) / kwidth]
    # i.e. a Gaussian peak centred at (kprp0, kprl0) in kprp,kprl space
    # with isotropic width
    return np.exp( -((Kprl - kprl0)**2 + (Kprp - kprp0)**2) / kwidth**2)

def powerlaw_spec(expo, expo_prl, Kprl, Kprp, Kmag, spectrum):
    kpow = get_kpow(expo, expo_prl)
    factor = 1.0
    if spectrum == 'anisotropic':
        # gives 2/3 for expo, expo_prl = -5/3, -2
        kprp_exp = (expo - 1) / (expo_prl - 1) 
        # kprl_exp = 1.0  # always 1
        factor *= np.exp(-Kprl / (Kprp**kprp_exp)) # see Cho2002, Maron2001 for explaination
    spec = Kprp if spectrum == 'anisotropic' else Kmag
    return 1 / (1 + spec**kpow) * factor
    

def truncate_r(r, n_X, Ls, KX, KY, KZ, n_cutoff):
    # Just a check in case I forget to add specific parameters
    n_low, n_high = (0, n_X[2]/2) if n_cutoff is None else n_cutoff

    NX = Ls[2]*np.imag(KX) / (2*np.pi)
    NY = Ls[1]*np.imag(KY) / (2*np.pi)
    NZ = Ls[0]*np.imag(KZ) / (2*np.pi)
    Nsqr = NX**2 + NY**2 + NZ**2

    # using same conditionals as used in alfven_wave_spectrum problem gen
    c1 = (abs(NX) >= n_low) & (abs(NY) >= n_low) & (abs(NZ) >= n_low)
    c2 = (abs(NX) <= n_high) & (abs(NY) <= n_high) & (abs(NZ) <= n_high)
    c3 = Nsqr < n_high**2
    # cut off all wavevectors that DON'T satisfy c1, c2, and c3
    mode_mask = np.logical_not(c1 & c2 & c3)
    r[mode_mask] = 0.0
    c1, c2, c3 = None, None, None
    NX, NY, NZ, Nsqr = None, None, None, None
    return r

def generate_alfven_spectrum(n_X, X_min, X_max, B_0, spectrum, expo=-5/3, expo_prl=-2.0, 
                             kpeak=(2, 2), kwidth=12.0, do_truncation=0, n_cutoff=None, run_test=0):
    '''Generate a superposition of random Alfvén waves within a numerical domain
    that follow a given energy spectrum.

    This function creates a grid in Fourier (k) space corresponding to the allowed wave modes
    in the numerical grid and generates modes with random amplitudes and phases. The amplitudes
    are scaled by the corresponding k spectrum. Alfvén wave modes are then calculated, which can then be
    used by `generate_ics` to setup initial conditons.

    Spectra
    -------
    The energy spectra this code can generate includes:

    - Isotropic: generates a spectrum of the form E(k) ~ k^(- |expo|)

    - Anisotropic: for expo = α and expo_prl = β, generates anisotropic
    1D spectra of the form E(kprp) ~ kprp^(- |α|) and E(kprl) ~ kprl^(- |β|).

           - For critical balance, α = -5/3 and β = -2 satisfying the relationship
           kprl ∝ kprp^(2/3).
           
           - In general, for kprp^(- |α|) and kprl^(- |β|) gives the relationship
           kprl ∝ kprp^[(|α| - |β|)/(|β| - 1)], which can be derived from
           the 3D energy spectrum.

    - Gaussian: generates an isotropic spectrum of the form E(k) ~ e^(-k^2 / kwidth^2)

    Note: The exponent for all power laws are always assumed to be negative as a positive power law is unphysical
    as it would mean that smaller scales have much more energy than large scales.


    Parameters
    ----------
    n_X : ndarray
        array containing the resolution of the grid in the order n_x, n_y, n_z
    X_min : ndarray
        array containing the minimum coordinate values of x, y, z
    X_max : ndarray
        array containing the maximum coordinate values of x, y, z
    B_0 : ndarray
        ndarray the size of the grid that contains the mean magnetic field
    expo : float
        exponent for the power law energy spectrum E(k)~k^(expo) 
        (or E(kprp)~kprp^(expo) for anisotropic spectrum)
    expo_prl : float, optional
        the exponent to raise kprl by for an anisotropic spectrum, by default -2.0
    spectrum : string, optional
        the spectrum to be generated, by default isotropic
    kpeak : tuple, optional
        the modes that the Gaussian spectrum peaks at in k_⟂k||-space, by default (2, 2)
    width: float, optional
        the scale width of the Gaussian distribution, by default 12.0
    do_truncation: boolean, optional
        setting to truncate modes outside of n_cutoff, by default 0
    n_cutoff: tuple, optional
        tuple of modes to keep, with all outside modes being cutoff, by default None
    run_test : boolean, optional
        run tests to check mode generation, by default 0

    Returns
    -------
    ndarray
        Returns the 3 components of the superposed Alfvén waves dB_x, dB_y, dB_z
        which have the same size as the original grid.
    '''
    if spectrum not in ['gaussian', 'isotropic', 'anisotropic']:
        raise ValueError(spectrum + ' is not a valid spectrum')
    
    # want in form Z, Y, X conforming to Athena++
    n_X = n_X[::-1]
    Ls = (X_max - X_min)[::-1]
    B0_x, B0_y, B0_z = B_0

    # grid of allowed wavenumbers corresponding to physical grid
    # isotropic to box by default
    KZ, KY, KX = diag.ft_grid('array', Ls=Ls, Ns=n_X)

    # getting wave vector magntiudes parallel and perpendicular to B_0
    # if B_0 is along x direction then Kprl = KX and Kprp = √(KY^2 + KZ^2)
    # added just in case we change the direction of B_0
    # Kprl, Kprp = decompose_k(KX, KY, KZ, B0_x, B0_y, B0_z)

    # Assuming that B_0 is along x-axis (or close to it) initially
    Kprl = abs(KX)
    Kprp = np.sqrt(abs(KY)**2 + abs(KZ)**2)
    Kmag = np.maximum(np.sqrt(Kprl**2 + Kprp**2), 1e-15)

    if run_test:
        z = run_tests(Ls, KX, KY, KZ)
    else:
        if spectrum == 'gaussian':
            κ_prl, κ_prp = kpeak
            # Assuming box normalized spectra here
            kprl0 = κ_prl * 2*np.pi
            kprp0 = κ_prp * 2*np.pi
            Kspec = gaussian_spec(Kprl, Kprp, kprl0, kprp0, kwidth)
        else:
            # making it easier to compare to Jono's/Athena's code
            # will always interpret as a spectrum of the form k^(-expo)
            expo = abs(expo)
            expo_prl = expo if spectrum == 'isotropic' else abs(expo_prl)
            Kspec = powerlaw_spec(expo, expo_prl, Kprl, Kprp, Kmag, spectrum)

        # generate random complex numbers on the grid and weight by spectrum
        # these complex numbers represent the amplitude (r) and phase (theta) of the corresponding
        # Fourier mode at that point in k-space
        r = random.normal(size=n_X)*Kspec
        Kspec, Kmag = None, None
        if do_truncation:
            r = truncate_r(r, n_X, Ls, KX, KY, KZ, n_cutoff)

        theta = random.uniform(0, 2*np.pi, size=n_X)
        z = r*np.exp(1j*theta)
        r, theta = None, None

        # excluding purely parallel waves
        if KZ.shape[0] > 1:
            prl_mask = (Kprp == 0.) | (KY == 0.) | (KZ == 0.)
        else:
            prl_mask = (Kprp == 0.) | (KY == 0.)
        z[prl_mask] = 0j
        # exclude purely perpendicular waves as they don't propagate
        # remember ω_A = k_prl * v_A
        prp_mask = (Kprl == 0.)
        z[prp_mask] = 0j
        Kprl, Kprp = None, None


    # Alfvén wave definition performed in k-space: δB = k × B
    # don't need to Fourier transform B0 as it is constant
    ft_dB_x = (KY*B0_z - KZ*B0_y)
    ft_dB_y = (KZ*B0_x - KX*B0_z)
    ft_dB_z = (KX*B0_y - KY*B0_x)
    KX, KY, KZ = None, None, None
    B0_x, B0_y, B0_z = None, None, None

    # rescaling the amplitude of the Alfvén waves by
    # removing the magnitude of the k vector and replacing
    # it with the randomly generated amplitude.
    # This removes the need to add 1 to kpow above.
    ft_dB_mag = np.sqrt(abs(ft_dB_x)**2 + abs(ft_dB_y)**2 + abs(ft_dB_z)**2)
    #ft_dB_mag = np.sqrt(abs(ft_dB_y)**2 + abs(ft_dB_z)**2)
    ft_dB_mag[ft_dB_mag == 0.] = 1e-15
    ft_dB_x *= z / ft_dB_mag
    ft_dB_y *= z / ft_dB_mag
    ft_dB_z *= z / ft_dB_mag
    ft_dB_mag, z = None, None

    # The IFT in Python is normalized by the total number of grid points.
    # This scales down the amplitude by a factor of Nx*Ny*Nz.
    # Verified after checking amplitude dependence on resolution for a single mode 
    # i.e. 256^3 box has smaller amplitude than 32^3 box on the order of 10^-3 (using 2^(3x)~10^x)
    # Just inverting this process.
    N_points = np.prod(n_X)
    ft_dB_x *= N_points
    ft_dB_y *= N_points
    ft_dB_z *= N_points

    # this generates a sum of waves of the form r*sin(k⋅x + theta)
    # for each point k in k-space
    dB_x = np.real(fft.ifftn(ft_dB_x))
    dB_y = np.real(fft.ifftn(ft_dB_y))
    dB_z = np.real(fft.ifftn(ft_dB_z))
    ft_dB_x, ft_dB_y, ft_dB_z = None, None, None

    return dB_x, dB_y, dB_z
    #return dB_y, dB_z


def run_tests(Ls, KX, KY, KZ, n=0, B_0x=1.0):
    # no weighting of amplitudes by spectrum
    # just want to test if it generates modes correctly
    # parallel wavenumbers
    kx, ky, kz = 2*np.pi / Ls[2], 2*np.pi / Ls[1], 2*np.pi / Ls[0]  
    if n == 0:
        # 2D Waves along y=x: works
        # k_x = k_y = 1, k_z = 0, amp = 2, phase shift = 0
        nx, ny, nz = 1., 1., 0.
        kx *= nx; ky *= ny; kz *= nz
        amp, theta = 0.01, 0.
    elif n == 1:
        # 2D phase shift: works
        # k_x = 1, k_y = 1, k_z = 0, amp = 2, phase shift = pi/2
        nx, ny, nz = 0., 1., 0.
        kx *= nx; ky *= ny; kz *= nz
        amp, theta = 2., np.pi / 2
    elif n == 2:
        # 2D parallel: works
        # k_x = 0, k_y = 1, k_z = 0, amp = 2, phase shift = 0
        nx, ny, nz = 1., 0., 0.
        kx *= nx; ky *= ny; kz *= nz
        amp, theta = 2., 0. 
        wave = amp*np.exp(1j*theta)
        mask = (np.imag(KX) == kx) & (np.imag(KY) == ky) & (np.imag(KZ) == kz) 
        z = np.zeros_like(KX)
        z[mask] = wave
        ft_dB_x = np.zeros_like(KX)
        ft_dB_y = np.zeros_like(KX)
        ft_dB_z = -z*KX*B_0x
        dB_x = np.real(fft.ifftn(ft_dB_x))
        dB_y = np.real(fft.ifftn(ft_dB_y))
        dB_z = np.real(fft.ifftn(ft_dB_z))
        return dB_x, dB_y, dB_z
    elif n == 3:
        # 2D perpendicular: works
        # # k_x = 0, k_y = 1, k_z = 0, amp = 2, phase shift = 0
        nx, ny, nz = 0., 1., 0.
        kx *= nx; ky *= ny; kz *= nz
        amp, theta = 2., 0. 
    elif n == 4:
        # 3D Waves: works
        # k_x = 0, k_y = k_z = 1, amp = 2, phase shift = 0
        nx, ny, nz = 1., 1., 2.
        kx *= nx; ky *= ny; kz *= nz
        amp, theta = 2., 0.
    
    wave = amp*np.exp(1j*theta)
    mask = (np.imag(KX) == kx) & (np.imag(KY) == ky) & (np.imag(KZ) == kz) 
    z = np.zeros_like(KX)
    z[mask] = wave
    return z