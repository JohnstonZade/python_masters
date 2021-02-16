# Code to generate a certain spectrum of Alfven waves
# referenceing alfven_wave_spectrum.cpp in Athena
# look at MCP noise spectra

import numpy as np
import numpy.random as random
import numpy.fft as fft

import diagnostics as diag


def decompose_k(KX, KY, KZ, B0_x, B0_y, B0_z):
    # want to write k = k_prl * b_0 + k_prp
    # where b_0 = B_0 / |B_0|, k_prl = k * b_0, k_prp = k - k_prl b_0
    B0_mag = np.sqrt(B0_x**2 + B0_y**2 + B0_z**2)
    b0_x = B0_x / B0_mag
    b0_y = B0_y / B0_mag
    b0_z = B0_z / B0_mag

    Kprl = KX*b0_x + KY*b0_y + KZ*b0_z
    Kprp_x = KX - Kprl*b0_x
    Kprp_y = KY - Kprl*b0_y
    Kprp_z = KZ - Kprl*b0_z
    Kprl = abs(Kprl)
    Kprp = np.maximum(np.sqrt(abs(Kprp_x)**2 + abs(Kprp_y)**2 + abs(Kprp_z)**2), 0.01)

    return Kprl, Kprp

def run_tests(Ls, KX, KY, KZ):
    # no weighting of amplitudes by spectrum
        # just want to test if it generates modes correctly

        # parallel wavenumbers
        kx, ky, kz = 2*np.pi / Ls[2], 2*np.pi / Ls[1], 2*np.pi / Ls[0]  

        # 2D Waves along y=x: works
        # k_x = k_y = 1, k_z = 0, amp = 2, phase shift = 0
        nx, ny, nz = 1., 1., 0.
        kx *= nx; ky *= ny; kz *= nz
        amp, theta = 2., 0.

        # 2D phase shift: works
        # k_x = 1, k_y = 1, k_z = 0, amp = 2, phase shift = pi/2
        # nx, ny, nz = 0., 1., 0.
        # kx *= nx; ky *= ny; kz *= nz
        # amp, theta = 2., np.pi / 2

        # 2D parallel: works
        # k_x = 0, k_y = 1, k_z = 0, amp = 2, phase shift = 0
        # nx, ny, nz = 1., 0., 0.
        # kx *= nx; ky *= ny; kz *= nz
        # amp, theta = 2., 0. 
        # wave = amp*np.exp(1j*theta)
        # mask = (np.imag(KX) == kx) & (np.imag(KY) == ky) & (np.imag(KZ) == kz) 
        # z = np.zeros_like(KX)
        # z[mask] = wave
        # ft_dB_x = np.zeros_like(KX)
        # ft_dB_y = np.zeros_like(KX)
        # ft_dB_z = -z*KX*B_0x
        # dB_x = np.real(fft.ifftn(ft_dB_x))
        # dB_y = np.real(fft.ifftn(ft_dB_y))
        # dB_z = np.real(fft.ifftn(ft_dB_z))
        # return dB_x, dB_y, dB_z

        # 2D perpendicular: works
        # k_x = 0, k_y = 1, k_z = 0, amp = 2, phase shift = 0
        # nx, ny, nz = 0., 1., 0.
        # kx *= nx; ky *= ny; kz *= nz
        # amp, theta = 2., 0. 

        # 3D Waves: works
        # k_x = 0, k_y = k_z = 1, amp = 2, phase shift = 0
        # nx, ny, nz = 1., 1., 2.
        # kx *= nx; ky *= ny; kz *= nz
        # amp, theta = 2., 0.
        
        wave = amp*np.exp(1j*theta)
        mask = (np.imag(KX) == kx) & (np.imag(KY) == ky) & (np.imag(KZ) == kz) 
        z = np.zeros_like(KX)
        z[mask] = wave
        return z

def generate_alfven(n_X, X_min, X_max, B_0, spec_expo, run_test=0):
    '''[summary]

    Parameters
    ----------
    n_X : [type]
        [description]
    X_min : [type]
        [description]
    X_max : [type]
        [description]
    B_0 : [type]
        [description]
    '''

    # want in form Z, Y, X
    n_X = n_X[::-1]
    Ls = (X_max - X_min)[::-1]
    B0_x, B0_y, B0_z = B_0

    # grid of allowed wavenumbers corresponding to physical grid
    KZ, KY, KX = diag.ft_grid('array', Ls=Ls, Ns=n_X)
    
    # getting wave vector magntiudes parallel and perpendicular to B_0
    # if B_0 is along x direction then Kprl = KX and Kprp = sqrt(KY^2 + KZ^2)
    # added just in case we change the direction of B_0
    Kprl, Kprp = decompose_k(KX, KY, KZ, B0_x, B0_y, B0_z)
    Kmag = np.sqrt(Kprl**2 + Kprp**2)

    if run_test:
        z = run_tests(Ls, KX, KY, KZ)
    else:
        # This is the same as Jono's reasoning except he puts in 5/3 
        # if he wants a slope of -5/3 for example
        # So his would be kpow = (expo + 4) / 2 
        expo = spec_expo 
        kpow = expo - 2  # accounting for increase in k modes in shell when integrating spectrum
        kpow /= 2  # initialising B not B^2
        kpow -= 1  # accounting for cross product with k for dB perturbation
        #  equivalent to kpow = (expo - 4) / 2

        Kspec = Kmag**kpow

        # generate random complex numbers on the grid and weight by spectrum
        # these complex numbers represent the amplitude and phase of the corresponding
        # Fourier mode at that point in k-space
        r = random.normal(size=n_X)*Kspec  # multiply by number of grid points?
        theta = random.uniform(0, 2*np.pi, size=n_X)
        z = r*np.exp(1j*theta)

        # don't need to worry about excluding purely parallel waves
        # as cross product below is automatically 0
        # exclude purely perpendicular waves as they don't propagate
        # remember omega_A = k_prl * v_A
        prp_mask = (Kprl == 0.)
        z[prp_mask] = 0j
    
    # don't need to Fourier transform B0 as it is constant
    ft_dB_x = z*(KY*B0_z - KZ*B0_y)
    ft_dB_y = z*(KZ*B0_x - KX*B0_z)
    ft_dB_z = z*(KX*B0_y - KY*B0_x)

    # this generates a sum of waves of the form r*sin(k*x + theta)
    # for each point k in k-space
    dB_x = np.real(fft.ifftn(ft_dB_x))
    dB_y = np.real(fft.ifftn(ft_dB_y))
    dB_z = np.real(fft.ifftn(ft_dB_z))

    return dB_x, dB_y, dB_z