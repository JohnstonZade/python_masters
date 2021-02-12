# Code to generate a certain spectrum of Alfven waves
# referenceing alfven_wave_spectrum.cpp in Athena
# look at MCP noise spectra

import numpy as np
import numpy.random as random
import numpy.fft as fft

import diagnostics as diag


def generate_alfven(n_X, X_min, X_max, B_0):
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

    n_X = n_X[::-1]  # want in form Z, Y, X
    Ls = (X_max - X_min)[::-1]
    B_0x, B_0y, B_0z = B_0

    # trying just k-grid first
    KZ, KY, KX = diag.ft_grid('array', Ls=Ls, Ns=n_X)
    Kprl = abs(KX)
    Kprp = np.maximum(np.sqrt(abs(KY)**2 + abs(KZ)**2), 0.01)
    Kmag = np.sqrt(Kprl**2 + Kprp**2)

    # testing only -5/3 spectrum, will add others later
    exp = -7.5/3
    kpow = exp # (exp + 1) / 2

    Kspec = Kmag**kpow


    # generate random complex numbers on the grid and weight by spectrum
    r = random.normal(size=n_X)*Kspec
    theta = random.uniform(0, 2*np.pi, size=n_X)
    z = r*np.exp(1j*theta)
    ft_dB_x = z*(KY*B_0z - KZ*B_0y)  # don't need to Fourier transform B0 as it is constant
    ft_dB_y = z*(KZ*B_0x - KX*B_0z)
    ft_dB_z = z*(KX*B_0y - KY*B_0x)

    dB_x = np.real(fft.ifftn(ft_dB_x))
    dB_y = np.real(fft.ifftn(ft_dB_y))
    dB_z = np.real(fft.ifftn(ft_dB_z))

    return dB_x, dB_y, dB_z