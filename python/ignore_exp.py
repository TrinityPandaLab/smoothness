# -*- coding: utf-8 -*-
import numpy as np

padlevel=4
fc=10.0
amp_th=0.05

fs = 10
movv = np.array([-1.,  1.,  1.,  1.,  0.,  1.,  1., -1., -1., -1.,  0., -1., -1.,
        1., -1., -1.,  1.,  0.,  1.,  1.,  0.,  0.,  1., -1.,  1.,  0.,
        0.,  0., -1.,  1., -1.,  0.,  1.,  1., -1.,  0., -1., -1.,  1.,
       -1.,  1.,  0.,  1.,  0., -1.,  0., -1., -1., -1.,  1.,  1., -1.,
       -1.,  0.,  1.,  0.,  1., -1.,  1.,  1.,  0.,  0., -1., -1.,  1.,
        1., -1., -1.,  1.,  0.,  1.,  1., -1.,  1.,  0.,  1.,  0., -1.,
       -1.,  0., -1.,  0.,  0.,  1., -1.,  0.,  0.,  1., -1.,  0., -1.,
        0.,  1.,  1.,  1.,  1.,  1., -1., -1.,  1.,  0.,  1., -1.,  0.,
       -1.,  0., -1.,  1.,  1.,  1., -1., -1.,  0., -1., -1.,  0.,  0.,
        0.,  0.,  0.,  1.,  1.,  0., -1.,  1., -1.,  0.,  1.,  0.,  0.,
        1.,  1.,  0.,  0.,  0., -1., -1.,  1.,  0.,  1.,  1.,  0.,  0.,
        1., -1., -1., -1., -1., -1.,  1.,  1.,  1., -1.,  0.,  1.,  1.,
       -1., -1., -1., -1., -1.,  0.,  1.,  1.,  0., -1.,  0., -1.,  0.,
       -1., -1.,  1.,  1.,  0.,  0.,  1., -1.,  1.,  1., -1.,  1., -1.,
        1., -1.,  0.,  0.,  1., -1., -1.,  0., -1.,  1.,  0.,  0.,  0.,
       -1.,  0.,  1., -1.,  1., -1.,  0., -1.,  1.,  0.,  1., -1., -1.,
        0.,  1.,  1.,  0.,  1.,  1., -1.,  0.,  1.,  1., -1.,  0.,  0.,
        1.,  0., -1.,  0.,  1.,  1.,  0.,  0., -1.,  0., -1., -1.,  1.,
       -1., -1.,  0.,  1.,  0.,  0.,  1., -1.,  1.,  0., -1.,  0.,  1.,
       -1.,  0.,  0.,  0.,  1.,  0., -1.,  0., -1.,  0., -1.,  1.,  1.,
        1.,  1., -1., -1., -1.,  1., -1.,  1.,  1.,  1., -1.,  0.,  1.,
       -1.,  0.,  1., -1., -1.,  1.,  1., -1., -1.,  0.,  0.,  1.,  0.,
       -1.,  0., -1., -1.,  1.,  0.,  1.,  1., -1., -1., -1.,  0., -1.,
        1.,  1.,  0.,  1.,  0.,  0., -1.,  1.,  0., -1.,  0.,  1., -1.,
        1.,  1.,  0., -1., -1.,  0., -1.,  1.,  0., -1., -1.,  1.,  0.,
        1.,  1.,  0.,  1.,  1.,  0., -1.,  1., -1.,  0., -1.,  1., -1.,
        0.,  0., -1.,  0., -1.,  0.,  0., -1.,  1.,  1.,  0.,  0.,  0.,
       -1., -1., -1.,  1.,  1.,  1.,  0.,  0., -1.,  1.,  1.,  0., -1.,
        0.,  1.,  0.,  1., -1.,  1., -1.,  0., -1.,  0.,  1.,  0.,  1.,
       -1.,  0., -1.,  0.,  0.,  0.,  1., -1.,  0.,  1.,  0.,  1., -1.,
       -1.,  0., -1.,  0.,  0.,  1., -1.,  1.,  0., -1.,  1., -1., -1.,
        0.,  1.,  0.,  1.,  1.,  0.,  0.,  1., -1.,  1.,  0., -1., -1.,
        0.,  1.,  1., -1.,  0.,  1.,  0.,  0.,  0., -1.,  0.,  1., -1.,
       -1.,  1., -1., -1.,  0.,  1.,  1., -1.,  1.,  0.,  0., -1.,  1.,
       -1.,  0., -1., -1., -1.,  1.,  1.,  1., -1.,  1.,  0., -1.,  1.,
        1.,  0.,  0., -1., -1.,  0.,  1.,  1.,  0.,  0.,  1., -1., -1.,
       -1.,  1.,  0., -1.,  0.,  1.,  1., -1., -1.,  1., -1., -1.,  1.,
        0.,  1.,  0., -1., -1., -1., -1.,  1., -1.,  1.,  0.,  1.,  0.,
        1.,  1., -1.,  0.,  1.,  0.,  0.,  1., -1., -1., -1., -1.,  0.,
       -1.,  0.,  1., -1., -1., -1.,  0., -1., -1., -1.,  0.,  1.,  1.,
       -1.,  1.,  0.,  1., -1., -1.,  0., -1.,  1., -1., -1., -1., -1.,
       -1., -1., -1.,  1.,  0.,  1., -1.,  1.,  0., -1.,  0., -1.,  0.,
       -1.,  1.,  0., -1., -1.,  0.,  1.,  1., -1.,  0.,  0.,  1.,  0.,
       -1.,  1., -1., -1.,  0.,  1.,  0.,  0.,  0., -1., -1., -1.,  0.,
        0., -1., -1., -1.,  1., -1.,  0.,  1., -1., -1.,  1.,  0.,  1.,
        1.,  1., -1., -1.,  1.,  0.,  0.,  1.,  1., -1.,  1.,  0., -1.,
        1.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  1., -1.,  1., -1., -1.,
       -1.,  0.,  1.,  0.,  0., -1.,  0., -1.,  0., -1., -1., -1.,  0.,
        1.,  0.,  0., -1.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  1.,
        0., -1., -1.,  1.,  0., -1.,  0.,  1.,  1.,  0.,  0.,  0., -1.,
        0.,  1.,  1., -1., -1., -1., -1., -1., -1.,  1.,  0.,  1.,  1.,
       -1., -1., -1., -1., -1.,  0.,  0.,  0., -1.,  0., -1.,  1.,  0.,
        1.,  1.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1., -1.,  1.,  1.,
        1.,  1.,  1., -1., -1.,  1.,  1., -1.,  0.,  1.,  1.,  0., -1.,
       -1.,  0., -1.,  0., -1., -1.,  1.,  1.,  1.,  1., -1.,  0.,  0.,
       -1.,  0.,  0., -1.,  0.,  0., -1., -1.,  1.,  0., -1., -1.,  1.,
       -1.,  1.,  1., -1.,  1., -1.,  1.,  0.,  1.,  0.,  0., -1.,  0.,
        1.,  1.,  1.,  0., -1., -1.,  1.,  0.,  1.,  1., -1.,  1.,  1.,
       -1., -1.,  0.,  1.,  1.,  1.,  0., -1., -1.,  1., -1.,  1., -1.,
        1.,  0.,  0.,  0., -1., -1.,  0.,  0., -1.,  0.,  1.,  1., -1.,
        0., -1.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  1.,  1., -1.,
       -1.,  1.,  1.,  0.,  0.,  0., -1., -1.,  1., -1.,  0., -1.,  1.,
        0.,  0.,  1.,  1., -1.,  0.,  1.,  0.,  0.,  1.,  1., -1., -1.,
        1.,  1.,  0.,  1., -1., -1., -1.,  0.,  1.,  1.,  1., -1.,  0.,
       -1., -1., -1., -1.,  1., -1.,  1., -1.,  0., -1.,  1.,  0.,  1.,
       -1.,  1.,  0., -1., -1.,  1.,  1., -1.,  1.,  0.,  0.,  0.,  0.,
       -1.,  1., -1.,  0.,  1., -1.,  1.,  0.,  1.,  1., -1.,  0.,  0.,
        0.,  1.,  1.,  1., -1., -1.,  0.,  1.,  1., -1.,  1.,  0., -1.,
        1.,  1.,  0., -1.,  1., -1., -1.,  1.,  1.,  0.,  1.,  1.,  0.,
        1.,  0.,  1.,  0.,  1.,  1.,  0., -1., -1.,  0., -1.,  0.,  1.,
       -1.,  1., -1.,  0., -1.,  1., -1.,  1., -1.,  1.,  1.,  1., -1.,
        1.,  0.,  0., -1.,  1.,  0.,  0.,  1.,  0., -1.,  1.,  0.,  1.,
        1.,  0., -1.,  1.,  1.,  1., -1.,  0.,  0.,  1.,  1.,  1.,  0.,
        0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  1., -1.,  0., -1.,
       -1.,  0.,  1.,  1.,  1., -1.,  0.,  1., -1.,  0.,  1.,  1.,  0.,
        0.,  0., -1.,  1.,  0.,  0., -1., -1., -1.,  1.,  0.,  1.,  0.,
        0.,  0.,  1., -1.,  1., -1., -1.,  0., -1.,  0., -1., -1.])


movement = movv
"""
Calcualtes the smoothness of the given speed profile using the modified spectral
arc length metric.

Parameters
----------
movement : np.array
           The array containing the movement speed profile.
fs       : float
           The sampling frequency of the data.
padlevel : integer, optional
           Indicates the amount of zero padding to be done to the movement
           data for estimating the spectral arc length. [default = 4]
fc       : float, optional
           The max. cut off frequency for calculating the spectral arc
           length metric. [default = 10.]
amp_th   : float, optional
           The amplitude threshold to used for determing the cut off
           frequency upto which the spectral arc length is to be estimated.
           [default = 0.05]

Returns
-------
sal      : float
           The spectral arc length estimate of the given movement's
           smoothness.
(f, Mf)  : tuple of two np.arrays
           This is the frequency(f) and the magntiude spectrum(Mf) of the
           given movement data. This spectral is from 0. to fs/2.
(f_sel, Mf_sel) : tuple of two np.arrays
                  This is the portion of the spectrum that is selected for
                  calculating the spectral arc length.

Notes
-----
This is the modfieid spectral arc length metric, which has been tested only
for discrete movements.
It is suitable for movements that are a few seconds long, but for long
movements it might be slow and results might not make sense (like any other
smoothness metric).

Examples
--------
>>> t = np.arange(-1, 1, 0.01)
>>> move = np.exp(-5*pow(t, 2))
>>> sal, _, _ = spectral_arclength(move, fs=100.)
>>> '%.5f' % sal
'-1.41403'

"""
# Number of zeros to be padded.
nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

# Frequency
f = np.arange(0, fs, fs/nfft)
# Normalized magnitude spectrum
Mf = abs(np.fft.fft(movement, nfft))
Mf = Mf/max(Mf)

# Indices to choose only the spectrum within the given cut off frequency Fc.
# NOTE: This is a low pass filtering operation to get rid of high frequency
# noise from affecting the next step (amplitude threshold based cut off for
# arc length calculation).
fc_inx = ((f <= fc)*1).nonzero()
f_sel = f[fc_inx]
Mf_sel = Mf[fc_inx]

# Choose the amplitude threshold based cut off frequency.
# Index of the last point on the magnitude spectrum that is greater than
# or equal to the amplitude threshold.
inx = ((Mf_sel >= amp_th)*1).nonzero()
inx = inx[0]
fc_inx = range(inx[0], inx[-1]+1)
f_sel = f_sel[fc_inx]
Mf_sel = Mf_sel[fc_inx]

# Calculate arc length
new_sal = -sum(np.sqrt(pow(np.diff(f_sel)/(f_sel[-1] - f_sel[0]), 2) +
                       pow(np.diff(Mf_sel), 2)))
# return new_sal, (f, Mf), (f_sel, Mf_sel)
    # return new_sal, (f, Mf), (f_sel, Mf_sel)