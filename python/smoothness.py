
"""
smoothness.py contains a list of functions for estimating movement smoothness. 
"""
import numpy as np


def spectral_arclength(movement, fs, window_len = 5, padlevel=4, fc=10.0, amp_th=0.05):
    pass
    N = len(movement)
    sal_arr = [];
    for i in range(N-window_len):
        win_movement = movement[i:i+window_len]
        sal, (f, Mf), (f_sel, Mf_sel) = calc_spectral_arclength(win_movement, fs)
        sal_arr.append(sal)
    return sal_arr
    

def calc_spectral_arclength(movement, fs, padlevel=4, fc=10.0, amp_th=0.05):
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
    inx = ((Mf_sel >= amp_th)*1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1]+1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate arc length
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel)/(f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    return new_sal, (f, Mf), (f_sel, Mf_sel)


# def dimensionless_jerk(movement, fs, window_len=5):
#     N = len(movement)
#     jerk = [];
#     for i in range(N-window_len):
#         win_movement = movement[i:i+window_len]
#         jerk.append(calc_dimensionless_jerk(win_movement,fs))
#     return jerk

def dimensionless_jerk(movement, fs, window_len=5):
    """
    Calculates the smoothness metric for the given speed profile using the dimensionless jerk 
    metric.
    
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.

    Returns
    -------
    dl       : float
               The dimensionless jerk estimate of the given movement's smoothness.

    Notes
    -----
    

    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> dl = dimensionless_jerk(move, fs=100.)
    >>> '%.5f' % dl
    '-335.74684'

    """
    # first enforce data into an numpy array.
    movement = np.array(movement)
    jerkarr = []

    # calculate the scale factor and jerk.
    N = len(movement)
    movement_peak = max(abs(movement))
    dt = 1./fs
    movement_dur = len(movement)*dt
    scale = pow(movement_dur, 3)/pow(movement_peak, 2)
    
    
    # jerk = np.diff(movement, 2)/pow(dt, 2)
    for i in range(N-window_len):
        win_movement = movement[i:i+window_len]
        jerk = np.diff(win_movement, 2)/pow(dt, 2)
        # Append dj
        jerkarr.append(- scale * sum(pow(jerk, 2)) * dt)

    # estimate dj
    return jerkarr


def log_dimensionless_jerk(movement, fs, window_len=5):
    """
    Calculates the smoothness metric for the given speed profile using the log dimensionless jerk 
    metric.
    
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.

    Returns
    -------
    ldl      : float
               The log dimensionless jerk estimate of the given movement's smoothness.

    Notes
    -----
    

    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> ldl = log_dimensionless_jerk(move, fs=100.)
    >>> '%.5f' % ldl
    '-5.81636'

    """
    return -np.log(list(map(abs,dimensionless_jerk(movement, fs, window_len))))


