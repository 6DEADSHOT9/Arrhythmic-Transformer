import pywt
import numpy as np


def calc_baseline(signal):
    """
    Calculate the baseline of signal.

    Args:
        signal (numpy 1d array): signal whose baseline should be calculated


    Returns:
        baseline (numpy 1d array with same size as signal): baseline of the signal
    """
    ssds = np.zeros((3))

    cur_lp = np.copy(signal)
    iterations = 0
    while True:
        # Decompose 1 level
        lp, hp = pywt.dwt(cur_lp, "db4")

        # Shift and calculate the energy of detail/high pass coefficient
        ssds = np.concatenate(([np.sum(hp ** 2)], ssds[:-1]))

        # Check if we are in the local minimum of energy function of high-pass signal
        if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
            break

        cur_lp = lp[:]
        iterations += 1

    # Reconstruct the baseline from this level low pass signal up to the original length
    baseline = cur_lp[:]
    for _ in range(iterations):
        baseline = pywt.idwt(baseline, np.zeros((len(baseline))), "db4")

    return signal-baseline[: len(signal)]

def normalize(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x

def powerline(signal):
    wavelet_name = 'db4'  # Wavelet name (you can choose a different one)
    level = 4  # Decomposition level
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = (pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:])
    denoised_ecg_signal = pywt.waverec(coeffs, wavelet_name)
    return denoised_ecg_signal

def all_transform(signal):
    rms = np.zeros([len(signal[:, 0])])

    tempp = np.zeros((12, len(signal[:, 0])))
    for i in range(12):
        tempp[i] = normalize(calc_baseline(powerline(signal[:, i]))) ** 2

    for i in range(len(signal[:, 0])):

        rms[i] = np.sqrt(np.mean(tempp[:, i]))
    rms = normalize(rms)
    return rms