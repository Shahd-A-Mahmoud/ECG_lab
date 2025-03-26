import numpy as np
import math
import sys

class Non_local_means():
    def __init__(self, data):
        self.data = data

    def apply_non_local_means(self, noise_var, window_size, patch_size):
        # noise_var --> controls denoising strength
        # window size (how far to look for similar patches) /  5 means look 5 steps left & right
        # if window_size is a single number, make it a range / 3 means compare 3 point segments
        if isinstance(window_size, int):
            window_size = window_size - 1
            Pvec = np.array(range(-window_size, window_size + 1))
        else:
            Pvec = window_size

        data = np.array(self.data)
        n = len(data)
        denoised_signal = np.empty(n)
        denoised_signal[:] = np.nan

        # patch size (how big each comparison segment is)
        # not denoising the edges (because patches would go out of bounds)
        iStart = patch_size + 1
        iEnd = n - patch_size
        # we only denoise from iStart to iEnd
        denoised_signal[iStart:iEnd] = 0

        weights_accumulated = np.zeros(n)
        Npatch = 2 * patch_size + 1
        denoising_strength = 2 * Npatch * noise_var ** 2

        for idx in Pvec:
            k = np.array(range(n))
            kplus = k + idx
            igood = np.where((kplus >= 0) & (kplus < n))

            # compute sum of Squared Differences
            SSD = np.zeros(len(k))
            SSD[igood] = (data[k[igood]] - data[kplus[igood]]) ** 2
            Sdx = np.cumsum(SSD)

            for ii in range(iStart, iEnd):
                distance = Sdx[ii + patch_size] - Sdx[ii - patch_size - 1]
                # if patches are similar, weight is high
                weight = math.exp(-distance / denoising_strength)
                # accumulating the eesult
                t = ii + idx
                if 0 < t < n:
                    denoised_signal[ii] += weight * data[t]
                    weights_accumulated[ii] += weight

        #normalizing and handling edges
        denoised_signal = denoised_signal / (weights_accumulated + sys.float_info.epsilon)
        denoised_signal[0:patch_size + 1] = data[0:patch_size + 1]
        denoised_signal[-patch_size:] = data[-patch_size:]
        return denoised_signal




