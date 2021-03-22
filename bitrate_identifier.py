"""
A tool for audiophiles to check the bitrate of the files.

Disclaimer: Different genres have different-looking spectrals; hence, this may not be the most accurate thing in the world.
            And... some files may not be supported.
"""

import functools

import numpy
from scipy.fftpack import rfft
from scipy.signal import hann
from soundfile import read


QUALITY_INDEX = {
    22: 'Lossless / FLAC',
    20.5: '320 kbps',
    20: '256 kbps',
    19: '192 kbps',
    16: '128 kbps',
    11: '64 kbps'
} # Source: https://interview.orpheus.network/spectral-analysis.php

def determine_bitrate(cutoff):
    
    for quality, quality_name in QUALITY_INDEX.items():
        if cutoff / 2000 > quality:
            return quality_name
    
    return 'Unidentifiable'

def moving_average(a, w):    
    x = numpy.empty((int(w / 2)))
    x.fill(numpy.nan)
    y = numpy.empty((int(w - len(x))))
    y.fill(numpy.nan)
    return numpy.concatenate((x, numpy.convolve(a, numpy.ones(int(w)) / float(w), 'valid'), y))

def find_cutoff(a, dx, diff, limit):
    
    for i in range(1, int(a.shape[0] - dx)):
        if a[-i] / a[-1] > limit:
            return a.shape[0]
        if a[int(-i - dx)] - a[-i] > diff:
            return a.shape[0] - i - dx
    return a.shape[0]


def determine_quality(filename):
    
    data, frequency  = read(filename)
    seconds = min(int(len(data[:, 0])) / frequency, 30)
    
    cutoff = find_cutoff(moving_average(numpy.lib.scimath.log10((functools.reduce(lambda x, y: x + y, [[0] * frequency, *(abs(rfft(data[t * frequency:(t + 1) * frequency, 0] * hann(frequency))) for t in range(0, seconds - 1))])) / seconds), frequency / 100), frequency / 50, 1.25, 1.1)
        
    return cutoff / 2000, determine_bitrate(cutoff)

if __name__ == '__main__':
    """
    Recursively get the bitrate of all the files in the directory of this program.
    
    Pass an extension as an argument ('*' to check all and any files; any unsupported file including this file containing this program will give errors), the default is FLAC.
    """


    import sys
    from pathlib import Path
    
    extension = ''.join(sys.argv[1:]).lstrip('.') or 'flac'    

    for recursive_item in Path('.').glob('**/*.%s' % extension):
        try:
            print('%s | %s kHz | Quality: %s' % (recursive_item.name, *determine_quality(recursive_item.as_posix())))
        except Exception as e:
            print(f'{recursive_item.name} failed due to an exception: {e}')