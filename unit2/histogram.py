#!/usr/bin/env python3
"Summarize the content of images by showing their statistics and histogram"
import sys, cv2, numpy, pylab, scipy

from scipy.ndimage import extrema


def histogram(im, bins = 64, limits = None):
    """
    Work out the histogram of the image 'im'.
    The histogram is accumulated in 'bins' bins.
    By default, these lie between the minimum and maximum values of 'im'
    but other extrema can be provided in 'limits',
    a list comprising the low and high limits to be used.
    Values outside these extrema are ignored.
    """

    # Find the extreme values in the image.
    if limits is None: limits = [im.min(), im.max()]
    lo, hi = limits

    # Create the arrays to hold the values to be plotted.
    h = numpy.zeros(bins)
    a = numpy.zeros(bins)
    # Fill the x array with the centres of the bins.
    inc = (hi - lo) / (bins - 1)
    for i in range(0, bins):
        a[i] = lo + i * inc

    # Accumulate the histogram.
    ny, nx, nc = im.shape
    for y in range(0, ny):
        for x in range(0, nx):
            for c in range(0, nc):
                v = int((im[y, x, c] - lo) / (hi - lo) * (bins - 1) + 0.5)
                if v >= 0 and v < bins:
                    h[v] += 1.0
    # Return the abscissa and coordinate array
    return a, h


def cumulative_histogram(im, bins=256, limits=None):
    "calculate a cumulative histogram"
    a, h = histogram(im, bins=bins, limits=limits)
    for i in range(1, len(h)):
        h[i] = h[i] + h[i - 1]
    return a, h


def lut (im, table, limits = None):
    "Look up each pixel of an image in a table"
    if limits is None: limits = extrema(im)
    lo, hi = limits
    ny, nx, nc = im.shape  #sizes(im)
    bins = len(table)
    for y in range(0, ny):
        for x in range(0, nx):
            for c in range (0, nc):
                v = (im[y,x,c] - lo)/(hi - lo) * (bins - 1)
                im[y,x,c] = table[int(v)]