#!/usr/bin/env python
"""EVE: the Easy Vision Environment

EVE provides easy-to-use functionality for performing common image
processing and computer vision tasks.  The intention is for them to be
used during interactive sessions, from the Python interpreter's command
prompt or from an enhanced interpreter such as Jupyter as well as in
scripts.  Python v3 is required.

EVE is built principally on top of the popular numpy ('numerical
python') extension to Python.  Images are represented as numpy arrays,
usually 32-bit floating-point ones, indexed by line, pixel and
channel, in that order: image[line,pixel,channel].  The choice of a
floating-point representation is deliberate: it permits images that
have been captured from sensors with more than 8 bits dynamic range to
be processed (e.g., astronomical images and digital radiographs); it
supports Fourier-space processing; and it avoids having to worry about
the problem of rounding integers except at output.  Images in EVE may
contain any number of channels, so EVE can be used with e.g. remote
sensing or hyperspectral imagery.

Other Python extensions are loaded by those routines that need them.  In
particular, PIL (the 'Python Imaging Extension') is used for the input
and output of common image file formats, though not for any processing.
scipy ('scientific python') is used by several routines, and so are a
few other extensions here and there.

On the other hand, EVE is slow.  If you're thinking of using EVE instead
of openCV for real-time video processing, forget it!  This is partly
because of the interpreted nature of Python and partly because EVE
attempts to provide algorithms that are understandable rather than fast:
it is intended as a prototyping environment rather than a real-time
delivery one.  This also makes it useful for teaching how vision
algorithms work, of course.

EVE was written by Adrian F. Clark <alien@essex.ac.uk>, though several
routines are adapted from code written by others; such code is
attributed in the relevant routines.  I have sought permission from
the original authors whenever I could find a way of contacting them.
EVE is made available entirely freely: you are at liberty to use it in
your own work, either as is or after modification.  Many other routines
mimic the functionality of his first image processing library, Adlib,
so much so that some of the routines are direct conversions of his
original Fortran to Python while others retain the Adlib names.

The author would be very happy to hear of improvements or enhancements
that you may make to EVE.
"""
from __future__ import division, print_function
import math, numpy, os, platform, re, struct, sys, tempfile

#-------------------------------------------------------------------------------
# TO DO
# [currently nothing]
#-------------------------------------------------------------------------------
# Symbolic constants and global variables.
#-------------------------------------------------------------------------------
# The operating system we are running under, used to select the appropriate
# external program for display or grabbing images and a few other things.
systype = platform.system ()

use_graphics = None      # the graphics subsystem we are using
sixel_quant = 256        # number of levels for img2sixel

tiny = 1.0e-7            # the smallest number worth bothering about
huge = 1.0e99            # a really large number
max_image_value = 255.0  # the largest value normally put into an image

character_height = 13    # height of characters in draw_text()
character_width = 10     # width of characters in draw_text()
character_bitmap = {
        ' ': [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
        '!': [0x00,0x00,0x18,0x18,0x00,0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x18],
        '"': [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x36,0x36,0x36,0x36],
        '#': [0x00,0x00,0x00,0x66,0x66,0xff,0x66,0x66,0xff,0x66,0x66,0x00,0x00],
        '$': [0x00,0x00,0x18,0x7e,0xff,0x1b,0x1f,0x7e,0xf8,0xd8,0xff,0x7e,0x18],
        '%': [0x00,0x00,0x0e,0x1b,0xdb,0x6e,0x30,0x18,0x0c,0x76,0xdb,0xd8,0x70],
        '&': [0x00,0x00,0x7f,0xc6,0xcf,0xd8,0x70,0x70,0xd8,0xcc,0xcc,0x6c,0x38],
        "'": [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x1c,0x0c,0x0e],
        '(': [0x00,0x00,0x0c,0x18,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x18,0x0c],
        ')': [0x00,0x00,0x30,0x18,0x0c,0x0c,0x0c,0x0c,0x0c,0x0c,0x0c,0x18,0x30],
        '*': [0x00,0x00,0x00,0x00,0x99,0x5a,0x3c,0xff,0x3c,0x5a,0x99,0x00,0x00],
        '+': [0x00,0x00,0x00,0x18,0x18,0x18,0xff,0xff,0x18,0x18,0x18,0x00,0x00],
        ',': [0x00,0x00,0x30,0x18,0x1c,0x1c,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
        '-': [0x00,0x00,0x00,0x00,0x00,0x00,0xff,0xff,0x00,0x00,0x00,0x00,0x00],
        '.': [0x00,0x00,0x00,0x38,0x38,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
        '/': [0x00,0x60,0x60,0x30,0x30,0x18,0x18,0x0c,0x0c,0x06,0x06,0x03,0x03],
        '0': [0x00,0x00,0x3c,0x66,0xc3,0xe3,0xf3,0xdb,0xcf,0xc7,0xc3,0x66,0x3c],
        '1': [0x00,0x00,0x7e,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x78,0x38,0x18],
        '2': [0x00,0x00,0xff,0xc0,0xc0,0x60,0x30,0x18,0x0c,0x06,0x03,0xe7,0x7e],
        '3': [0x00,0x00,0x7e,0xe7,0x03,0x03,0x07,0x7e,0x07,0x03,0x03,0xe7,0x7e],
        '4': [0x00,0x00,0x0c,0x0c,0x0c,0x0c,0x0c,0xff,0xcc,0x6c,0x3c,0x1c,0x0c],
        '5': [0x00,0x00,0x7e,0xe7,0x03,0x03,0x07,0xfe,0xc0,0xc0,0xc0,0xc0,0xff],
        '6': [0x00,0x00,0x7e,0xe7,0xc3,0xc3,0xc7,0xfe,0xc0,0xc0,0xc0,0xe7,0x7e],
        '7': [0x00,0x00,0x30,0x30,0x30,0x30,0x18,0x0c,0x06,0x03,0x03,0x03,0xff],
        '8': [0x00,0x00,0x7e,0xe7,0xc3,0xc3,0xe7,0x7e,0xe7,0xc3,0xc3,0xe7,0x7e],
        '9': [0x00,0x00,0x7e,0xe7,0x03,0x03,0x03,0x7f,0xe7,0xc3,0xc3,0xe7,0x7e],
        ':': [0x00,0x00,0x00,0x38,0x38,0x00,0x00,0x38,0x38,0x00,0x00,0x00,0x00],
        ';': [0x00,0x00,0x30,0x18,0x1c,0x1c,0x00,0x00,0x1c,0x1c,0x00,0x00,0x00],
        '<': [0x00,0x00,0x06,0x0c,0x18,0x30,0x60,0xc0,0x60,0x30,0x18,0x0c,0x06],
        '=': [0x00,0x00,0x00,0x00,0xff,0xff,0x00,0xff,0xff,0x00,0x00,0x00,0x00],
        '>': [0x00,0x00,0x60,0x30,0x18,0x0c,0x06,0x03,0x06,0x0c,0x18,0x30,0x60],
        '?': [0x00,0x00,0x18,0x00,0x00,0x18,0x18,0x0c,0x06,0x03,0xc3,0xc3,0x7e],
        '@': [0x00,0x00,0x3f,0x60,0xcf,0xdb,0xd3,0xdd,0xc3,0x7e,0x00,0x00,0x00],
        'A': [0x00,0x00,0xc3,0xc3,0xc3,0xc3,0xff,0xc3,0xc3,0xc3,0x66,0x3c,0x18],
        'B': [0x00,0x00,0xfe,0xc7,0xc3,0xc3,0xc7,0xfe,0xc7,0xc3,0xc3,0xc7,0xfe],
        'C': [0x00,0x00,0x7e,0xe7,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xe7,0x7e],
        'D': [0x00,0x00,0xfc,0xce,0xc7,0xc3,0xc3,0xc3,0xc3,0xc3,0xc7,0xce,0xfc],
        'E': [0x00,0x00,0xff,0xc0,0xc0,0xc0,0xc0,0xfc,0xc0,0xc0,0xc0,0xc0,0xff],
        'F': [0x00,0x00,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xfc,0xc0,0xc0,0xc0,0xff],
        'G': [0x00,0x00,0x7e,0xe7,0xc3,0xc3,0xcf,0xc0,0xc0,0xc0,0xc0,0xe7,0x7e],
        'H': [0x00,0x00,0xc3,0xc3,0xc3,0xc3,0xc3,0xff,0xc3,0xc3,0xc3,0xc3,0xc3],
        'I': [0x00,0x00,0x7e,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x7e],
        'J': [0x00,0x00,0x7c,0xee,0xc6,0x06,0x06,0x06,0x06,0x06,0x06,0x06,0x06],
        'K': [0x00,0x00,0xc3,0xc6,0xcc,0xd8,0xf0,0xe0,0xf0,0xd8,0xcc,0xc6,0xc3],
        'L': [0x00,0x00,0xff,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0],
        'M': [0x00,0x00,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3,0xdb,0xff,0xff,0xe7,0xc3],
        'N': [0x00,0x00,0xc7,0xc7,0xcf,0xcf,0xdf,0xdb,0xfb,0xf3,0xf3,0xe3,0xe3],
        'O': [0x00,0x00,0x7e,0xe7,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3,0xe7,0x7e],
        'P': [0x00,0x00,0xc0,0xc0,0xc0,0xc0,0xc0,0xfe,0xc7,0xc3,0xc3,0xc7,0xfe],
        'Q': [0x00,0x00,0x3f,0x6e,0xdf,0xdb,0xc3,0xc3,0xc3,0xc3,0xc3,0x66,0x3c],
        'R': [0x00,0x00,0xc3,0xc6,0xcc,0xd8,0xf0,0xfe,0xc7,0xc3,0xc3,0xc7,0xfe],
        'S': [0x00,0x00,0x7e,0xe7,0x03,0x03,0x07,0x7e,0xe0,0xc0,0xc0,0xe7,0x7e],
        'T': [0x00,0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0xff],
        'U': [0x00,0x00,0x7e,0xe7,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3],
        'V': [0x00,0x00,0x18,0x3c,0x3c,0x66,0x66,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3],
        'W': [0x00,0x00,0xc3,0xe7,0xff,0xff,0xdb,0xdb,0xc3,0xc3,0xc3,0xc3,0xc3],
        'X': [0x00,0x00,0xc3,0x66,0x66,0x3c,0x3c,0x18,0x3c,0x3c,0x66,0x66,0xc3],
        'Y': [0x00,0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x3c,0x3c,0x66,0x66,0xc3],
        'Z': [0x00,0x00,0xff,0xc0,0xc0,0x60,0x30,0x7e,0x0c,0x06,0x03,0x03,0xff],
        '[': [0x00,0x00,0x3c,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x3c],
       '\\': [0x00,0x03,0x03,0x06,0x06,0x0c,0x0c,0x18,0x18,0x30,0x30,0x60,0x60],
        ']': [0x00,0x00,0x3c,0x0c,0x0c,0x0c,0x0c,0x0c,0x0c,0x0c,0x0c,0x0c,0x3c],
        '^': [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xc3,0x66,0x3c,0x18],
        '_': [0xff,0xff,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
        '`': [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x38,0x30,0x70],
        'a': [0x00,0x00,0x7f,0xc3,0xc3,0x7f,0x03,0xc3,0x7e,0x00,0x00,0x00,0x00],
        'b': [0x00,0x00,0xfe,0xc3,0xc3,0xc3,0xc3,0xfe,0xc0,0xc0,0xc0,0xc0,0xc0],
        'c': [0x00,0x00,0x7e,0xc3,0xc0,0xc0,0xc0,0xc3,0x7e,0x00,0x00,0x00,0x00],
        'd': [0x00,0x00,0x7f,0xc3,0xc3,0xc3,0xc3,0x7f,0x03,0x03,0x03,0x03,0x03],
        'e': [0x00,0x00,0x7f,0xc0,0xc0,0xfe,0xc3,0xc3,0x7e,0x00,0x00,0x00,0x00],
        'f': [0x00,0x00,0x30,0x30,0x30,0x30,0x30,0xfc,0x30,0x30,0x30,0x33,0x1e],
        'g': [0x7e,0xc3,0x03,0x03,0x7f,0xc3,0xc3,0xc3,0x7e,0x00,0x00,0x00,0x00],
        'h': [0x00,0x00,0xc3,0xc3,0xc3,0xc3,0xc3,0xc3,0xfe,0xc0,0xc0,0xc0,0xc0],
        'i': [0x00,0x00,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00,0x00,0x18,0x00],
        'j': [0x38,0x6c,0x0c,0x0c,0x0c,0x0c,0x0c,0x0c,0x0c,0x00,0x00,0x0c,0x00],
        'k': [0x00,0x00,0xc6,0xcc,0xf8,0xf0,0xd8,0xcc,0xc6,0xc0,0xc0,0xc0,0xc0],
        'l': [0x00,0x00,0x7e,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x78],
        'm': [0x00,0x00,0xdb,0xdb,0xdb,0xdb,0xdb,0xdb,0xfe,0x00,0x00,0x00,0x00],
        'n': [0x00,0x00,0xc6,0xc6,0xc6,0xc6,0xc6,0xc6,0xfc,0x00,0x00,0x00,0x00],
        'o': [0x00,0x00,0x7c,0xc6,0xc6,0xc6,0xc6,0xc6,0x7c,0x00,0x00,0x00,0x00],
        'p': [0xc0,0xc0,0xc0,0xfe,0xc3,0xc3,0xc3,0xc3,0xfe,0x00,0x00,0x00,0x00],
        'q': [0x03,0x03,0x03,0x7f,0xc3,0xc3,0xc3,0xc3,0x7f,0x00,0x00,0x00,0x00],
        'r': [0x00,0x00,0xc0,0xc0,0xc0,0xc0,0xc0,0xe0,0xfe,0x00,0x00,0x00,0x00],
        's': [0x00,0x00,0xfe,0x03,0x03,0x7e,0xc0,0xc0,0x7f,0x00,0x00,0x00,0x00],
        't': [0x00,0x00,0x1c,0x36,0x30,0x30,0x30,0x30,0xfc,0x30,0x30,0x30,0x00],
        'u': [0x00,0x00,0x7e,0xc6,0xc6,0xc6,0xc6,0xc6,0xc6,0x00,0x00,0x00,0x00],
        'v': [0x00,0x00,0x18,0x3c,0x3c,0x66,0x66,0xc3,0xc3,0x00,0x00,0x00,0x00],
        'w': [0x00,0x00,0xc3,0xe7,0xff,0xdb,0xc3,0xc3,0xc3,0x00,0x00,0x00,0x00],
        'x': [0x00,0x00,0xc3,0x66,0x3c,0x18,0x3c,0x66,0xc3,0x00,0x00,0x00,0x00],
        'y': [0xc0,0x60,0x60,0x30,0x18,0x3c,0x66,0x66,0xc3,0x00,0x00,0x00,0x00],
        'z': [0x00,0x00,0xff,0x60,0x30,0x18,0x0c,0x06,0xff,0x00,0x00,0x00,0x00],
        '{': [0x00,0x00,0x0f,0x18,0x18,0x18,0x38,0xf0,0x38,0x18,0x18,0x18,0x0f],
        '|': [0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18],
        '}': [0x00,0x00,0xf0,0x18,0x18,0x18,0x1c,0x0f,0x1c,0x18,0x18,0x18,0xf0],
        '~': [0x00,0x00,0x00,0x00,0x00,0x00,0x06,0x8f,0xf1,0x60,0x00,0x00,0x00]
        }

#-------------------------------------------------------------------------------
# External programs that we invoke in routines.
#-------------------------------------------------------------------------------
# Some of EVE's functionality is provided by external programs, invoked wihin
# routines.  For each function that is performed, the following table lists
# both the program name (used to find the executable) and a template for
# invoking the program.

# Within each template:
#    %o_pgm represents a PGM-format input to the program produced by EVE
#    %i_pgm represents a PGM-format image that needs to be read in
# and likewise for:
#    %i_ppm and %o_ppm
#    %i_png and %o_png
#    %i_txt and %o_txt

# Note that NOT ALL ROUTINES currently use this mechanism, especially for image
# display and graph-plotting.

PROGRAMS = {
    "display": [
        ["xv", "xv %o_pgm"],
        ["display", "display %o_pgm"],
        ["/Applicaions/Preview.app", "open -a /Applications/Preview.app %o_png"]
    ],

    "grab image": [
        ["isightcapture", "isightcapture -t png %i_png"],
        ["imagesnap", "imagesnap -q -t png %i_png"],
        ["streamer", "streamer -q -f ppm -o %i_ppm"]
    ],

    "grab screen": [
        ["screencapture", "screencapture -x -m %i_png"],
        ["shutter", "shutter -f  -e -o %i_png"],
        ["scrot", "scrot %i_png"]
    ],

    "sift": [
        ["sift", "sift %o_ppm -o %q_txt"]
    ],

    # img2sixel is part of the "libsixel-bin" package in the Ubuntu world.
    # On a Mac, "brew install libsixel" provides img2sixel.
    "sixel display": [
        ["img2sixel", "img2sixel -p %d 2>/dev/null"] # a special case
    ],

    "susan": [
        ["susan", "susan %o_pgm %i_pgm -c"]
    ],

    "graph": [
        ["gnuplot", "gnuplot"]
    ],
}


#-------------------------------------------------------------------------------
def add_gaussian_noise (im, mean=0.0, sd=1.0, seed=None):
    """
    Add Gaussian-distributed noise to each pixel of image `im`.

    Arguments:
      im  the image to which noise will be added (modified)
    mean  the mean of the Gaussian-distributed noise (default: 0.0)
      sd  the standard deviation of the noise (default: 1.0)
    seed  if supplied, the value used to seed the random number generator
    """
    if not seed is None: numpy.random.seed (seed)
    im += numpy.random.normal (mean, sd, im.shape)

#-------------------------------------------------------------------------------
def annular_mean (im, y0=None, x0=None, rlo=0.0, rhi=None, alo=-math.pi,
                  ahi=math.pi):
    """
    Return the mean of an annular region of image `im`.

    Arguments:
     im  the image to be examined
     y0  the y-value of the centre of the annulus (default: centre pixel)
     x0  the x-value of the centre of the annulus (default: centre pixel)
    rlo  the inner radius of the annular region (default: the middle)
    rhi  the outer radius of the annular region (default: largest circle)
    alo  the lower angle of the annular region (default: -pi)
    ahi  the higher angle of the annular region (default: pi)
    """
    # Fill in the default values as necessary.
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    if y0 is None: y0 = ny / 2.0
    if x0 is None: x0 = nx / 2.0
    if rhi is None: rhi = math.sqrt ((nx - x0)**2 + (ny - y0)**2)
    ave = num = 0.0

    # Cycle through the image.
    for y in range (0, ny):
        yy = (y - y0)**2
        for x in range (0, nx):
            r = math.sqrt (yy + (x-x0)**2)
            if r <= 0.0: angle = 0.0
            else: angle = -math.atan2 (y-y0, x-x0)
            for c in range (0, nc):
                if angle >= alo and angle <= ahi and r >= rlo and r <= rhi:
                    ave += im[y,x,c]
                    num += 1

    if num > 0: ave /= num
    return ave

#-------------------------------------------------------------------------------
def annular_profile (im, y0=None, x0=None, rlo=0.0, rhi=None, alo=-math.pi,
                     ahi=math.pi):
    """
    Return an array of the rotational means at one-pixel radial spacings
    in an annular region of image `im`.

    Arguments:
     im  the image to be examined
     y0  the y-value of the centre of the rotation (default: centre pixel)
     x0  the x-value of the centre of the rotation (default: centre pixel)
    rlo  the inner radius of the annular region
    rhi  the outer radius of the annular region
    alo  the lower angle of the annular region (default: -pi)
    ahi  the higher angle of the annular region (default: pi)
    """
    # Fill in the default values as necessary.
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    if y0 is None: y0 = ny / 2.0
    if x0 is None: x0 = nx / 2.0
    if rhi is None: rhi = math.sqrt ((nx - x0)**2 + (ny - y0)**2)
    n = int (rhi + 1.0)
    ave = numpy.zeros ((n))
    num = numpy.zeros ((n))
    # Cycle through the image.
    for y in range (0, ny):
        yy = (y - y0)**2
        for x in range (0, nx):
            r = math.sqrt (yy + (x-x0)**2)
            if r <= 0.0: angle = 0.0
            else: angle = -math.atan2 (y-y0, x-x0)
            for c in range (0, nc):
                if angle >= alo and angle <= ahi and r >= rlo and r <= rhi:
                    i = int (r - rlo)
                    ave[i] += im[y,x,c]
                    num[i] += 1
    # Convert the sums into means.
    for i in range (0, n):
        if num[i] > 0: ave[i] /= num[i]
    return ave

#-------------------------------------------------------------------------------
def annular_set (im, v, y0=None, x0=None, rlo=0.0, rhi=None, alo=-math.pi,
                 ahi=math.pi):
    """
    Set an annular region of image `im` to value `v`.

    Arguments:
     im  the image to be set (modified)
      v  value to which the region is to be set
     y0  the y-value of the centre of the annulus (default: centre pixel)
     x0  the x-value of the centre of the annulus (default: centre pixel)
    rlo  the inner radius of the annular region
    rhi  the outer radius of the annular region
    alo  the lower angle of the annular region (default: -pi)
    ahi  the higher angle of the annular region (default: pi)
    """
    # Fill in the default values as necessary.
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    if y0 is None: y0 = ny / 2.0
    if x0 is None: x0 = nx / 2.0
    if rhi is None: rhi = math.sqrt ((nx - x0)**2 + (ny - y0)**2)

    # Cycle through the image.
    for y in range (0, ny):
        yy = (y - y0)**2
        for x in range (0, nx):
            r = math.sqrt (yy + (x-x0)**2)
            if r <= 0.0: angle = 0.0
            else: angle = -math.atan2 (y-y0, x-x0)
            if angle >= alo and angle <= ahi and r >= rlo and r <= rhi:
                im[y,x] = v

#-------------------------------------------------------------------------------
def binarize (im, threshold, below=False, bg=0.0, fg=max_image_value):
    """
    Binarize an image, returning the result.

    Arguments:
           im  image to be binarized
    threshold  the threshold to be used for binarization
        below  values below, rather than above, the threshold are foreground
           bg  value to which pixels at or below the threshold will be set
               (default: 0.0)
           fg  value to which pixels equal to or above the threshold will
               be set (default: max_image_value)
    """
    bim = image (im)
    set (bim, bg)
    if below:
        bim[numpy.where (im < threshold)] = fg
    else:
        bim[numpy.where (im >= threshold)] = fg
    return bim

#-------------------------------------------------------------------------------
def blend_pixel (im, y, x, v, opac):
    """
    Blend the value `v` into the pixel `im[y,x]` according to the
    opacity `opac`.

    Arguments:
      im  image in which the pixel is drawn (modified)
       y  y-position of the pixel to be modified
       x  x-position of the pixel to be modified
       v  new value to which the pixel is to be set
    opac  opacity with which the value will be drawn into the pixel
    """
    ny, nx, nc = sizes (im)
    if y >= 0 and y < ny and x >= 0 and x < nx:
        if not isinstance (v, list): v = [v] * nc
        for c in range (0, nc):
            im[y,x,c] = v[c] * opac + (1.0 - opac) * im[y,x,c]

#-------------------------------------------------------------------------------
def bounding_box (im):
    """
    Return the bounding box of a feature in binary image `im` as
    `[ystart, xstart, ystop, xstop]`.

    Arguments:
    im  image for which the bounding box is to be returned
    """
    B = numpy.argwhere (im)
    (ystart, xstart, junk), (ystop, xstop, junk) = B.min(0), B.max(0) + 1
    return [ystart, xstart, ystop, xstop]

#-------------------------------------------------------------------------------
def canny (im, lo, hi):
    """
    Perform edge detection in `im` using the Canny operator, returning the
    result.

    Arguments:
    im  image in which the edges are to be found
    lo  threshold below which edge segments are discarded
    hi  threshold above which edge segments are definitely edges

    This routine is adapted from code originally written by
    Zachary Pincus <zachary.pincus@yale.edu>.
    """
    import scipy
    import scipy.ndimage as ndimage

    # Convert the EVE-format image into one compatible with scipy.
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    if nc == 1: sci_im = im[:,:,0]
    else:       sci_im = mono(im)[:,:,0]

    # The following filter kernels are for calculating the value of
    # neighbours in the required directions.
    _N  = scipy.array ([[0, 1, 0],
                        [0, 0, 0],
                        [0, 1, 0]], dtype=bool)
    _NE = scipy.array ([[0, 0, 1],
                        [0, 0, 0],
                        [1, 0, 0]], dtype=bool)
    _W  = scipy.array ([[0, 0, 0],
                        [1, 0, 1],
                        [0, 0, 0]], dtype=bool)
    _NW = scipy.array ([[1, 0, 0],
                        [0, 0, 0],
                        [0, 0, 1]], dtype=bool)

    # After quantizing the orientations of gradients, vertical
    # (north-south) edges get values of 3, northwest-southeast edges get
    # values of 2, and so on, as below.
    _NE_d = 0
    _W_d = 1
    _NW_d = 2
    _N_d = 3
    grad_x = ndimage.sobel (sci_im, 0)
    grad_y = ndimage.sobel (sci_im, 1)
    grad_mag = scipy.sqrt (grad_x**2 + grad_y**2)
    grad_angle = scipy.arctan2 (grad_y, grad_x)

    # Scale the angles in the range [0,3] and then round to quantize.
    quantized_angle = scipy.around (3 * (grad_angle + numpy.pi) \
                                        / (numpy.pi * 2))

    # Perform non-maximal suppression.  An edge pixel is only good if
    # its magnitude is greater than its neighbours normal to the edge
    # direction.  We quantize the edge direction into four angles, so we
    # only need to look at four sets of neighbours.
    NE = ndimage.maximum_filter (grad_mag, footprint=_NE)
    W  = ndimage.maximum_filter (grad_mag, footprint=_W)
    NW = ndimage.maximum_filter (grad_mag, footprint=_NW)
    N  = ndimage.maximum_filter (grad_mag, footprint=_N)
    thinned = (((grad_mag > W)  & (quantized_angle == _N_d )) |
               ((grad_mag > N)  & (quantized_angle == _W_d )) |
               ((grad_mag > NW) & (quantized_angle == _NE_d)) |
               ((grad_mag > NE) & (quantized_angle == _NW_d)) )
    thinned_grad = thinned * grad_mag

    # Perform hysteresis thresholding: find seeds above the high
    # threshold, then expand out until the line segment goes below the
    # low threshold.
    high = thinned_grad > hi
    low = thinned_grad > lo
    canny_edges = ndimage.binary_dilation (high, structure=scipy.ones((3,3)),
                                           iterations=-1, mask=low)

    # Convert the results back to an EVE-format image and return it.
    ce = image ((ny,nx,1))
    ce[:,:,0] = canny_edges[:,:] * max_image_value
    return ce

#-------------------------------------------------------------------------------
def centroid (im, c=0):
    """
    Return the centroid (centre of mass) of a channel of an image.

    This routine is normally used on a binarized image (see binarize())
    after labelling (see label_regions() and labelled_region()) to
    locate the centres of regions.

    Arguments:
    im  image for which the centroid is to be found
     c  channel to be examined (default: 0)
    """
    m00 = m01 = m10 = 0.0
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    for y in range (0, ny):
        for x in range (0, nx):
            m00 += im[y,x,c]
            m10 += im[y,x,c] * y
            m01 += im[y,x,c] * x
    if m00 < tiny:
        y = ny / 2.0
        x = nx / 2.0
    else:
        y = m10 / m00
        x = m01 / m00
    return [y, x]

#-------------------------------------------------------------------------------
def clip (im, lo, hi):
    """
    Ensure all pixels in an image are in the range lo to hi.

    Arguments:
    im  the image to be clipped (modified)
    lo  the lowest value to be in the image after clipping
    hi  the highest value to be in the image after clipping
    """
    numpy.clip (im, lo, hi, out=im)

#-------------------------------------------------------------------------------
def compare (im1, im2, tol=tiny, report=20, indent='  ', fd=sys.stdout):
    """
    Compare two images, reporting up to report pixels that differ.
    The routine returns the number of differences found.

    Arguments:
       im1  image to be compared with im2
       im2  image to be compared with im1
       tol  minimum amount by which pixels must differ (default: eve.tiny)
    report  the maximum number of differences reported (default: 20)
            (the presence of further differences is indicated by '...')
    indent  indentation output before a difference (default: '  ')
        fd  file on which the output is to be written (default: sys.stdout)
    """
    im1 = reshape3 (im1)
    ny, nx, nc = sizes (im1)
    ndiffs = 0
    diffs = []
    for y in range (0, ny):
        for x in range (0, nx):
            for c in range (0, nc):
                if abs (im1[y,x,c] - im2[y,x,c]) > tol:
                    ndiffs += 1
                    if ndiffs <= report:
                        diffs.append ([y,x,c])
    if ndiffs > 0 and report > 0:
        if ndiffs == 1:
            print (ndiffs, 'difference found:', file=fd)
        else:
            print (ndiffs, 'differences found:', file=fd)
        for d in range (0, len(diffs)):
            y,x,c = diffs[d]
            print (indent, y,x,c, '->', im1[y,x,c], '&', im2[y,x,c], file=fd)
        if ndiffs > report:
            print (indent, '...', file=fd)
    return ndiffs

#-------------------------------------------------------------------------------
def compare_lists (l1, l2, tol):
    """Compare a pair of lists, element by element, returning the number
    of differences found."""
    n1 = len (l1)
    n2 = len (l2)
    if n1 == n2:
        ndiff = 0
        for i in range (0, n1):
            if abs (l1[i] - l2[i]) > tol:
                ndiff += 1
    else:
        ndiff = abs (n1 - n2)
    return ndiff

#-------------------------------------------------------------------------------
def contrast_stretch (im, low=0.0, high=max_image_value):
    """
    Stretch the contrast in the image to the supplied low and high values.

    Arguments:
      im  image whose contrast is to be stretched (modified)
     low  new value to which the lowest value in im is to be scaled
          (default: 0.0)
    high  new value to which the highest value in im is to be scaled
          (default: max_image_value)
    """
    oldmin, oldmax = extrema (im)
    fac = (high - low) / (oldmax - oldmin)
    # For some reason, the following line doesn't work but the subsequent
    # three do!
    # im = (im - oldmin) * fac + low
    im -= oldmin
    im *= fac
    im += low

#-------------------------------------------------------------------------------
def convolve (im, mask, statistic='sum'):
    """
    Perform a convolution of im with mask, returning the result.

    Arguments:
           im  the image to be convolved with mask (modified)
         mask  the convolution mask to be used
    statistic  one of:
                  sum  conventional convolution
                 mean  conventional convolution
               median  median filtering
                  min  grey-scale shrink (reduces light areas)
                  max  grey-scale expand (enlarges light areas)
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    mask = reshape3 (mask)
    my, mx, mc = sizes (mask)
    yo = my // 2
    xo = mx // 2

    # Create an output image of the same size as the input.
    result = image (im)

    # We need a special case for 'min' statistic to erase the mask elements
    # that are zero.
    nzeros = len ([x for x in mask.ravel() if x == 0])

    # Loop over the pixels in the image.  For each pixel position, multiply
    # the region around it with the mask, summing the elements and storing
    # that in the equivalent pixel of the output image.
    v = numpy.zeros ((my*mx*mc))
    vi = 0
    for yi in range (0, ny):
        for xi in range (0, nx):
            for ym in range (0, my):
                yy = (ym + yi - yo) % ny
                for xm in range (0, mx):
                    xx = (xm + xi - xo) % nx
                    v[vi] = im[yy,xx,0] * mask[ym,xm,0]
                    vi += 1
            if statistic == 'sum':
                ave = numpy.sum (v)
            elif statistic == 'mean':
                ave = numpy.mean (v)
            elif statistic == 'max':
                ave = numpy.max (v)
            elif statistic == 'min':
                v = sorted (v)
                ave = numpy.min (v[nzeros:])
            elif statistic == 'median':
                ave = numpy.median (v)
            result[yi,xi,0] = ave
            vi = 0
    return result

#-------------------------------------------------------------------------------
def copy (im):
    """
    Copy the pixels from image 'im' into a new image, which is returned.

    Arguments:
    im  the image to be copied
    """
    return im.copy ()

#-------------------------------------------------------------------------------
def correlate (im1, im2):
    """
    Return the unnormalized Fourier correlation surface between two
    images.

    Arguments:
    im1  image to be correlated with im2
    im2  image to be correlated with im1
    """
    # Work out how big the images need to be to avoid aliasing.
    ny1, nx1, nc1 = sizes (im1)
    ny2, nx2, nc2 = sizes (im2)
    if nc1 != nc2:
        raise ValueError ('Images need to have the same number of channels.')
    ny = max (ny1, ny2)
    nx = max (nx1, nx2)
    
    # Transform the two images.
    temp1 = fourier (im1)
    temp2 = fourier (im2)

    # Create a pair of double-sized images and insert the transforms into them
    # so that their zero frequencies are in the middle.
    ft1 = image ((ny*2, nx*2, nc1), type=temp1.dtype)
    ft2 = image ((ny*2, nx*2, nc1), type=temp2.dtype)
    insert (ft1, temp1, ny, nx)
    insert (ft2, temp2, ny, nx)

    # Multiply one with the conjugate of the other and Fourier transform back.
    ft1 *= numpy.conjugate (ft2)
    res = fourier (ft1, forward=False)

    # Return the result.
    return res

#-------------------------------------------------------------------------------
def correlation_coefficient (im1, im2):
    """
    Return the correlation coefficient between two images.

    Arguments:
    im1  image to be correlated with im2
    im2  image to be correlated with im1
    """
    im1 = reshape3 (im1)
    ny, nx, nc = sizes (im1)
    im2 = reshape3 (im2)
    sumx = sumy = sumxx = sumyy = sumxy = 0.0
    for y in range (0, ny):
        for x in range (0, nx):
            for c in range (0, nc):
                v1 = im1[y,x,c]
                v2 = im2[y,x,c]
                sumx += v1
                sumy += v2
                sumxx += v1 * v1
                sumxy += v1 * v2
                sumyy += v2 * v2
    n = ny * nx * nc
    v1 = sumxy - sumx * sumy / n
    v2 = math.sqrt((sumxx-sumx*sumx/n) * (sumyy-sumy*sumy/n))
    return v1 / v2

#-------------------------------------------------------------------------------
def count_pixels (im, v):
    """
    Return the number of pixels having the value 'v'.

    Arguments:
    im  image to be searched
     v  value to be looked for
    """

    # This is equivalent to:
    #    im = reshape3 (im)
    #    ny, nx, nc = sizes (im)
    #    count = 0
    #    for y in range (0, ny):
    #        for x in range (0, nx):
    #            for c in range (0, nc):
    #                if im[y,x,c] == v: count += 1

    count = len (numpy.where (im == v)[0])
    return count    

#-------------------------------------------------------------------------------
def covariance_matrix (im):
    """
    Return the covariance matrix and means of the channels of im.

    Arguments:
    im  image for which the covariance matrix is to be calculated
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    covmat = numpy.ndarray ((nc, nc))
    ave = numpy.ndarray ((nc))
    for c in range (0, nc):
        ch = get_channel (im, c)
        ave[c] = mean (ch)
    for c1 in range (0, nc):
        ch1 = get_channel (im, c1)
        for c2 in range (0, c1+1):
            ch2 = get_channel (im, c2)
            covmat[c1,c2] = ((ch1 - ave[c1]) * (ch2 - ave[c2])).mean()
            covmat[c2,c1] = covmat[c1,c2]
    return covmat, ave

#-------------------------------------------------------------------------------
def cumulative_histogram (im, bins=64, limits=None, disp=False):
    """
    Find the cumulative histogram of an image.

    Arguments:
        im  image for which the cumulative histogram is to be found
      bins  number of bins in the histogram (default: 64)
    limits  extrema between which the histogram is to be found
      disp  when True, the histogram will be drawn
    """
    a, h = histogram (im, bins=bins, limits=limits, disp=False)
    h = h.cumsum ()
    if disp: graph (a, h, 'Cumulative histogram', 'bin', 'number of pixels',
                    style='histogram')
    return a, h

#-------------------------------------------------------------------------------
def closing (im, mask):
    """
    Perform a morphological closing on `im` with `mask`, returning the result.

    Arguments:
      im  image to be processed
    mask  mask image
    """
    expanded = expand (im, mask)
    return shrink (expanded, mask)

#-------------------------------------------------------------------------------
def describe (im):
    """
    Return a set of descriptors for each region of a labelled image.

    Arguments:
       im  labelled image whose regions are to be described
    """
    im = reshape3 (im)

    # Compute the areas of all the regions.
    unique, counts = numpy.unique (im, return_counts=True)
    info = {}
    nr = len (unique)
    # Now look at each region in turn.
    for i in range (1, nr):
        # Store away the area.
        info[i] = {"area": counts[i]}
        # The perimeter.
        reg = labelled_region (im, i)
        perimeter (reg)
        info[i]["perimeter"] = count_pixels (reg, 255)
        # The axis-aligned bounding box.
        info[i]["BouBox"] = bounding_box (reg)
        # The oriented bounding box.
        info[i]["OriBouBox"] = oriented_bounding_box (reg)

    return info
    
#-------------------------------------------------------------------------------
def draw_border (im, v=max_image_value, width=2):
    """
    Draw a border around an image.

    Arguments:
       im  image to which the border is to be added (modified)
        v  value to which the border is to be set (default: max_image_value)
    width  width of the border in pixels (default: 2)
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    im[0:width,:,:] = v       # top
    im[ny-width-1:ny,:,:] = v # bottom
    im[:,0:width,:] = v       # left
    im[:,nx-width-1:nx,:] = v # right

#-------------------------------------------------------------------------------
def draw_box (im, ylo, xlo, yhi, xhi, border=max_image_value, fill=None):
    """
    Draw a rectangular box, optionally filled.

    Arguments:
        im  image in which the box is to be drawn (modified)
       ylo  y-value of the lower left corner of the box
       xlo  x-value of the lower left corner of the box
       yhi  y-value of the upper right corner of the box
       xhi  x-value of the upper right corner of the box
    border  value used for drawing the box (default: max_image_value)
      fill  value with which the inside of the box is to be filled
            (default: None, meaning it is unfilled)
    """
    im = reshape3 (im)
    draw_line_fast (im, ylo, xlo, ylo, xhi, border)
    draw_line_fast (im, ylo, xhi, yhi, xhi, border)
    draw_line_fast (im, yhi, xhi, yhi, xlo, border)
    draw_line_fast (im, yhi, xlo, ylo, xlo, border)
    if not fill is None:
        set_region (im, ylo+1, xlo+1, yhi, xhi, fill)

#-------------------------------------------------------------------------------
def draw_circle (im, yc, xc, r, v, fast=True, fill=None, threshold=20):
    """
    Draw a circle with value v of radius r centred on (xc, yc) in image im.

    Arguments:
           im  image upon which the circle is to be drawn (modified)
           yc  y-value (row) of the centre of the circle
           xc  x-value (column) of the centre of the circle
            r  radius of he circle in pixels
            v  value to which pixels forming the circle are set
         fast  draw a circle using Bresenham's algorithm, which aliases,
               or Wu's algorithm, which doesn't (default: True)
         fill  value with which the inside of the circle is to be filled
               (default: None, meaning it is unfilled)
    threshold  value used to distinguish pixels on the circle from
               surrounding ones, passed on into fill_outline (default: 20)
    """
    if fast:
        draw_circle_fast (im, yc, xc, r, v, fill=None)
    else:
        draw_circle_aa (im, yc, xc, r, v, fill=fill, threshold=threshold)

#-------------------------------------------------------------------------------
def draw_circle_aa (im, yc, xc, r, v, fill=None, threshold=20):
    """
    Draw an anti-aliased circle with value v of radius r centred on
    (xc, yc) in image im.

    Arguments:
           im  image upon which the circle is to be drawn (modified)
           yc  y-value (row) of the centre of the circle
           xc  x-value (column) of the centre of the circle
            r  radius of he circle in pixels
            v  value to which pixels forming the circle are set
         fill  value with which the inside of the circle is to be filled
               (default: None, meaning it is unfilled)
    threshold  value used to distinguish pixels on the circle from
               surrounding ones, passed on into fill_outline (default: 20)

    The circle is anti-aliased, using an algorithm due to Xiaolin Wu
    (published in "Graphics Gems II").  Although fairly fast, it is slower
    than Bresenham's algorithm, used in the routine draw_circle_fast, and
    paints a range of values into the image; if it is important that all
    pixels of the line have the value v, use draw_circle_fast.
    The implementation was adapted from the PHP code in
    http://mapidev.blogspot.com/2009/03/xiaolin-wu-circle-php-implementation.html
    """
    im = reshape3 (im)
    x = xx = r
    y = yy = -1
    t = 0
    while x > y:
        y += 1
        d = math.sqrt (r**2 - y**2)
        opac = int (d + 0.5) - d
        if opac < t: x -= 1
        trans = 1.0 - opac
        im[yc+y, xc+x] = v
        blend_pixel (im, y + yc, x + xc - 1, v, trans)
        blend_pixel (im, y + yc, x + xc + 1, v, opac)
        im[yc+x, xc+y] = v
        blend_pixel (im, x + yc - 1, y + xc, v, trans)
        blend_pixel (im, x + yc + 1, y + xc, v, opac)
        im[yc+y, xc-x] = v
        blend_pixel (im, y + yc, xc - x + 1, v, trans)
        blend_pixel (im, y + yc, xc - x - 1, v, opac)
        im[yc+x, xc-y] = v
        blend_pixel (im, x + yc - 1, xc - y, v, trans)
        blend_pixel (im, x + yc + 1, xc - y, v, opac)
        im[yc-y, xc+x] = v
        blend_pixel (im, yc - y, x + xc - 1, v, trans)
        blend_pixel (im, yc - y, x + xc + 1, v, opac)
        im[yc-x, xc+y] = v
        blend_pixel (im, yc - x - 1, y + xc, v, opac)
        blend_pixel (im, yc - x + 1, y + xc, v, trans)
        im[yc-x, xc-y] = v
        blend_pixel (im, yc - x - 1, xc - y, v, opac)
        blend_pixel (im, yc - x + 1, xc - y, v, trans)
        im[yc-y, xc-x] = v
        blend_pixel (im, yc - y, xc - x - 1, v, opac)
        blend_pixel (im, yc - y, xc - x + 1, v, trans)
        t = opac
    if not fill is None: fill_outline (im, yc, yc, v)

#-------------------------------------------------------------------------------
def draw_circle_fast (im, yc, xc, r, v, fill=None):
    """
    Draw a circle with value v of radius r centred on (xc, yc) in image im.

    Arguments:
           im  image upon which the circle is to be drawn (modified)
           yc  y-value (row) of the centre of the circle
           xc  x-value (column) of the centre of the circle
            r  radius of he circle in pixels
            v  value to which pixels forming the circle are set
         fill  value with which the inside of the circle is to be filled
               (default: None, and so is unfilled)
    threshold  value used to distinguish pixels on the circle from
               surrounding ones, passed on into fill_outline (default: 20)
    """
    im = reshape3 (im)
    x = 0
    y = r
    p = 3 - 2 * r
    ny, nx, nc = sizes (im)
    while x < y:
        im[yc+y,xc+x] = v
        im[yc+y,xc-x] = v
        im[yc-y,xc+x] = v
        im[yc-y,xc-x] = v
        im[yc+x,xc+y] = v
        im[yc+x,xc-y] = v
        im[yc-x,xc+y] = v
        im[yc-x,xc-y] = v
        if p < 0:
            p += 4 * x + 6
        else:
            p += 4 * (x - y) + 6
            y -= 1
        x += 1
    if x == y:
        im[yc+y,xc+x] = v
        im[yc+y,xc-x] = v
        im[yc-y,xc+x] = v
        im[yc-y,xc-x] = v
        im[yc+x,xc+y] = v
        im[yc+x,xc-y] = v
        im[yc-x,xc+y] = v
        im[yc-x,xc-y] = v
    if not fill is None: fill_outline (im, yc, xc, v)

#-------------------------------------------------------------------------------
def draw_line (im, y0, x0, y1, x1, v, fast=True):
    """
    Draw a line from (x0, y0) to (x1, y1) with value v in image im.

    Arguments:
      im  image upon which the line is to be drawn (modified)
      y0  y-value (row) of the start of the line
      x0  x-value (column) of the start of the line
      y1  y-value (row) of the end of the line
      x1  x-value (column) of the end of the line
       v  value to which pixels on the line are to be set
    fast  draw the line using Bresenham's algorithm, which aliases,
          or Wu's algorithm, which doesn't (default: True)
    """
    if fast:
        draw_line_fast (im, y0, x0, y1, x1, v)
    else:
        draw_line_aa (im, y0, x0, y1, x1, v)

#-------------------------------------------------------------------------------
def draw_line_aa (im, y0, x0, y1, x1, v):
    """
    Draw an anti-aliased line from (x0, y0) to (x1, y1) with value v
    in image im.

    Arguments:
    im  image upon which the line is to be drawn (modified)
    y0  y-value (row) of the start of the line
    x0  x-value (column) of the start of the line
    y1  y-value (row) of the end of the line
    x1  x-value (column) of the end of the line
     v  value to which pixels on the line are to be set

    The line is anti-aliased, using an algorithm due to Xiaolin Wu ("An
    Efficient Antialiasing Technique", Computer Graphics July 1991); the
    code is a corrected version of that in the relevant Wikipedia entry.
    The algorithm draws pairs of pixels straddling the line, coloured
    according to proximity; pixels at the line ends are handled
    separately.  Although fairly fast, it is slower than Bresenham's
    algorithm and paints a range of values into the image; if it is
    important that all pixels of the line have the value v, there is a
    separate routine, draw_line_fast, which implements Bresnham's
    algorithm.
    """
    im = reshape3 (im)
    if abs(y1 - y0) > abs(x1 - x0): steep = True
    else:                           steep = False
    if steep:
        y0, x0 = x0, y0
        y1, x1 = x1, y1
    if x0 > x1:
        x1, x0 = x0, x1
        y1, y0 = y0, y1
    dx = x1 - x0 + 0.0
    dy = y1 - y0
    if dx == 0.0: de = 1.0e30
    else:         de = dy / dx

    # Handle the first end-point.
    xend = int (x0 + 0.5)
    yend = y0 + de * (xend - x0)
    xgap = 1.0 - (x0 + 0.5 - int(x0 + 0.5))
    xpxl1 = int (xend)  # this will be used in the main loop
    ypxl1 = int (yend)
    if steep:
        blend_pixel (im, xpxl1, ypxl1,   v, 1.0 - (yend - int(yend)))
        blend_pixel (im, xpxl1, ypxl1+1, v, yend - int(yend))
    else:
        blend_pixel (im, ypxl1,   xpxl1, v, 1.0 - (yend - int(yend)))
        blend_pixel (im, ypxl1+1, xpxl1, v, yend - int(yend))
    intery = yend + de  # first y-intersection for the main loop

    # Handle the second end-point.
    xend = int (x1 + 0.5)
    yend = y1 + de * (xend - x1)
    xgap = x1 + 0.5 - int(x1 + 0.5)
    xpxl2 = int (xend)  # this will be used in the main loop
    ypxl2 = int (yend)
    if steep:
        blend_pixel (im, xpxl2, ypxl2,   v, 1.0 - (yend - int (yend)))
        blend_pixel (im, xpxl2, ypxl2+1, v, yend - int(yend))
    else:
        blend_pixel (im, ypxl2,   xpxl2, v, 1.0 - (yend - int (yend)))
        blend_pixel (im, ypxl2+1, xpxl2, v, yend - int(yend))

    # The main loop.
    for x in range (xpxl1+1, xpxl2):
        if steep:
            blend_pixel (im, x, int (intery),   v,
                         math.sqrt(1.0 - (intery - int(intery))))
            blend_pixel (im, x, int (intery)+1, v,
                         math.sqrt (intery - int(intery)))
        else:
            blend_pixel (im, int (intery),   x, v,
                         math.sqrt(1.0 - (intery - int(intery))))
            blend_pixel (im, int (intery)+1, x, v,
                         math.sqrt(intery - int(intery)))
        intery += de

#-------------------------------------------------------------------------------
def draw_line_fast (im, y0, x0, y1, x1, v):
    """Draw a line from (x0, y0) to (x1, y1) with value v in image im.

    Arguments:
    im  image upon which the line is to be drawn (modified)
    y0  y-value (row) of the start of the line
    x0  x-value (column) of the start of the line
    y1  y-value (row) of the end of the line
    x1  x-value (column) of the end of the line
     v  value to which pixels on the line are to be set

    This routine uses the classic line-drawing due to Bresenham, which
    aliases badly for most lines; if appearance is more important than
    speed, there is a separate EVE routine that implements anti-aliased
    line-drawing using an algorithm due to Xiaolin Wu.

    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    y0 = int (y0)
    x0 = int (x0)
    y1 = int (y1)
    x1 = int (x1)
    if abs(y1 - y0) > abs(x1 - x0): steep = True
    else:                           steep = False
    if steep:
        y0, x0 = x0, y0
        y1, x1 = x1, y1
    if x0 > x1:
        x1, x0 = x0, x1
        y1, y0 = y0, y1
    dx = x1 - x0 + 0.0
    dy = abs(y1 - y0)
    e = 0.0
    if dx == 0.0: de = 1.0e30
    else:         de = dy / dx
    y = y0
    if y0 < y1: ystep =  1
    else:       ystep = -1
    for x in range (x0,x1+1):
        if steep:
            if x >= 0 and x < ny and y >= 0 and y < nx: im[x,y,:] = v
        else:
            if x >= 0 and x < nx and y >= 0 and y < ny: im[y,x,:] = v
        e += de
        if e >= 0.5:
            y += ystep
            e -= 1.0

#-------------------------------------------------------------------------------
def draw_oriented_box (im, corners, v=255):
    """
    """
    nc = len (corners)
    for c in range (1, nc):
        draw_line_fast (im, corners[c-1][0], corners[c-1][1],
                        corners[c][0], corners[c][1], v=v)
    
    draw_line_fast (im, corners[nc-1][0], corners[nc-1][1],
                    corners[0][0], corners[0][1], v=v)

#-------------------------------------------------------------------------------
def draw_polygon (im, yc, xc, r, nsides, v=max_image_value, fast=False,
                  rotate=0, fill=None, threshold=50):
    """
    Draw an nsides-sided polygon of radius r centred at (yc, xc),
    returning a list of its vertices.

    Arguments:
           im  image upon which the text is to be written (modified)
           yc  y-value (row) of the centre of the polygon
           xc  x-value (column) of the centre of the polygon
       radius  radius of the circle in which the polygon is enclosed
       nsides  number of sides that the polygon is to have
            v  value to which pixels are to be set
               (default: max_image_value)
         fast  if True, don't use anti-aliased lines (default: False)
       rotate  angle in radians through which the polygon should be rotated
               clockwise (default: 0)
         fill  value with which the inside of the box is to be filled
               (default: None, meaning it is unfilled)
    threshold  value used to distinguish pixels on the circle from
               surrounding ones, passed on into fill_outline (default: 50)
    """
    im = reshape3 (im)
    angle = 2.0 * math.pi / nsides
    y0 = yc + r * math.sin (rotate)
    x0 = xc + r * math.cos (rotate)
    vertices = []
    for i in range (1, nsides+1):
        vertices.append ((y0, x0))
        y1 = yc + r * math.sin (i * angle + rotate)
        x1 = xc + r * math.cos (i * angle + rotate)
        draw_line (im, y0, x0, y1, x1, v, fast=fast)
        y0 = y1
        x0 = x1
    if not fill is None:
        fill_outline (im, yc, xc, fill, threshold=threshold)
    return vertices

#-------------------------------------------------------------------------------
def draw_star (im, yc, xc, radius, npoints, inner_radius=None,
               v=max_image_value, fast=False, rotate=0, fill=False,
               threshold=50):
    """
    Draw an npoints-pointed star of radius r centred at (yc, xc).

    Arguments:
              im  image upon which the text is to be written (modified)
              yc  y-value (row) of the centre of the star
              xc  x-value (column) of the centre of the star
          radius  radius of the circle in which the polygon is enclosed
         npoints  number of points that the star is to have
    inner_radius  radius of the inner parts of the star
               v  value to which pixels are to be set
                  (default: max_image_value)
            fast  if True, don't use anti-aliased lines (default: False)
          rotate  angle in radians through which the polygon should be
                  rotated clockwise (default: 0)
            fill  value with which the inside of the box is to be filled
                  (default: None, meaning it is unfilled)
       threshold  value used to distinguish pixels on the circle from
                  surrounding ones, passed on into fill_outline
                  (default: 50)
    """
    im = reshape3 (im)
    angle = math.pi / npoints
    if inner_radius is None: inner_radius = radius / 2
    y0 = yc + radius * math.sin (rotate)
    x0 = xc + radius * math.cos (rotate)
    np = 2 * npoints + 1
    vertices = []
    for i in range (1, np):
        vertices.append ((y0, x0))
        if (i // 2) * 2 == i: r = radius
        else:                 r = inner_radius
        y1 = yc + r * math.sin (i * angle + rotate)
        x1 = xc + r * math.cos (i * angle + rotate)
        draw_line (im, y0, x0, y1, x1, v, fast=fast)
        y0 = y1
        x0 = x1
    if fill:
        fill_outline (im, yc, xc, v, v//2, threshold=threshold)
    return vertices

#-------------------------------------------------------------------------------
def draw_text (im, text, y, x, v=max_image_value, size=1, bg=None, align="c"):
    """
    Write text onto an image.

    Arguments:
       im  image upon which the text is to be written (modified)
     text  string of characters to be written onto the image
        y  y-value (row) at which the text is to be written
        x  x-value (column) at which the text is to be written
        v  value to which pixels in the text are to be set
           (default: max_image_value)
     size  integer scale factor for repeating pixels when writing the text
    align  alignment of the text, one of (default: 'c')
           'c': centred
           'l': left-justified
           'r': right-justified

    This routine is based on C code kindly provided by Nick Glazzard of
    Speedsix.
    """
    global character_height, character_width, character_bitmap

    size = int (size)

    # Work out the start position on the image depending on the text alignment.
    if    align == 'l' or align == 'L':
         offset = 0
    elif  align == 'r' or align == 'R':
        offset = - character_width * len(text) * size
    else:
        offset = - character_width * len(text) * size // 2

    # Draw each character in turn, repeating each pixel size times.
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    for c in text:
        yy = y
        for row in range (0, character_height):
            b = character_bitmap[c][row]
            for ys in range (0, size):
                if yy >= 0 and yy < ny:
                    xx = x + offset
                    for col in range (0, character_width):
                        for xs in range (0, size):
                            if xx >= 0 and xx < nx:
                                cc = character_width - col
                                if b & (1<<cc):
                                    im[yy,xx,:] = v
                                elif not bg is None:
                                    im[yy,xx,:] = bg
                                xx += 1
                yy -= 1
        offset += character_width * size

#-------------------------------------------------------------------------------
def examine (im, name="", format="%3.0f", lformat=None, ff=False, fd=sys.stdout,
             ylo=0, xlo=0, yhi=None, xhi=None, clo=0, chi=None):
    """
    Output an image in human-readable form.

    Arguments:
         im  image whose pixels are to be output
       name  name of the image (default : '')
     format  format used for writing pixels (default: '%3.0f')
    lformat  format used for column and row numbers (default: contextual)
         ff  if True, output a form-feed before the output (default: False)
         fd  file on which output is to be written (default: sys.stdout)
        ylo  lower y-value of the region to be output (default: 0)
        xlo  lower x-value of the region to be output (default: 0)
        yhi  upper y-value of the region to be output (default: last pixel)
        xhi  upper x-value of the region to be output (default: last pixel)
        clo  first channel of the region to be output (default: 0)
        chi  last channel of the region to be output (default: last channel)
         ff  if True, output a form feed before the image (default: False)
    """
    # Work out the width of a printed pixel by setting "0".  We use that to
    # determine lformat, unless set explicitly by the caller.
    width = len (format % 0.0)
    if lformat is None: lformat = "%%%dd" % width

    # Print the introduction.
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    py = px = pc = "s"
    if ny == 1: py = ""
    if nx == 1: px = ""
    if nc == 1: pc = ""
    if ff: ffc = ''
    else:  ffc = ''
    if name != "": name += " "
    print (ffc + "Image %sis %d line%s, %d pixel%s/line, %d channel%s " \
           % (name, ny, py, nx, px, nc, pc), file=fd)

    # Print the element numbers across the top and a line.
    if yhi is None: yhi = ny
    if xhi is None: xhi = nx
    if chi is None: chi = nc
    nyl = yhi - ylo
    nxl = xhi - xlo
    sep = ' ' * width + '+' + '-' * (width+1) * nxl + '-+'
    print (' ' * (width+1), file=fd, end=" ")
    for x in range (xlo, xhi):
        print (lformat % (x), file=fd, end=" ")
    print ()
    print (sep, file=fd)

    # Print the pixels of the rows, with the channels above each other.
    for y in range (ylo, yhi):
        for c in range (clo, chi):
            if c == clo:
                print (lformat % (y) + '|', file=fd, end=" ")
            else:
                print (len (lformat % (y)) *  ' ' + '|', file=fd, end=" ")
            for x in range (xlo, xhi):
                print (format % (im[y,x,c]), end=" ", file=fd)
            print ("|", file=fd)
        print (sep, file=fd)

#-------------------------------------------------------------------------------
def examine_latex (im, name="", format="%3.0f", lformat=None, fd=sys.stdout,
                   ylo=0, xlo=0, yhi=None, xhi=None, clo=0, chi=None):
    """
    Output a region of an image in human-readable form as a LaTeX table.

    Arguments:
         im  image whose pixels are to be output
       name  name of the image (default : '')
     format  format used for writing pixels (default: '%3.0f')
    lformat  format used for column and row numbers (default: contextual)
         fd  file on which output is to be written (default: sys.stdout)
        ylo  lower y-value of the region to be output (default: 0)
        xlo  lower x-value of the region to be output (default: 0)
        yhi  upper y-value of the region to be output (default: last pixel)
        xhi  upper x-value of the region to be output (default: last pixel)
        clo  first channel of the region to be output (default: 0)
        chi  last channel of the region to be output (default: last channel)
    """
    # Work out the width of a printed pixel by setting "0".  We use that to
    # determine lformat, unless set explicitly by the caller.
    width = len (format % 0.0)
    if lformat is None: lformat = "%%%dd" % width
    sep = "  \\hline\n"

    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    if yhi is None: yhi = ny
    if xhi is None: xhi = nx
    if chi is None: chi = nc

    # Format the introduction to the table.
    text = "\\begin{table}\n  \\centering\n  \\begin{tabular}{r|"
    for x in range (xlo, xhi):
        text += "r"
    text += "|}\n"

    # Print the element numbers across the top and a line.
    for x in range (xlo, xhi):
        text += " & " + lformat % x
    text += "\\\\\n"
    text += sep

    # Print the pixels of the rows, with the channels above each other.
    for y in range (ylo, yhi):
        for c in range (clo, chi):
            if c == clo:
                text += lformat % (y)
            for x in range (xlo, xhi):
                text += " & " + format % (im[y,x,c])
            text += "\\\\\n"
    text += sep

    # Finish off the table.
    text += "  \\end{tabular}\n"
    py = px = pc = "s"
    if ny == 1: py = ""
    if nx == 1: px = ""
    if nc == 1: pc = ""
    if name != "": name += " "
    text += \
        "  \\caption{Image %sis %d line%s, %d pixel%s/line, %d channel%s}\n" \
        % (name, ny, py, nx, px, nc, pc)
    text += "\\end{table}"

    # Output the text we've produced.
    print (text, file=fd)

#-------------------------------------------------------------------------------
def examine_markdown (im, name="", format="%3.0f", lformat=None, fd=sys.stdout,
                   ylo=0, xlo=0, yhi=None, xhi=None, clo=0, chi=None):
    """
    Output a region of an image in human-readable form as a Makedown table
    compatible with Pandoc.

    Arguments:
         im  image whose pixels are to be output
       name  name of the image (default : '')
     format  format used for writing pixels (default: '%3.0f')
    lformat  format used for column and row numbers (default: contextual)
         fd  file on which output is to be written (default: sys.stdout)
        ylo  lower y-value of the region to be output (default: 0)
        xlo  lower x-value of the region to be output (default: 0)
        yhi  upper y-value of the region to be output (default: last pixel)
        xhi  upper x-value of the region to be output (default: last pixel)
        clo  first channel of the region to be output (default: 0)
        chi  last channel of the region to be output (default: last channel)
    """
    # Work out the width of a printed pixel by setting "0".  We use that to
    # determine lformat, unless set explicitly by the caller.
    width = len (format % 0.0)
    if lformat is None: lformat = "%%%dd" % width

    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    if yhi is None: yhi = ny
    if xhi is None: xhi = nx
    if chi is None: chi = nc

    # Print the element numbers across the top and a line.
    text = "| "
    format_line = "|----:|"
    for x in range (xlo, xhi):
        text += "|" + lformat % x
        format_line += "------:|"
    text = text + "|\n" + format_line + "\n"

    # Print the pixels of the rows, with the channels above each other.
    for y in range (ylo, yhi):
        for c in range (clo, chi):
            text += "|"
            if c == clo:
                text += lformat % (y)
            for x in range (xlo, xhi):
                text += "|" + format % (im[y,x,c])
            text += "|\n"

    # Finish off the table.
    py = px = pc = "s"
    if ny == 1: py = ""
    if nx == 1: px = ""
    if nc == 1: pc = ""
    if name != "": name += " "
    text += ": Image %sis %d line%s, %d pixel%s/line, %d channel%s\n" \
        % (name, ny, py, nx, px, nc, pc)

    # Output the text we've produced.
    print (text, file=fd)

#-------------------------------------------------------------------------------
def effect_drawing (im, blursize=17, opacity=0.9):
    """
    Convert an image into a 'drawing' and return it.

    Arguments:
         im  image to be converted into a 'drawing'
       blur  size of the square mask to be used for blurring the image
             (default: 17)
    opacity  the opacity to be used when blending the blur with
             the original (default: 0.9)
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    if nc > 1: im1 = mono (im)
    else:      im1 = copy (im)

    # Invert the contrast.
    hi = maxval (im1)
    im2 = hi - im1

    # Blur the image.
    blurmask = image ((blursize, blursize, 1))
    set (blurmask, 1.0)
    convolve (im2, blurmask, 'mean')

    # Blend the layers, clipping the result to keep it sensible.
    fac = opacity / maxval (im2)
    im2 =  im1 / (1.0 - im2 * fac)
    clip (im2, 0.0, max_image_value)
    return im2

#-------------------------------------------------------------------------------
def effect_sepia (im):
    """
    Make an image have a sepia appearance.

    Arguments:
    im  image to be made sepia (modified)
    """
    im = reshape3 (im)
    r = get_channel (im, 0)
    g = get_channel (im, 1)
    b = get_channel (im, 2)
    rr = 0.393 * r + 0.769 * g + 0.189 * b
    gg = 0.349 * r + 0.686 * g + 0.168 * b
    bb = 0.272 * r + 0.534 * g + 0.131 * b
    set_channel (im, 0, rr)
    set_channel (im, 1, gg)
    set_channel (im, 2, bb)
    clip (im, 0.0, max_image_value)

#-------------------------------------------------------------------------------
def effect_streaks (im, width=1, height=6, direction='h', occ=0.9, fg=0.0,
                    bg=max_image_value):
    """
    Return a representation of an image as horizontal or vertical streaks.

    Convert an image `im` into a series of two-level streaks, the width of
    the streak indicating the darkness of that part of the original
    image.  The inspiration for the routine is the illustrations in the
    book 'The Cloudspotter's Guide' by Gavin Pretor-Pinny.

    Arguments:
           im  image to be processed
        width  width of each region to be processed (default: 1)
       height  width of each region to be processed (default: 6)
    direction  direction in which the streaks will go (default: 'h')
               'h': horizontal
               'v': vertical
          occ  maximum occupancy of each region (default: 0.9)
           fg  foreground value (0.0)
           bg  background value (max_image_value)

    """
    # Produce single-channel output, even from a multi-channel input image.
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    to = image ((ny, nx, 1))

    # Get the image range and handle the case when the image is blank.
    lo, hi = extrema (im)
    if lo >= hi:
        set (to, 0.0)
        return to

    # Process the image in regions of width x height pixels.  For each region,
    # we calculate its mean, then determine what proportion of that region
    # should be filled with the foreground value.  We then set the entire
    # region to the background value, and finally fill the relevant proportion
    # of the region with the foreground value.

    # There are two aesthetic refinements to this process.  The first is that
    # we incorporate a contrast reversal if the foreground value is darker
    # than the background one.  The second is that we reduce the proportion of
    # the region that is filled by an occupancy factor as this makes the end
    # result look better.
    if direction == 'v' or direction == 'V':
        horiz = False
        fac = width * occ / (hi - lo)
    else:
        horiz = True
        fac = height * occ / (hi - lo)
    for y in range (0, ny, height):
        yto = y + height
        if yto > ny:  yto = ny
        for x in range (0, nx, width):
            xto = x + width
            if xto > nx:  xto = nx
            reg = region (im, y, yto, x, xto)
            lo, hi = extrema (reg)
            if bg > fg:
                val = int((hi - mean(reg)) * fac + 0.5)
            else:
                val = int((mean (reg) - lo) * fac + 0.5)
            if val < 0: val = 0
            set_region (to, y, x, yto, xto, bg)
            if horiz:
                set_region (to, y, x, y + val, xto, fg)
            else:
                set_region (to, y, x, yto, x + val, fg)
    return to

#-------------------------------------------------------------------------------
def effect_solarize (im, threshold=None):
    """
    Solarize an image.  Solarization, sometimes called the Sabattier effect
    in photography, involves reversing the contrast of the values of pixels
    in `im` above `threshold`.  If `threshold` is not supplied, a value of
    half the maximum image value is used.

    Arguments:
           im  image to be solarized (modified)
    threshold  threshold above which the effect is applied

    """
    im = reshape3 (im)
    value = maxval (im)
    if threshold is None: threshold = value / 2.0
    ny, nx, nc = sizes (im)
    for y in range (0, ny):
        for x in range (0, nx):
            for c in range (0, nc):
                if im[y,x,c] < threshold:
                    im[y,x,c] = value - im[y,x,c]

#-------------------------------------------------------------------------------
def expand (im, mask):
    """
    Perform a morphological expand on `im` with `mask`, returning the result.

    Arguments:
      im  image to be processed
    mask  mask image
    """
    return convolve (im, mask, "max")

#-------------------------------------------------------------------------------
def extract (im, ry, rx, yc, xc, step=1.0, angle=0.0, wrap=False, val=0,
             interpolator='gradient'):
    """
    Return a ry x rx-pixel region of im centred at (yc, xc).

    Arguments:
              im  image from which the region is to be extracted
              ry  number of pixels in the y-direction of the region
              rx  number of pixels in the x-direction of the  region
              yc  y-position of the centre of the region to be extracted
              xc  x-position of the centre of the region to be extracted
            step  step size on im (default: 1.0)
                  or a list [ystep, xstep]
           angle  angle of sampling grid relative to im, measured
                  anticlockwise in radians (default: 0.0)
            wrap  if True, 'falling off' one size of the image will wrap
                  around to the opposite side (otherwise, such pixels will
                  be zero) (default: False)
             val  value to which pixels outside the image are set if not
                  wrapping (default: 0)
    interpolator  interpolation scheme, one of 'gradient', 'bilinear' or
                  'nearest' (default: 'gradient')

    The region extracted from im can be centred around a non-integer
    position in im, and extracted with an arbitrary step size at an
    arbitrary angle.  The interpolation schemes supported are either
    conventional bilinear or a gradient-based one described in
    P.R. Smith (Ultramicroscopy vol 6, pp 201--204, 1981), as well as
    simple nearest neighbour.
    """
    if isinstance (step, list):
        ystep, xstep = step
    else:
        ystep = xstep = step
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    region = image ((ry, rx, nc))
    y0 = yc - ry // 2
    x0 = xc - rx // 2
    if abs (angle) < tiny and \
            abs (ystep - 1.0) < tiny and abs (xstep - 1.0) < tiny and \
            ry < ny and rx < nx and y0 >= 0 and x0 >= 0 and \
            abs (y0 - int(y0)) < tiny and abs (x0 - int(x0)) < tiny:
        region = im[y0:y0+ry,x0:x0+rx]
    else:
        mim = im
        if interpolator == 'gradient':
            interp = 2
            mim = mono (im)
        elif interpolator == 'bilinear':  interp = 3
        elif interpolator == 'nearest':   interp = 1
        else:
            print (('extract: invalid interpolator "%s"; ' + \
                  'using "nearest"') % interpolator, file=fd)
            interp = 1
        cosfac = math.cos (-angle)
        sinfac = math.sin (-angle)
        disty = ry / 2
        distx = rx / 2
        yst = yc + distx * xstep * sinfac - disty * ystep * cosfac
        xst = xc - distx * xstep * cosfac - disty * ystep * sinfac
        for y in range (0, ry):
            ypos = yst
            xpos = xst
            for x in range (0, rx):
                if (ypos < 0 or ypos >= ny or xpos < 0 or xpos >= nx) \
                        and not wrap:
                    region[y,x] = val
                else:
                    ylo = int (ypos)
                    dy =  ypos - ylo
                    dy1 = 1 - dy
                    ylo = (ylo + ny) % ny
                    yhi = ylo + 1
                    if yhi >= ny and not wrap:
                        region[y,x] = val
                    else:
                        yhi = (yhi + ny) % ny
                        xlo = int (xpos)
                        dx = xpos - xlo
                        dx1 = 1 - dx
                        xlo = (xlo + nx) % nx
                        xhi = xlo + 1
                        if xhi >= nx  and not wrap:
                            region[y,x] = val
                        else:
                            xhi = (xhi + nx) % nx
                            if interp == 3:
                                region[y,x] = \
                                    dy *dx*im[yhi,xhi] + dy *dx1*im[yhi,xlo] + \
                                    dy1*dx*im[ylo,xhi] + dy1*dx1*im[ylo,xlo]
                            elif interp == 2:
                                if abs (mim[ylo,xlo] - mim[yhi,xhi]) > \
                                        abs (mim[yhi,xlo] - mim[ylo,xhi]):
                                    region[y,x] = (dx-dy) * im[ylo,xhi] + \
                                        dx1*im[ylo,xlo] + dy*im[yhi,xhi]
                                else:
                                    region[y,x] = (dx1-dy) * im[ylo,xlo] + \
                                        dx*im[ylo,xhi] + dy*im[yhi,xlo]
                            else:
                                region[y,x] = im[int(ylo+0.5),int(xlo+0.5)]
                ypos -= ystep * sinfac
                xpos += xstep * cosfac
            yst += ystep * cosfac
            xst += xstep * sinfac
    return region

#-------------------------------------------------------------------------------
def extrema (im):
    """
    Return the minimum and maximum of an image.

    Arguments:
    im  image whose extrema are to be found
    """
    return [im.min(), im.max()]

#-------------------------------------------------------------------------------
def fill_outline (im, y, x, v=max_image_value, threshold=50):
    """
    Flood fill the region lying within a border.

    Arguments:
           im  image containing region to be filled (modified)
           yc  y-coordinate of point at which filling is to start
           xc  x-coordinate of point at which filling is to start
            v  value to which the filled region is to be set
               (default: max_image_value)
    threshold  minimum difference in value from centre pixel at boundary
               (default: 50)

    This code is based on that written by Eric S. Raymond at
    http://mail.python.org/pipermail/image-sig/2005-September/003559.html
    This code is an elegant Python implementation of Paul Heckbert's
    classic flood-fill algorithm, presented in"Graphics Gems".
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    if x < 0 or x >= nx or y < 0 or y >= ny: return
    vc = im[y,x].sum()
    vv = sum_elements (v)
    if abs(vc - vv) < threshold: return
    im[y,x] = v

    # At each step there is a list of edge pixels for the flood-filled
    # region.  Check every pixel adjacent to the edge; for each, if it is
    # eligible to be coloured, colour it and add it to a new edge list.
    # Then you replace the old edge list with the new one.  Stop when the
    # list is empty.
    edge = [(y, x)]
    while edge:
        newedge = []
        for (y, x) in edge:
            for (t, s) in ((y, x+1), (y, x-1), (y+1, x), (y-1, x)):
                if s >= 0 and s < nx and t >= 0 and t < ny and \
                   abs(im[t,s].sum() - vc) < threshold:
                    im[t,s] = v
                    newedge.append ((t, s))
        edge = newedge

#-------------------------------------------------------------------------------
def find_in_path (prog):
    """
    Return the absolute pathname of a program which is in the search path.

    Arguments:
    prog  program whose absolute filename is to be found
    """
    # First, split the PATH variable into a list of directories, then find
    # the first program from our list that is in the path.
    path = os.environ['PATH'].split (os.pathsep)
    for p in path:
        fp = os.path.join(p, prog)
        if os.path.exists(fp): return os.path.abspath(fp)
    return None

#-------------------------------------------------------------------------------
def find_peaks (im, threshold):
    """
    Return a list of the peaks in an image in descending order of height.

    A peak is defined as a pixel whose value is larger than those of all
    surrounding pixels and has a value greater than threshold.  Each
    peak is described by a three-element list containing its pixel value
    and the y- and x-values at which the peak was found.

    Arguments:
           im  image whose peaks are to be found
    threshold  value used for determining which peaks are significant
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    peaks = list ()
    for y in range (1, ny-1):
        for x in range (1, nx-1):
            if      im[y,x,0] > im[y-1,x-1,0] \
                and im[y,x,0] > im[y-1,x  ,0] \
                and im[y,x,0] > im[y-1,x+1,0] \
                and im[y,x,0] > im[y  ,x-1,0] \
                and im[y,x,0] > im[y  ,x+1,0] \
                and im[y,x,0] > im[y+1,x-1,0] \
                and im[y,x,0] > im[y+1,x  ,0] \
                and im[y,x,0] > im[y+1,x+1,0] \
                and im[y,x,0] > threshold:
                peaks.append ([im[y,x,0], y, x])
    # Return the peaks sorted into descending order.
    peaks.sort (reverse=True)
    return peaks

#-------------------------------------------------------------------------------
def find_skin (im, hlo=300, hhi=30, slo=10, shi=70, vlo=10, vhi=80,
               ishsv=False):
    """
    Return a binary mask identifying pixels that are likely to be skin.

    This routine identifies potential skin pixels in the image im.  The
    image is converted to HSV format unless ishsv is True, and then
    pixels in the HSV region bounded by [hlo:hhi], [slo:shi] and
    [vlo:vhi] are identified as being skin.  As skin is vaguely red, and
    red in HSV space is 0, hlo will normally be about 330 (degrees) and
    hhi about 30 degrees.  The returned image has non-zero pixels where
    skin has been identified.

    Note that this is not a reliable skin detector it can be confused by
    the use of incandescent lighting or by similarly-coloured materials
    such as wood.

    Arguments:
       im  image in which skin regions are to be found
      hlo  lowest skin hue (default: 300)
      hhi  highest skin hue (default: 30)
      slo  lowest skin saturation (default: 10)
      shi  highest skin saturation (default: 70)
      vlo  lowest skin value (default: 10)
      vhi  highest skin value (default: 80)
    ishsv  if True, the input image contains pixels in HSV format
            rather than RGB (default: False)

    """
    return segment_hsv (im, hlo, hhi, slo, shi, vlo, vhi, ishsv)

#-------------------------------------------------------------------------------
def find_threshold_otsu (im):
    """
    Return the optimal image threshold, found using Otsu's method.

    This routine is minimally adapted from the code in 'ImageP.py' by
    Tamas Haraszti, which he says is ultimately derived from Octave code
    written by Barre-Piquot.  I note, in passing, that this facility is
    also available as part of Matlab, though I've never seen the code.
    The algorithm itself is N. Otsu: 'A Threshold Selection Method from
    Gray-Level Histograms', IEEE Transactions on Systems, Man and
    Cybernetics vol 9 no 1 pp 62-66 (1979).

    Arguments:
    im  image for which the threshold is to be found

    """
    vals = im.copy ()
    mn = vals.min ()
    vals = vals - mn
    N = int (vals.max ())
    h, x = numpy.histogram (vals, bins=N)
    h = h / (h.sum() + 1.0)
    w = h.cumsum ()
    i = numpy.arange (N, dtype=float) + 1.0
    mu = numpy.zeros (N, float)
    mu = (h*i).cumsum()
    w1 = 1.0 - w
    mu0 = mu / w
    mu1 = (mu[-1] - mu) / w1
    s = w * w1 * (mu1 - mu0)**2
    return float ((s == s.max()).nonzero()[0][0]) + mn

#-------------------------------------------------------------------------------
def flatten_list (l, ltypes=(list, tuple)):
    "Flatten an arbitrarily-nested set of lists."
    ltype = type (l)
    i = 0
    while i < len (l):
        while isinstance (l[i], ltypes):
            if not l[i]:
                l.pop (i)
                i -= 1
                break
            else:
                l[i:i+1] = l[i]
        i += 1
    return ltype(l)

#-------------------------------------------------------------------------------
def fourier (im, forward=True):
    """
    Return the Fourier transform of an image.  Note that the zero frequency is
    in the centre of the image.

    Arguments:
         im  image to be transformed
    forward  if True, preform a forward transform (default: True)
    """
    if forward:
        # Transform, then move the origin to the centre of the image.
        temp = numpy.fft.fft2 (im, axes=(-3,-2))
        res = numpy.fft.fftshift (temp, axes=(-3,-2))
    else:
        # Move the origin from the centre to the corner, then transform.
        temp = numpy.fft.ifftshift (im, axes=(-3,-2))
        res = numpy.fft.ifft2 (temp, axes=(-3,-2))
    return res

#-------------------------------------------------------------------------------
def frange (lo, hi, inc):
    """
    A floating-point analogue of `range`, returning a list of values from `lo`
    to `hi` inclusive in `inc` steps.

    Arguments:
     lo  first value to be returned
     hi  last value that may be returned
    inc  increment between values

    The developers of Python avoided having `range` support floating-point
    values because they were concerned that results may vary from machine to
    machine.  This remains the case but the convenience of having a single
    routine doing all this work often outweighs the possible disadvantages.
    However, you should use this routine with caution.
    """
    # Although we're working in Python 3, we avoid `yield` and generate a list.
    # We run separate loops for positive and negative values of `inc`.
    vals= []
    if inc > 0:
        v = lo
        while v <= hi:
            vals += [v]
            v += inc
    else:
        v = lo
        while v >= hi:
            vals += [v]
            v += inc
    return vals

#-------------------------------------------------------------------------------
def glcm (im, yshift, xshift, max=256):
    """
    Return the grey-level co-occurrence matrix (GLCM) of im for yshift and
    xshift.

    Arguments:
        im  the image from which the channel is to be extracted
    yshift  downward shift on the image
    xshift  rightward shift on the image
       max  number of possible grey level values
    """
    im = reshape3 (im)
    glcm = numpy.zeros((max, max, 1), dtype=int)

    # Make a shifted version of the image, then calculate the GLCM.
    copy = im.copy ()
    shifted = shift_and_wrap (im, yshift, xshift)

    for i, j in zip(copy.ravel(), shifted.ravel()):
        glcm[int(i), int(j), 0] += 1
    return glcm

#-------------------------------------------------------------------------------
def get_channel (im, c):
    """
    Return a channel of an image.

    Arguments:
    im  the image from which the channel is to be extracted
     c  the index of the channel that is to be extracted
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    ch = image ((ny, nx, 1))
    ch[:,:,0] = im[:,:,c]
    return ch

#-------------------------------------------------------------------------------
def grab ():
    """
    Return an image grabbed using the computer's camera.
    """
    global PROGRAMS

    # Most of the work is done in a single call.
    pic = invoke_image_program ("grab image")
    return pic

#-------------------------------------------------------------------------------
def grab_screen ():
    """
    Return an image of the screen
    """
    # Most of the work is done in a single call.
    pic = invoke_image_program ("grab screen")

    # Apple's screencapture returns a PNG file with alpha channel but for
    # consistency we want only a three-channel image.  Handle this case and
    # return the image.
    ny, nx, nc = sizes (pic)
    if nc > 3:
        pic = pic[:,:,:3]
    return pic

#-------------------------------------------------------------------------------
def harris (im, window_size=10, k=0.04, threshold=10000, min_separation=20,
            disp=False, offset=None):
    """
    Return corners found in an image using the Harris and Stephens
    detector.

    Arguments:
              im  image to be processed
     window_size  the size of the window slid over the image
                  (default: 10)
               k  the constant used with calculating the response of
                  a corner (default: 0.04)
       threshold  minimum response for a pixel to be considered a
                  corner (default: 10000)
  min_separation  if greater than zero, set the minimum distance apart
                  of corners (default: 20)
            disp  if set, display the corners on a darkened image
                  (default: False)
          offset  if supplied, this is added to corner locations in an attempt
                  to overcome the systematic error in the operator; if omitted,
                  a value is calculated from window_size

    This routine was adapted from code written by Jordan Hughes at UC
    Santa Barbara (see https://github.com/hughesj919/HarrisCorner) with
    a few refinements inspired by code written by Jan Erik Solem (see
    http://www.janeriksolem.net/2009/01/harris-corner-detector-in-python.html).
    """
    # Ensure we have a one-channel image.
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    if nc == 1:
        mim = copy (im)
    else:
        mim = mono (im)

    # Find the x and y derivatives; numpy requires a 2D shape for the data.
    mim = im.reshape (ny, nx)
    dy, dx = numpy.gradient (mim)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2

    # Set the fiddle factor that attempts to overcome the systematic error
    # in locations introduced by the operator.
    if offset is None:
        offset = window_size // 2

    # Loop through the image and find the corners.  At each position, we
    # calculate the response as in the paper by Harris and Stephens.
    # Response values above threshold are candidate corners.
    responses = []
    for y in range (offset, ny-offset):
        for x in range (offset, nx-offset):
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = windowIxx.sum ()
            Sxy = windowIxy.sum ()
            Syy = windowIyy.sum ()

            # Calculate the determinant and trace, and hence the corner
            # response.
            det = Sxx * Syy - Sxy**2
            trace = Sxx + Syy
            r = det - k * trace**2
            if r > threshold:
                responses += [[r, y, x]]

    # If min_separation is greater than zero, sort the responses into
    # descending order, then make sure no corner location is within
    # min_distance of another.  The calculation of the response strengths
    # introduces a systematic error which we attempt to correct by shifting
    # the corners detected by the offset used above; however, I appreciate
    # this is an imperfect correction.
    if min_separation > 0:
        sorted (responses, key= lambda resp: resp[0], reverse=True)
        corners = [[responses[0][0], responses[0][1], responses[0][2]]]
        d2 = min_separation**2
        for rr, ry, rx in responses:
            far = True
            for cr, cy, cx in corners:
                sep2 = (ry - cy)**2 + (rx - cx)**2
                if sep2 < d2:
                    far = False
                    break
            if far:
                corners += [[rr, ry, rx]]
        # Shift the corners to (hopefully) put them in the right places.
        for ic in range (0, len(corners)):
            r, cy, cx = corners[ic]
            corners[ic] = [r, cy+offset, cx+offset]
    else:
        # Copy across all the responses.
        corners = []
        for rr, ry, rx in responses:
            corners += [[rr, ry, rx]]
    sorted (corners, reverse=True)

    # Optionally display the locations on the image.
    if disp:
        dispim = mono_to_rgb (mim)
        mark_peaks (dispim, corners, symbol=".", size=1, disp=True)

    return corners

#-------------------------------------------------------------------------------
def high_peaks (peaks, factor=0.5):
    """
    Given a sorted list of peaks, return those within factor of the
    highest.

    Arguments:
    peaks   list of peaks sorted into descending order
    factor  peaks of height within factor of the highest are returned
            (default: 0.5)
    """
    threshold = peaks[0][0] * factor
    n = 0
    for ht, y, x in peaks:
        if ht < threshold: break
        n += 1
    return peaks[:n]

#-------------------------------------------------------------------------------
def histogram (im, bins=64, limits=None, disp=False):
    """
    Find the histogram of an image.

    Arguments:
        im  image for which the histogram is to be found
      bins  number of bins in the histogram (default: 64)
    limits  extrema between which the histogram is to be found
            (default: calculated from the image)
      disp  if True, the histogram will be drawn (default: False)
    """
    h, a = numpy.histogram (im, bins, limits)
    a = a[:bins]
    if disp:
        graph (a, h, title="Histogram",
               xlabel="bin", ylabel="number of pixels",
               style="histogram")
    return a, h

#-------------------------------------------------------------------------------
def hough_line (im, nr=300, na=200, threshold=10, max_peaks=None,
                disp=False, dispacc=False, v=max_image_value):
    """
    Perform a straight-line Hough transform of the image 'im', which
    should normally contain output from an edge detector.  It returns a
    list of the significant peaks found (see find_peaks for a
    description of its content) and the image that forms the
    accumulator.  The accumulator is of dimension [nr, na], the distance
    from the origin (yc, xc) being plotted along the y-direction (first
    subscript) and the corresponding angle along the x-direction (second
    subscript).

    Arguments:
           im  image for which the Hough transform is to be performed
           nr  number of radial values (y-direction of the accumulator)
           na  number of angle values (x-direction of the accumulator)
    threshold  minimum value for a significant peak in the accumulator
               (default: 10)
    max_peaks  maximum number of peaks to retain (default: None)
         disp  if True, draw the lines found over the image (default: False)
      dispacc  if True, display the accumulator array (default: False)
            v  value used when drawing lines (default: max_image_value)
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    acc = image ((nr, na, 1))
    ainc = math.pi / na
    rinc = math.sqrt (ny**2 + nx**2) / nr

    # Fill arrays with the radius and angle values, to be returned.
    rvals = []
    for i in range (0, nr):
        rvals += [i * rinc]

    avals = []
    for i in range (0, na):
        avals += [i * ainc]

    # Find non-zero points and update the Hough accumulator.
    for y in range (0, ny):
        for x in range (0, nx):
            val = im[y,x,0]
            if val > 0:
                for ia in range (0, na):
                    ang = ia * ainc
                    r = x * math.cos(ang) + y * math.sin (ang)
                    ir = int (r / rinc)
                    acc[ir,ia,0] += 1

    # Find peaks in the accumulator.
    peaks = find_peaks (acc, threshold=threshold)
    if max_peaks is None:
        max_peaks = len (peaks)

    # If the user wants to display what has been found, show them.
    if dispacc:
        dacc = log1 (acc)
        contrast_stretch (dacc)
        mark_peaks (dacc, peaks[:max_peaks], v=255, disp=True,
                    name="Hough peaks")

    if disp:
        nn = max (ny, nx)
        d = make_three_channel (im)
        for val, r, a in peaks[0:max_peaks]:
            radius = rvals[r]
            angle = avals[a]
            x0 = int (radius * math.cos (angle))
            y0 = int (radius * math.sin (angle))
            x1 = x0 - int (nn * math.sin(angle))
            y1 = y0 + int (nn * math.cos(angle))
            x2 = x0 + int (nn * math.sin(angle))
            y2 = y0 - int (nn * math.cos(angle))
            draw_line (d, y1, x1, y2, x2, v)
        display (d, name="Lines from Hough peaks")

    return peaks, acc, rvals, avals

#-------------------------------------------------------------------------------
def hsv_to_rgb (im):
    """
    Convert an image from HSV space to RGB.

    This routine converts an image in which the hue, saturation and
    value components are in channels 0, 1 and 2 respectively to the RGB
    colour space.  It is assumed that hue lies in the range [0,359],
    while saturation and value are percentages; these are compatible
    with the popular display program 'xv'.  The red, green and blue
    components are returned in channels 0, 1 and 2 respectively, each in
    the range [0,max_image_value].

    This routine is adapted from code written by Frank Warmerdam
    <warmerdam@pobox.com> and Trent Hare; see
    http://svn.osgeo.org/gdal/trunk/gdal/swig/python/samples/hsv_merge.py

    Arguments:
    im  image to be converted (modified)

    """
    h = im[:,:,0] / 360.0
    s = im[:,:,1] / 100.0
    v = im[:,:,2] * max_image_value / 100.0
    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    im[:,:,0] = i.choose (v, q, p, p, t, v)
    im[:,:,1] = i.choose (t, v, v, q, p, p)
    im[:,:,2] = i.choose (p, p, t, v, v, q)

#-------------------------------------------------------------------------------
def image (fromwhat, type=numpy.float32):
    """
    Create and return an EVE image.

    Arguments:
    fromwhat  the source from which the image is to be created, one of:
                     a string:  the name of a file to be read in
                a numpy array:  the new image is a copy of this image
              a list or tuple:  the dimensions (ny, nx, nc)
        type  the type of the image to be ceated (default: numpy.float32)
    """
    if isinstance (fromwhat, str):
        fn = fromwhat.lower ()
        # We read in PNM-format images ourselves (albeit slowly) so that
        # we work on machines where PIL isn't available (such as the author's
        # Planet Gemini under termux).
        if fn.endswith (".pbm") or fn.endswith (".pgm") or \
           fn.endswith (".ppm") or fn.endswith (".pnm"):
            return read_pnm (fromwhat)
        # Otherwise, use PIL to read in the image.
        from PIL import Image
        fn = os.path.expanduser (fromwhat)
        pic = Image.open (fn)
        # Something seems to be broken with at least 16-bit TIFFs...
        if pic.mode == "I;16":
            temp = numpy.fromstring(pic.tostring(), dtype=numpy.uint16)
            im = numpy.asarray (temp, dtype=type)
            nc = 1
        else:
            im = numpy.asarray (pic, dtype=type)
            nc = len (pic.getbands ())
            nx, ny = pic.size
            im.shape = [ny, nx, nc]
    elif isinstance (fromwhat, numpy.ndarray):
        fromwhat = reshape3 (fromwhat)
        ny, nx, nc = fromwhat.shape
        im = numpy.zeros ((ny, nx, nc))
    elif isinstance (fromwhat, list) or isinstance (fromwhat, tuple):
        im = numpy.zeros (fromwhat, dtype=type)
    else:
        raise ValueError ('Illegal argument type')
    return im

#-------------------------------------------------------------------------------
def image_from_values (shape, vals, type=numpy.float32):
    """
    Create an image from the three-element list or tuple `shape` and set it
    to the values in `vals`, which must be a 1-D list containing the same
    number of values.  The image that has been created and set is returned.
    The routine is intended principally for testing EVE.

    Arguments:
    shape  a three-element list or tuple containing (ny, nx,nc)
     vals  a list containing the values to be stored in the image
     type  data type of the created image (default: numpy.float32)
    """
    im = image (shape, type=type)
    el = 0
    ny, nx, nc = shape
    for y in range (0, ny):
        for x in range (0, nx):
            for c in range (0, nc):
                im[y,x,c] = vals[el]
                el += 1

    return im
    
#-------------------------------------------------------------------------------
def image_to_values (im):
    """
    Return the values stored in the elements of `im` as a list.
    The routine is intended principally for testing EVE.

    Arguments:
    im  image whose values are to be listed
    """
    vals = []
    ny, nx, nc = sizes (im)
    for y in range (0, ny):
        for x in range (0, nx):
            for c in range (0, nc):
                vals += [im[y,x,c]]

    return vals
    
#-------------------------------------------------------------------------------
def insert (im, reg, yc, xc, operation='='):
    """
    Insert image reg into im, centred at (yc, xc).

    Arguments:
           im  image into which the region is to be inserted (modified)
          reg  image which is to be inserted
           yc  y-value of im at which the centre of reg is to be inserted
           xc  x-value of im at which the centre of reg is to be inserted
    operation  way in which im is inserted into output, one of
               '=' (assign)
               '+' (add)
               '-' (subtract)
               '*' (multiply),
               '/' (divide)
    """
    im = reshape3 (im)
    reg = reshape3 (reg)
    ny, nx, nc = sizes (reg)
    ylo = yc - ny // 2
    yhi = ylo + ny
    xlo = xc - nx // 2
    xhi = xlo + nx
    if operation == '=':
        im[ylo:yhi,xlo:xhi,:] = reg
    elif operation == '+':
        im[ylo:yhi,xlo:xhi,:] += reg
    elif operation == '-':
        im[ylo:yhi,xlo:xhi,:] -= reg
    elif operation == '*':
        im[ylo:yhi,xlo:xhi,:] *= reg
    elif operation == '/':
        im[ylo:yhi,xlo:xhi,:] /= reg
    else:
        raise ValueError ('Invalid operation type')

#-------------------------------------------------------------------------------
def invoke_image_program (category, delete=True, im=None):
    global PROGRAMS

    fnout = None
    for prog, inv in PROGRAMS[category]:
        if find_in_path (prog):
            cmd = inv
            # Generate any output file needed.
            for phrase in ["%o_pgm", "%o_ppm", "%o_png"]:
                if cmd.find (phrase) >= 0:
                    fnout = tempfile.mkstemp (suffix="." + phrase[-3:])[1]
                    cmd = cmd.replace (phrase, fnout)
                    write (im, fnout)

            for phrase in ["%i_pgm", "%i_ppm", "%i_png"]:
                if cmd.find (phrase) >= 0:
                    fnin = tempfile.mkstemp (suffix="." + phrase[-3:])[1]
                    cmd = cmd.replace (phrase, fnin)

            # Execute the program.
            # print ("[" + cmd + "]")
            os.system (cmd)
            break

    # Read in the result.  There seems to be a bug on Linux when using scrot
    # under Ubuntu 20.04: the filename has a trailing "_000.png" rather than
    # ".png".  Try with the proper filename and if that doesn't work, try the
    # 'wrong' one.
    try:
        input = image (fnin)
        fnin_000 = ""
    except:
        fnin_000 = fnin[:-4] + "_000.png"
        input = image (fnin_000)

    # Tidy up by deleting temporary files.
    if delete:
        os.remove (fnin)
        if fnin_000 != "":
            os.remove (fnin_000)
    if fnout is not None:
        os.remove (fnout)
    return input
    
#-------------------------------------------------------------------------------
def label_regions (im, con8=False):
    """
    Given a segmented image, return an image with its regions labelled and
    the number of regions found.

    Arguments:
      im  image to be labelled
    con8  if True, consider all 8 nearest neighbours
          if False, consider only 4 nearest neighbours (default)
    """
    import scipy.ndimage

    if con8: ele = [[[ 1,  1,  1,], [ 1,  1,  1,], [ 1,  1,  1,]],
                    [[ 1,  1,  1,], [ 1,  1,  1,], [ 1,  1,  1,]],
                    [[ 1,  1,  1,], [ 1,  1,  1,], [ 1,  1,  1,]]]
    else:    ele = None
    res, nlabs = scipy.ndimage.measurements.label (im, structure=ele)
    return res, nlabs

#-------------------------------------------------------------------------------
def label_regions_slow (im, con8=True):
    """
    Given a segmented image, return an image with its regions labelled
    and the number of regions found.

    Arguments:
      im  image to be labelled
    con8  if True, consider all 8 nearest neighbours
          if False, consider only 4 nearest neighbours
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    lab = image ((ny, nx, nc), type=numpy.int32)
    vals = [0, 0, 0, 0]
    labs = [1, 0, 0, 0]

    # The upper left pixel is in region zero.
    lastlabel = 0
    equiv = [lastlabel]
    lab[0,0,0] = lastlabel

    # Process the rest of the first row of the image.
    y = 0
    for x in range (1, nx):
        if im[y,x,0] != im[y,x-1,0]:
            lastlabel += 1
            equiv.append (lastlabel)
        lab[y,x,0] = lastlabel

    # Process the first column of the image.
    x = 0
    for y in range (1, ny):
        if im[y,x,0] == im[y-1,x,0]:
            lv = lab[y-1,x,0]
        else:
            lastlabel += 1
            equiv.append (lastlabel)
            lv = lastlabel
        lab[y,x,0] = lv

    # Process the remainder of the image.
    for y in range (1, ny):
        y1 = y - 1
        for x in range (1, nx):
            if con8: nv = 4
            else:    nv = 2
            x1 = x - 1
            x2 = x + 1
            if x2 >= nx - 1 and con8: lastcol = True; nv -= 1
            else: lastcol = False
            val = im[y,x,0]
            # Get the four neighbours' values and labels, taking care not
            # to index off the end of the image.
            vals[0] = im[y, x1,0]; labs[0] = lab[y, x1,0]
            vals[1] = im[y1,x, 0]; labs[1] = lab[y1,x, 0]
            if con8:
                vals[2] = im[y1,x1,0]; labs[2] = lab[y1,x1,0]
                if not lastcol:
                    vals[3] = im[y1,x2,0]; labs[3] = lab[y1,x2,0]
            inreg = False
            for i in range (0, nv):
                if val == vals[i]: inreg = True
            if not inreg:
                # We're in a new region.
                lastlabel += 1
                equiv.append (lastlabel)
                lv = lastlabel
            else:
                # We must be in the same region as a neighbour.
                matches = []
                for i in range (0, nv):
                    if val == vals[i]: matches.append (labs[i])
                matches.sort ()
                lv = int(matches[0])
                for v in matches[1:]:
                    if equiv[v] > lv:
                        equiv[v] = lv
                    elif lv > equiv[v]:
                        equiv[lv] = equiv[v]
            lab[y,x,0] = lv

    # Tidy up the equivalence table.
    remap = list()
    nc = -1
    for i in range (0, len(equiv)):
        if equiv[i] == i:
            nc += 1
            v = nc
        else:
            v = i
            while equiv[v] != v:
                v = equiv[v]
            v = remap[v]
        remap.append (v)

    # Make a second pass through the image, re-labelling the regions, then
    # return the labelled image.
    for y in range (0, ny):
        for x in range (0, nx):
            lab[y,x,0] = remap[lab[y,x,0]]
    return lab, maxval(lab)

#-------------------------------------------------------------------------------
def labelled_region (labim, lab, bg=0.0, fg=max_image_value):
    """
    Return region with label `lab` from labelled image `labim`.

    Arguments:
     im  labelled image
    lab  the label that defines which region of the image to return
     bg  value to which pixels outside the region are set (default: 0.0)
     fg  value to which pixels inside the region are set
         (default: max_image_value)
    """
    im = image (labim)
    set (im, bg)
    im[numpy.where (labim == lab)] = fg
    return im

#-------------------------------------------------------------------------------
def log1 (im):
    """
    Return the result of adding unity to every pixel of `im` and taking its
    logarithm; this keeps zeros unchanged.

    Arguments:
    im  image

    """
    im = numpy.log (im + 1)
    return im

#-------------------------------------------------------------------------------
def lut (im, table, stretch=False, limits=None):
    """
    Use a lookup table to adjust pixel values.

    Arguments:
         im  image to be adjusted (modified)
      table  look-up table used to adjust pixel values
    stretch  if True, the image will first be contrast-stretched
             between limits
     limits  a two-element list containing the minimum and maximum
             values to be used for scaling the extrema of the image
             after every pixel of the image has been processed
             (default: [0, max_image_value])
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    ntable = len (table)
    if stretch:
        if limits is None:
            lo = 0.0
            hi = max_image_value
        else:
            lo, hi = extrema (im)
        contrast_stretch (im, low=lo, high=hi)
    for y in range (0, ny):
        for x in range (0, nx):
            for c in range (0, nc):
                v = im[y,x,c]
                if v >= 0 and v < ntable: im[y,x,c] = table[int(v)]

#-------------------------------------------------------------------------------
def make_three_channel (im):
    """
    Return a three-channel version of im.
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)

    # If im is a three-channel one, return it.
    if nc == 3:
        return im

    # If im isn't a three-channel image, make it mono then convert it into
    # a three-channel grey-scale one.
    if nc == 1:
        im1 = im
    else:
        im1 = mono (im)

    im3 = image ((ny, nx, 3))
    im3[:,:,0] = im1[:,:,0].copy ()
    im3[:,:,1] = im1[:,:,0].copy ()
    im3[:,:,2] = im1[:,:,0].copy ()
    return im3    
    
#-------------------------------------------------------------------------------
def mark_at_position (im, y, x, v=max_image_value, symbol='.', size=9):
    """
    Plot a marker in an image.

    Arguments:
        im  image in which the positions are to be marked (modified)
         y  y-position of the centre of the mark
         x  x-position of the centre of the mark
     value  value to which peak locations will be set
            (default: max_image_value)
    symbol  what is to be plotted, one of (default: '+'):
            '.'  a single pixel
            '+'  a vertical cross
            'x'  a diagonal cross
            'o'  a 3 x 3 circle
      size  size of the plotting symbol (default: 9)
    """
    im = reshape3 (im)
    half = math.ceil (size / 2)
    yy = y - half
    xx = x - half
    if symbol == '.':
        im[y,x,:] = v
    elif symbol == '+':
        draw_line_fast (im, yy, x, yy+size+1, x, v)
        draw_line_fast (im, y, xx, y, xx+size+1, v)
    elif symbol == 'x':
        draw_line_fast (im, yy, xx, yy+size+1, xx+size+1, v)
        draw_line_fast (im, yy, xx+size+1, yy+size+1, xx, v)
    elif symbol == 'o':
        im[y-1,x-1] = im[y-1,x] = im[y-1,x+1] = v
        im[y,x-1] = im[y,x] = im[y,x+1] = v
        im[y+1,x-1] = im[y+1,x] = im[y+1,x+1] = v
    else:
        raise ValueError ('Unrecognised plotting point: "' + symbol + '"')

#-------------------------------------------------------------------------------
def mark_features (im, locs, v=max_image_value, fac=1.0, fast=False,
                   disp=True, scale=1.0):
    """
    Mark the positions, sizes and orientations of features in an image.

    Given a list of feature locations such as those returned by SIFT, in
    which each element of the list consists of a list of y-position,
    x-position, scale and orientation, this routine marks them on the
    image im.  Each line drawn is drawn with value val and is scaled by
    the factor fac.  If disp is True, the result is displayed.

    Arguments:
       im  image in which the features are to be drawn (modified)
     locs  a list of the features to be drawn
      fac  scale factor for features drawn on im (default: 1.0)
     fast  if True, lines are drawn for speed rather than appearance
           (default: False)
     disp  if True, the resulting image is displayed (default: True)
    scale  factor to multiply the image by before marking points
           (default: 1.0)
    """
    im = reshape3 (im)
    im *= scale
    fac = 1.0
    ny, nx, nc = sizes (im)
    for y, x, s, o in locs:
        yy = y + fac * s * math.sin (-o)
        xx = x + fac * s * math.cos (-o)
        if yy < 0: yy = 0
        if yy >= ny: yy = ny - 1
        if xx < 0: xx = 0
        if xx >= nx: xx = nx - 1
        draw_line (im, y, x, yy, xx, v, fast=fast)
    if disp: display (im)

#-------------------------------------------------------------------------------
def mark_matches (im1, im2, matches, v=max_image_value, fast=False,
                  threshold=0.0, disp=True, name="Matches", number=False):
    """
    Draw lines between corresponding match points, returning the result.

    Arguments:
          im1  first image for which matches have been found
          im2  second image for which matches have been found
      matches  matches between features found by match_descriptors and
               refined by select_matches
            v  value with which the line will be drawn
               (default: max_image_value)
         fast  if True, lines are drawn for speed rather than appearance
               (default: False)
         disp  if True, the resulting image will be displayed
               (default: True)
         name  name passed to eve.display if the image is displayed
               (default: "Matches")
       number  if True, add the match score alongside each line
               (default: False)

    Given im1 and im2, a new image is formed whch displays them side by
    side, and then lines are drawn between corresponding points stored
    in matches.  The routine is intended for displaying the results of
    feature found by SIFT, matched by match_descriptors, and refined by
    select_matches.
    """
    im1 = reshape3 (im1)
    ny1, nx1, nc1 = sizes (im1)
    im2 = reshape3 (im2)
    ny2, nx2, nc2 = sizes (im2)
    ny = ny1 if ny1 > ny2 else ny2
    nx = nx1 + nx2
    nc = nc1 if nc1 > nc2 else nc2
    dim = image ((ny,nx,nc))
    dim[0:ny1,0:nx1,0:nc1] = im1
    dim[0:ny2,nx1:,0:nc2] = im2
    for y1, x1, y2, x2, val in matches:
        y1 = int (y1)
        x1 = int (x1)
        y2 = int (y2)
        x2 = int (x2) + nx1
        draw_line (dim, y1, x1, y2, x2, v, fast=fast)
        if number:
            offset = 3
            if x2 + character_width + offset >= nx1 + nx2:
                draw_text (dim, '%g' % val, y2, x2-offset, v, align="r")
            else:
                draw_text (dim, '%g' % val, y2, x2+offset, v, align="l")
    if disp: display (dim, name=name)
    return dim

#-------------------------------------------------------------------------------
def mark_peaks (im, pos, v=max_image_value, disp=False, scale=1.0,
                symbol='+', size=9, name='Peaks'):
    """
    Mark the positions of peaks in an image, such as those returned by
    find_peaks().

    Arguments:
        im  image in which the peak positions are to be marked (modified)
       pos  list of peaks, each element itself a list of [height, y, x]
         v  value to which peak locations will be set
            (default: max_image_value)
      disp  if True, display the marked-up image
     scale  multiply the image by the factor before marking points
    symbol  what is to be plotted, one of (default: '+'):
            '.'  a single pixel
            '+'  a vertical cross
            'x'  a diagonal cross
            'o'  a 3 x 3 blob
      size  size of the plotting symbol (default: 9)
      name  name for eve.display if the image is displayed
            (default: 'Peaks')
    """
    im = reshape3 (im)
    im *= scale
    for ht, y, x in pos:
        mark_at_position (im, y, x, v, symbol, size)
    if disp: display (im, name=name)

#-------------------------------------------------------------------------------
def mark_positions (im, pos, v=max_image_value, disp=False, scale=1.0,
                    symbol='+', size=9, name='Positions'):
    """
    Mark positions in an image.

    Arguments:
        im  image in which the positions are to be marked (modified)
       pos  list of positions, each element itself a list of [y, x]
         v  value to which peak locations will be set
            (default: max_image_value)
      disp  if True, display the marked-up image (default: False)
     scale  multiply the image by the factor before marking points
    symbol  what is to be plotted, one of (default: '+'):
            '.'  a single pixel
            '+'  a vertical cross
            'x'  a diagonal cross
      size  size of the plotting symbol (default: 9)
      name  name for eve.display if the image is displayed
            (default: 'Positions')
    """
    im = reshape3 (im)
    im *= scale
    for y, x in pos:
        mark_at_position (im, y, x, v, symbol, size)
    if disp: display (im, name=name)

#-------------------------------------------------------------------------------
def match_descriptors_euclidean (desc1, desc2):
    """
    Given pairs of descriptors from SIFT or similar, return the Euclidean
    distances between all pairs, sorted into ascending order.

    Arguments:
    desc1  first set of descriptors
    desc2  second set of descriptors
    """
    score = []
    for i1 in range (0, len(desc1)):
        d1 = desc1[i1]
        for i2 in range (0, len(desc2)):
            d2 = desc2[i2]
            s = ((d1 - d2)**2).sum()
            score.append ([s, i1, i2])
    score.sort()
    return score

#-------------------------------------------------------------------------------
def match_descriptors_anglewise (d1, d2, factor=0.6):
    """
    Given pairs of normalized descriptors from SIFT or similar, return
    their best matches sorted into ascending order of match score.

    The match score is calculated as follows.  For each descriptor in
    d1, the scalar product is calculated with all descriptors in d2 and
    the best value (smallest angle between scalar products) taken.  If
    that value is greater than factor of the second-best value, the
    match is considered ambiguous and discarded; otherwise, triplet of
    the score and the indices into d1 and d2 are inserted into a list of
    scores.  When all possible combinations of d1 and d2 have been
    considered, that list is sorted into ascending order and returned.

    Arguments:
        d1  first set of descriptors
        d2  second set of descriptors
    factor  largest permissible value for a match (default: 0.6)

    """
    nd = d1.shape[0]
    score = []
    for i in range (0,nd):
        inprod = numpy.dot (d1[i], d2.T)
        inprod[numpy.where (inprod >  1.0)] =  1.0
        inprod[numpy.where (inprod < -1.0)] = -1.0
        angles = numpy.arccos (inprod)
        ix = numpy.argsort (angles)
        if angles[ix[0]] < factor * angles[ix[1]]:
            score.append ([angles[ix[0]], i, ix[0]])
    score.sort()
    return score

#-------------------------------------------------------------------------------
def maxval (im):
    """
    Return the maximum value of the pixels of an image.

    Arguments:
    im  image for which the maximum value is to be found
    """
    return im.max()

#-------------------------------------------------------------------------------
def mean (im):
    """
    Return the mean of the pixel values an image.

    Arguments:
    im  image for which the mean value is to be found
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    return numpy.sum (im) / (ny * nx * nc)

#-------------------------------------------------------------------------------
def minval (im):
    """
    Return the minimum of the pixel values of an image.

    Arguments:
    im  image for which the minimum value is to be found
    """
    return im.min()

#-------------------------------------------------------------------------------
def modulus_squared (im):
    """
    Return the squared modulus of each pixel of an image, usually to form
    the power spectrum after a Fourier transform.

    Arguments:
    im  image for which the power spectrum is to be formed

    """
    t = im * numpy.conjugate (im)
    return t.real.copy ()

#-------------------------------------------------------------------------------
def mono (im):
    """
    Average the channels of colour image to give a monochrome one,
    returning the result.

    Arguments:
    im  image to be converted to monochrome

    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    monoim = image ([ny, nx, 1])
    for c in range (0, nc):
        monoim[:,:,0] += im[:,:,c]
    monoim /= nc
    return monoim

#-------------------------------------------------------------------------------
def mono_to_rgb (im):
    """
    Produce a three-channel image of a monochrome image, returning the
    result.

    Arguments:
    im  image to be converted from monochrome to colour

    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    cim = image ([ny, nx, 3])
    cim[:,:,0] = im[:,:,0]
    cim[:,:,1] = im[:,:,0]
    cim[:,:,2] = im[:,:,0]
    return cim

#-------------------------------------------------------------------------------
def moravec (im, threshold=10):
    """
    Return the result of applying Moravec's corner detector on image `im`.
    A list of peaks is returned in the format produced by EVE routine
    `find_peaks`.

    Arguments:
           im  the image in which the corners are to be found
    threshold  minimum difference from background to be considered a corner
               (default: 10)
    """
    shifts = [(1, 0), (1, 1), (0, 1), (-1, 1)]
    ny, nx, nc = sizes (im)
    vals = image ((ny, nx, nc))
    for y in range (1, ny-1):
        for x in range (1, nx-1):
            lo = huge
            for s in shifts:
                diff = im[y + s[1], x + s[0]]
                diff = (diff - im[y,x])**2
                if diff < lo: lo = diff
            vals[y,x,0] = lo

    return find_peaks (numpy.sqrt(vals), threshold)

#-------------------------------------------------------------------------------
def mse (im1, im2):
    """
    Return the mean-squared error (mean-squared difference) between two
    images.

    Arguments:
    im1  image form which im2 is to be subtracted
    im2  image to be subtracted from im1
    """
    im1 = reshape3 (im1)
    ny, nx, nc = sizes (im1)
    return ssd (im1, im2) / float(ny * nx * nc)

#-------------------------------------------------------------------------------
def opening (im, mask):
    """
    Perform a morphological opening on `im` with `mask`, returning the result.

    Arguments:
      im  image to be processed
    mask  mask image
    """
    eroded = shrink (im, mask)
    return expand (eroded, mask)

#-------------------------------------------------------------------------------
def oriented_bounding_box (im, disp=False, v=255):
    """
    Return the bounding box of a feature in binary image `im` as
    `[ystart, xstart, ystop, xstop]`.

    Arguments:
      im  image for which the bounding box is to be returned
    disp  if True, display the bounding box on the image (default: False)
       v  value used to draw the bounding box on the image (default: 255)
    """
    # Adapted from
    # https://stackoverflow.com/questions/32892932/create-the-oriented-bounding-box-obb-with-python-and-numpy
    y, x, c = numpy.where (im == 255)
    npts = len (y)
    if npts <= 0:
        print ("There appear to be no foreground pixels!", file=sys.stderr)
        return None
    elif npts == 1:
        #print ("There are too few foreground pixels!", file=sys.stderr)
        return None

    pts = numpy.zeros ((npts,2), dtype="int")
    for i in range (0, npts):
        pts[i,1] = x[i]
        pts[i,0] = y[i]

    ca = numpy.cov (pts, y=None, rowvar=0, bias=1)
    v, vect = numpy.linalg.eig (ca)
    tvect = numpy.transpose (vect)

    # Use the inverse of the eigenvectors as a rotation matrix and rotate the
    # points so they align with the x and y axes.
    ar = numpy.dot (pts,numpy.linalg.inv (tvect))

    # Get the minimum and maximum x and y.
    mina = numpy.min (ar, axis=0)
    maxa = numpy.max (ar, axis=0)
    diff = (maxa - mina) / 2

    # The centre is just midway between the min and max x,y.
    centre = mina + diff

    # Work out the four corners by subtracting and adding half the bounding
    # boxes' height and width from the centre.
    corners = numpy.array ([centre + [-diff[0], -diff[1]],
                            centre + [ diff[0], -diff[1]],
                            centre + [ diff[0],  diff[1]],
                            centre + [-diff[0],  diff[1]],
                            centre + [-diff[0], -diff[1]]])

    # Use the the eigenvectors as a rotation matrix and rotate the corners and
    # the centre back.
    corners = numpy.dot (corners, tvect)
    centre = numpy.dot (centre, tvect)

    # If required, display the result.
    if disp:
        for c in range (1, 5):
            draw_line_fast (im, corners[c-1][0], corners[c-1][1],
                            corners[c][0], corners[c][1], v=v)
            display (im)
    return corners[:4]

#-------------------------------------------------------------------------------
def pca_channels (im):
    """
    Perform a principal component analysis of the channels of im,
    returning the eigenvalues, kernel (eigenvectors) and means.  For an
    image with nc channels, there are nc eigenvalues etc; this form of
    the PCA is best-suited to analysing the similarity of images or for
    reducing noise.  This form of the PCA is not suitable for
    applications such as eigenfaces -- use pca_images for that.

    Arguments:
    im  image for which the PCA is to be calculated (modified)

    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    covmat, aves = covariance_matrix (im)
    vals, vecs = numpy.linalg.eigh (covmat)
    perm = numpy.argsort(-vals)  # sort in descending order of eigenvalue
    vecs = vecs[:,perm].T        # transpose to give the kernel
    vals = vals[perm]
    im = pca_channels_project (im, vecs, aves)
    return vals, vecs, aves

#-------------------------------------------------------------------------------
def pca_channels_project (im, vecs, aves):
    """
    Project the image im using PCA eigenvectors vecs and channel means
    aves, both of which are produced by EVE's routine pca.

    Arguments:
      im  image for which the PCA is to be calculated (modified)
    vecs  eigenvectors from eve.pca_channels
    aves  channel means from eve.pca_channels

    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    for y in range (0, ny):
        for x in range (0, nx):
            v = im[y,x,:] - aves
            im[y,x,:] = numpy.dot (vecs, v)

#-------------------------------------------------------------------------------
def pca_decompose (data, turk_pentland=None):
    """
    Perform a principal component analysis a set of data, returning the
    eigenvalues, kernel (eigenvectors) and means.

    Each row (first subscript) of data needs to hold one set of samples
    (typically an image), while the columns (second subscript) hold the
    equivalent samples from a series of images (i.e., the same pixel in
    each image).

    Arguments:
             data  data to be decomposed
    turk_pentland  if True, eigenfaces-style decomposition is used
                   if False, conventional decomposition is performed
                   if unset, eigenfaces-style is done if the first
                   dimension is less than or equal to the second

    """
    # Mean-zero the data.
    nims, nvals = data.shape
    aves = data.mean (axis=0)
    data -= aves
    # Do the decomposition.
    if turk_pentland is None: turk_pentland = nims <= nvals
    if turk_pentland:
        # Using the `trick' due to Turk and Pentland.
        covmat = numpy.dot (data, data.T) / nvals
        ctrace = covmat.trace()
        evals, evecs = numpy.linalg.eigh (covmat)
        evecs = numpy.dot (data.T, evecs)
        # We need to normalise the eigenvectors.
        for i in range (0, nims):
            evecs[:,i] /= numpy.linalg.norm (evecs[:,i])
    else:
        # Straightforward (as in my PhD).
        covmat = numpy.dot (data.T, data) / nvals
        ctrace = covmat.trace()
        evals, evecs = numpy.linalg.eigh (covmat)

    # Sort the eigenvalues and eigenvectors into decreasing magnitude
    # of eigenvalue.
    idx = numpy.argsort (-evals)
    evals = evals[idx]
    evecs = evecs[:,idx].T
    return evals, evecs, aves

#-------------------------------------------------------------------------------
def pca_images (imageset, turk_pentland=None):
    """
    Perform a principal component analysis of a list of images,
    returning the eigenvalues, kernel (eigenvectors) and means.  This
    routine is suitable for use in applications such as eigenfaces.

    Arguments:
         imageset  list of image for which the PCA is to be calculated
    turk_pentland  if True, eigenfaces-style decomposition is used
                   if False, conventional decomposition is performed
                   if unset, eigenfaces-style is done if the first
                   dimension is less than or equal to the second

    """
    # Get the imge dimensions and create an array to hold all the data in a
    # form suitable for the decomposition, i.e. one image per row and as many
    # rows as there were images in the set.
    nims = len (imageset)
    im = reshape3 (imageset[0])
    ny, nx, nc = sizes (im)
    nvals = ny * nx
    data = numpy.zeros ((nims, nvals))

    # Fill the data array, converting to monochrome and re-sizing if required.
    for i in range (0, nims):
        im = reshape3 (imageset[i])
        imy, imx, imc = sizes (im)
        if imc != 1: im = mono (im)
        if imy != ny or imx != nx: im = resize (im, ny, nx, 2)
        data[i,:] = im.reshape (nvals)

    # Decompose the data and return the result.
    return pca_decompose (data, turk_pentland)

#-------------------------------------------------------------------------------
def pca_project (im, kernel, aves):
    """
    Project im into the principal components defined by kernel and aves,
    returning the vector of coefficients that describe im.

    Arguments:
        im  image to be projected onto the principal components
    kernel  the PCA kernel returned by eve.pca_images
      aves  the PVA means returned by eve.pca_images
    """
    p = numpy.dot (im.reshape (1, -1) - aves, kernel.T)
    return p[0]

#-------------------------------------------------------------------------------
def pca_reconstruct (vals, kernel, aves):
    """
    Reconstruct an image from a set of coefficients following principal
    component analysis.  The data that are returned will need to be
    reshaped into the correct image dimensions.

    Arguments:
      vals  vector of coefficients that describe the image to be
            reconstructed
    kernel  the PCA kernel returned by eve.pca_images
      aves  the PVA means returned by eve.pca_images

    """
    return numpy.dot (vals, kernel) + aves

#-------------------------------------------------------------------------------
def perimeter (im, size=3, plus=False):
    """
    Given an image (usually containing only the values 0 and 255), find the
    perimeter of regions by performing a shrink and subtracting that from the
    original image.

    Arguments:
      im  image for which the perimeter is to be found (modified)
      mask  size of the mask used in the shrink (default: 3)
      plus  if True, use a +-shaped mask (default: False)

    """
    mask = image ((size,size,1))
    if plus:
        n2 = size // 2
        draw_line (mask, 0, n2, size, n2, v=1)
        draw_line (mask, n2, 0, n2, size, v=1)
    else:
        set (mask, 1.0)

    sh = shrink (im, mask)
    im -= sh

#-------------------------------------------------------------------------------
def pico (im, transform, fail=0, maxerr=10):
    """
    Apply `transform` to each pixel of the image `im`, where `transform` is
    a string containing an expression to be applied to each pixel in turn.
    The result of applying the transform is returned; the input image is
    unchanged.

    The book "Beyond Photography" by Gerard J. Holzmann (Prentice-Hall,
    1988) describes the use of a picture editor called `popi` to perform
    transformations on images, and `popi` was a portable version of a
    previous picture editor called `pico`, described in "PICO —-- A
    Picture Editor" by Gerard J. Holzmann (AT&T Technical Journal Vol. 66,
    No. 2 pp. 2-13, March/April 1987); see http://spinroot.com/pico/ for
    more details.  This routine was inspired by Holzmann's `pico`, though
    it uses a different syntax for the transformations.

    The following variables can be used in `transform`:
       y -- the y location of the pixel being processed
       x -- the x location of the pixel being processed
       r -- the radius of the pixel being processed
       a -- the angle of the pixel being processed in radians

       Y -- the maximum y location of the pixel being processed
       X -- the maximum x location of the pixel being processed
       R -- the maximum radius of the image
       Z -- the largest grey-level value that can be used
       H -- the largest grey-level value in the image
       L -- the smallest grey-level value in the image

       hist[] -- array containing the histogram of the image

       ycart(r,a) -- returns the y-value corresponding to polar
                     coordinates r,a
       xcart(r,a) -- returns the x-value corresponding to polar
                     coordinates r,a

    It is quite easy for `transform` to fail, in which case the relevant
    output pixel is set to `fail`.  The first `maxerr` errors generated
    are written to the standard output stream; subsequent errors are
    handled silently but the total number generated is written out after
    the whole image has been  processed.

    Arguments
           im  the image to be processed
    transform  a string describing the operation to be performed
         fail  value to which a pixel is set if `transform` generates an
               error (default: 0)
       maxerr  the maximum number of errors reported (default: 10)

    Some typical transforms are:
    contrast reversal:
      new[y,x] = H - im[y,x]

    contrast stretching:
      new[y,x] = Z * (im[y,x] - L) / (H - L)

    logarithmic grey-scaling:
      new[y,x] = Z * math.log (im[y,x,0]) / math.log(Z)

    solarization of blue channel:
      new[y,x] = (Z - im[y,x]) if (im[y,x,2] < H/2) else im[y,x]

    mirroring:
      new[y,x] = im[y,X-1-x]

    vertical flip:
      new[y,x] = im[Y-1-y,x]

    shrink:
      new[y,x] = im[y*5,x*3]

    enlarge:
      new[y,x] = im[y//3,x//3]

    3x3 blur:
      new[y,x] = (im[y-1,x-1] + im[y-1,x] + im[y-1,x+1] +
                  im[y  ,x-1] + im[y  ,x] + im[y  ,x+1] +
                  im[y+1,x-1] + im[y+1,x] + im[y+1,x+1]) // 9

    edge-detect:
      new[y,x] = im[y,x] - im[y+2, x+2]

    shrink the centre:
      new[y,x] = im[ycart(5*math.sqrt(r),a), xcart(5*math.sqrt(r),a)]

    expand the centre:
      new[y,x] = im[ycart(r*r/R,a), xcart(r*r/R,a)]

    swirl:
      new[y,x] = im[ycart(r,a+2*r), xcart(r,a+2*r)]

    the tartan of this image:
      new[y,x] = hist[y] + hist[x]

    superimpose a grid:
      new[y,x] = Z if (x % 10 == 0) or (y % 10 == 0) else im[y,x]

    make a triangle pattern:
      new[y,x] = y % x

    You can undoubtedly think of others.
    """

    # Remove extraneous whitespace from the transform: leading whitespace
    # causes eval to fail.
    transform = transform.strip ()

    # Create an output image of the same size as the input one.
    new = image (im)

    # Set a few useful values that can be used in the expression, mostly
    # limiting values.
    ny, nx, nc = sizes (im)
    yc = ny // 2
    xc = nx // 2
    Y = ny
    X = nx
    Z = int (max_image_value)
    L = minval (im)
    H = maxval (im)
    R = math.sqrt ((max (yc, xc) + 1)**2)
    junk, hist = histogram (im, bins=Z+1, limits=[0, Z])

    # Routines to convert back from polar to Cartesian.
    def xcart (r, a):
        return int (r * math.cos (a * math.pi / 180) + 0.5) + xc

    def ycart (r, a):
        return int (r * math.sin (a * math.pi / 180) + 0.5) + yc

    # Cycle over the pixels of im, evaluating the expression for each pixel
    # and handling any places where the expression fails.
    nerrs = 0
    for y in range (0, ny):
        for x in range (0, nx):
            r = math.sqrt ((y - yc)**2 + (x - xc)**2)
            a = math.atan2 (y - yc, x - xc) * 180 / math.pi  # in degrees
            try:
                exec (transform)
            except Exception as e:
                if nerrs < maxerr:
                    print (e, file=sys.stderr)
                elif nerrs == maxerr:
                    print ("...", file=sys.stderr)
                nerrs += 1
                new[y,x] = fail

    # Report the number of errors, if there were any.
    if nerrs > 0:
        print ("A total of %d errors were generated." % nerrs,
               file=sys.stderr)

    # Return the processed image.
    return new

#-------------------------------------------------------------------------------
def print_peaks (pos, format="%4d %4d: %.2f", intro=None, fd=sys.stdout):
    """
    Print a series of peaks out, one per line.

    Arguments:
       pos  list containing the peaks to be printed out
    format  format for (y,x) and height to be printed out
            (default: "%4d %4d: %.2f")
     intro  if supplied, this string is printed out before the peaks
        fd  file on which the output is to be written (default: sys.stdout)
    """
    if not intro is None:
        print (intro, file=fd)
    for ht, y, x in pos:
        print (format % (y, x, ht), file=fd)

#-------------------------------------------------------------------------------
def print_positions (pos, format="%4d %4d", intro=None, fd=sys.stdout):
    """
    Print a series of positions out, one per line.

    Arguments:
       pos  list containing the positions to be printed out
    format  format for (y,x) to be printed out (default: "%4d %4d")
     intro  if supplied, this string is printed out before the positions
        fd  file on which the output is to be written (default: sys.stdout)
    """
    if not intro is None:
        print (intro, file=fd)
    for y, x in pos:
        print (format % (y, x), file=fd)

#-------------------------------------------------------------------------------
def profile (im, y0, x0, y1, x1, disp=False):
    """
    Extract the image values on a straight line from (y0,x0) to (y1,x1),
    optionally displaying the result.

    Arguments:
      im  image from which the profile is desired
      y0  y-value (row) of the start of the line
      x0  x-value (column) of the start of the line
      y1  y-value (row) of the end of the line
      x1  x-value (column) of the end of the line
    disp  if set, display the profile on the image (default: False)
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    y0 = int (y0)
    x0 = int (x0)
    y1 = int (y1)
    x1 = int (x1)

    # If we're do display the result, make a copy of the image to scribble on.
    if disp:
        sketch = eve.copy (im)

    # The basis of the routine is the same code as draw_line_fast.
    steep = True if abs(y1 - y0) > abs(x1 - x0) else False
    if steep:
        y0, x0 = x0, y0
        y1, x1 = x1, y1
    if x0 > x1:
        x1, x0 = x0, x1
        y1, y0 = y0, y1
    dx = x1 - x0 + 0.0
    dy = abs(y1 - y0)
    e = 0.0
    if dx == 0.0:
        de = 1.0e30
    else:
        de = dy / dx
    y = y0
    if y0 < y1:
        ystep =  1
    else:
        ystep = -1
    np = x1 - x0 + 1

    npts = x1 - x0 + 1
    xvals = numpy.ndarray (npts)
    for c in range (0, npts):
        xvals[c] = x0 + c
    profile = numpy.ndarray ((nc, npts))
    ip = 0
    for x in range (x0,x1+1):
        v = None
        if steep:
            if x >= 0 and x < ny and y >= 0 and y < nx:
                v = im[x,y,:]
        else:
            if x >= 0 and x < nx and y >= 0 and y < ny:
                v = im[y,x,:]
        if v is not None:
            for c in range (0, nc):
                profile[c,ip] = v[c]
            ip += 1
        e += de
        if e >= 0.5:
            y += ystep
            e -= 1.0

    # Return the result.
    return xvals, profile

#-------------------------------------------------------------------------------
def ramp (im):
    """
    Fill an image with a grey-scale ramp.

    Arguments:
    im  image into which the pattern is written (modified)
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    im[:,:,:] = numpy.fromfunction (lambda i, j, k: i + j + k, ((ny, nx, nc)))

#-------------------------------------------------------------------------------
def read_pnm (fn):
    """
    Read a PBMPLUS-format (often called NETPBM-format) image and return it.
    Both ASCII and binary variants are supported.  The values stored
    actually stored in the file are returned; the format definition
    suggests that they should be re-scaled according to the so-called
    `maxval` stored in the file but it seems crazy to do this given the
    internal representation used in EVE.  If the image seems to have too
    little contrast after reading it in, you might consider using EVE
    routine contrast_stretch to re-scale it.

    Arguments:
    fn  filename containing the image or "-" to read standard input
    """
    if fn == "-":
        content = sys.stdin.read ()
    else:
        with open (fn, "rb") as f:
            content = f.read ()

    # For a file format that looks straightforward, PBMPLUS-format files are
    # surprisingly difficult to read, mostly because the header is textual
    # even when the content is binary.  There are actually six flavours of
    # file, identified by the first two bytes:
    #        ASCII binary  content
    #        P1    P4      bit-map
    #        P2    P5      grey-scale
    #        P3    P6      RGB colour
    # so let's look at those bytes and jump to the relevant piece of code.
    id = str (content[0:2], "ascii")

    if id == "P1":
        # Textual bitmap.  Set the number of channels and split the image into
        # tokens we can parse.
        tokens = _pnm_tokenize (content)
        ny = int (tokens[2])
        nx = int (tokens[1])
        nc = 1
        # Create and fill the image.
        im = image ((ny, nx, nc))
        y = 0
        x = 0
        # Unpack the image.  Spaces between the individual pixels are optional
        # so we walk along the tokens we have cut from the input, setting a
        # pixel for each character.  We should strictly check that ch below is
        # '0' or '1'...
        for t in tokens[3:]:
            for ch in t:
                im[y,x,0] = 0 if ch == '0' else 1
                x += 1
                if x >= nx:
                    x = 0
                    y += 1

    elif id == "P2" or id == "P3":
        # Textual grey-scale or colour image.  Set the number of channels and
        # split the image into tokens we can parse.
        nc = 1 if id == "P2" else 3
        tokens = _pnm_tokenize (content)
        ny = int (tokens[2])
        nx = int (tokens[1])
        # Make sure the number of tokens agrees with the image size.
        if ny * nx * nc + 4 != len (tokens):
            raise ValueError ("There are %d values to read, not %d!" \
                              % (len (tokens), ny*nx*nc+4))
        # Create and fill the image.
        im = image ((ny, nx, nc))
        it = 4
        for y in range (0, ny):
            for x in range (0, nx):
                for c in range (0, nc):
                    im[y,x,c] = int (tokens[it])
                    it += 1

    elif id == "P4":
        # Binary bitmap.  We start by parsing the header.
        header, ch = _pnm_header (content, 3)
        # Pull out the sizes and create the image.
        ny = int (header[2])
        nx = int (header[1])
        im = image ((ny, nx, 1))
        # Copy across the content, one bit at a time.
        y = 0
        x = 0
        for byte in content[ch:]:
            for bitpos in range (0, 8):
                if bitpos >= nx: break
                bitpos = 7 - bitpos
                bit = 1 << bitpos
                v = 0 if (byte & bit) == 0 else 1
                im[y,x,0] = v
                x += 1
                if x >= nx:
                    y += 1
                    x = 0

    elif id == "P5" or id == "P6":
        # Binary grey-scale or colour image.  Set the number of channels and
        # parse the header.
        nc = 1 if id == "P5" else 3
        header, ch = _pnm_header (content, 4)
        # Pull out the sizes and create the image.
        ny = int (header[2])
        nx = int (header[1])
        im = image ((ny, nx, nc))
        # Copy across the content.
        for y in range (0, ny):
            for x in range (0, nx):
                for c in range (0, nc):
                    im[y,x,c] = content[ch]
                    ch += 1
    else:
        raise ValueError ("First two bytes of PBMPLUS image file are '%s'!" \
                          % str (content))

    return im

#-------------------------------------------------------------------------------
def reduce (im, blocksize):
    """
    Reduce the size of an image by averaging each region of
    blocksize x blocksize pixels to a single pixel, returning the result.

    Arguments:
           im  image to be reduced in size
    blocksize  factor by which the size of the image is to be reduced
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    nny = ny // blocksize
    nnx = nx // blocksize
    nim = image ((nny, nnx, nc))
    for y in range (0, nny):
        ylo = y * blocksize
        yhi = ylo + blocksize
        for x in range (0, nnx):
            xlo = x * blocksize
            xhi = xlo + blocksize
            for c in range (0, nc):
                nim[y,x,c] = im[ylo:yhi,xlo:xhi,c].mean()
    return nim

#-------------------------------------------------------------------------------
def reflect_horizontally (im):
    """
    Reflect an image horizontally.

    Arguments:
    im  image to be reflected (modified)
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    nx2 = nx // 2
    for y in range (0, ny):
        for x in range (0, nx2):
            t = im[y,x,:].copy()
            im[y,x,:] = im[y,nx-x-1,:].copy()
            im[y,nx-x-1,:] = t

#-------------------------------------------------------------------------------
def reflect_vertically (im):
    """
    Reflect an image vertically.

    Arguments:
    im  image to be reflected (modified)
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    ny2 = ny // 2
    for y in range (0, ny2):
        for x in range (0, nx):
            t = im[y,x,:].copy()
            im[y,x,:] = im[ny-y-1,x,:].copy()
            im[ny-y-1,x,:] = t

#-------------------------------------------------------------------------------
def region (im, ylo, yhi, xlo, xhi):
    """
    Return a rectangular region of an image.

    Arguments:
     im  image from which the region is to be taken
    ylo  lower y-value (row) of the region
    yhi  higher y-value (row) of the region
    xlo  lower x-value (column) of the region
    xhi  higher x-value (column) of the region
    """
    im = reshape3 (im)
    return im[ylo:yhi,xlo:xhi,:]

#-------------------------------------------------------------------------------
def reshape2 (im):
    """
    For a three-index image as used by EVE, convert it to a two-index
    one if there is only one channel.  This converts an image from the
    format used by EVE into one that is compatible with OpenCV and
    scikit-image.

    Arguments:
       im  image that may need to be re-shaped

    """
    shape = im.shape
    if len (shape) == 3 and shape[2] == 1: im = im.reshape (shape[0], shape[1])
    return im

#-------------------------------------------------------------------------------
def reshape3 (im):
    """
    Ensure im has three indices, even if there is only one band.  This
    is principally used within EVE to avoid indexing errors on
    monochrome images that have been manipulated by OpenCV or
    scikit-image.

    Arguments:
       im  image that may need to be re-shaped

    """
    shape = im.shape
    if len (shape) == 2: im = im.reshape (shape[0], shape[1], 1)
    return im

#-------------------------------------------------------------------------------
def resize (im, nny, nnx, order=1):
    """
    Return im, re-sized to be of size (nny, nnx) by interpolation.

    Arguments:
       im  image to be re-sized
      nny  number of rows in the re-sized image
      nnx  number of columns in the re-sized image
    order  order of interpolating function (default: 1)
    """
    # The following is adapted from an example in the scipy cookbook.
    import scipy.ndimage
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    yl, xl = numpy.mgrid[0:ny-1:nny*1j,0:nx-1:nnx*1j]
    coords = scipy.array ([yl, xl])
    result = image ((nny, nnx, nc))
    for c in range (0, nc):
        result[:,:,c] = scipy.ndimage.map_coordinates (im[:,:,c], coords,
                                                       order=order)
    return result

#-------------------------------------------------------------------------------
def reverse_contrast (im):
    """
    Reverses the contrast in `im`, returning the result.

    Arguments:
       im  image whose contrast is to be reversed
    """
    temp = image (im)
    hi = maxval (im)
    set (temp, hi)
    return temp - im

#-------------------------------------------------------------------------------
def rgb_to_hsv (im):
    """
    Convert an image from RGB space to HSV.

    This routine converts an image in which the red, green and blue
    components are in channels 0, 1 and 2 respectively to the HSV colour
    space.  The hue, saturation and value components are returned in
    channels 0, 1 and 2 respectively.  Hue lies in the range [0,359]
    while saturation and value are percentages; these are compatible
    with the popular display program 'xv'.

    Arguments:
    im  image to be converted (modified)

    This routine is adapted from code written by Frank Warmerdam
    <warmerdam@pobox.com> and Trent Hare; see
    http://svn.osgeo.org/gdal/trunk/gdal/swig/python/samples/hsv_merge.py

    """
    im = reshape3 (im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]

    maxc = numpy.maximum (r, numpy.maximum(g, b))
    minc = numpy.minimum (r, numpy.minimum(g, b))
    v = maxc
    minc_eq_maxc = numpy.equal(minc,maxc)

    # Compute the difference, but reset zeros to ones to avoid divide
    # by zeros later.
    ones = numpy.ones ((r.shape[0], r.shape[1]))
    maxc_minus_minc = numpy.choose (minc_eq_maxc, (maxc-minc,ones))
    s = (maxc - minc) / numpy.maximum (ones, maxc)
    rc = (maxc - r) / maxc_minus_minc
    gc = (maxc - g) / maxc_minus_minc
    bc = (maxc - b) / maxc_minus_minc
    maxc_is_r = numpy.equal (maxc,r)
    maxc_is_g = numpy.equal (maxc,g)
    maxc_is_b = numpy.equal (maxc,b)
    h = numpy.zeros ((r.shape[0], r.shape[1]))
    h = numpy.choose (maxc_is_b, (h, 4.0 + gc - rc))
    h = numpy.choose (maxc_is_g, (h, 2.0 + rc - bc))
    h = numpy.choose (maxc_is_r, (h, bc - gc))
    im[:,:,0] = numpy.mod (h/6.0, 1.0) * 360.0
    im[:,:,1] = s * 100.0   # to be a percentage
    im[:,:,2] = v * 100.0 / max_image_value # to be a percentage

#-------------------------------------------------------------------------------
def rgb_to_mono (im):
    """
    Convert an image from RGB space to luminence (the Y of YIQ).

    This routine converts an image in which the red, green and blue
    components are in channels 0, 1 and 2 respectively to luminance,
    assuming the standard NTSC phosphor.  The result is returned in a
    new image.

    Arguments:
    im  image to be converted

    """
    im = reshape3 (im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    ny, nx, nc = sizes (im)
    lum = image ((ny, nx, 1))
    lum[:,:,0] = 0.299*r + 0.587*g + 0.114*b
    return lum

#-------------------------------------------------------------------------------
def rgb_to_yiq (im):
    """
    Convert an image from RGB space to YIQ.

    This routine converts an image in which the red, green and blue
    components are in channels 0, 1 and 2 respectively to the YIQ colour
    space, assuming the standard NTSC phosphor.  The Y, I and Q
    components are returned in channels 0, 1 and 2 respectively.

    Arguments:
    im  image to be converted (modified)

    """
    im = reshape3 (im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    im[:,:,0] = 0.299*r + 0.587*g + 0.114*b
    im[:,:,1] = 0.596*r - 0.275*g - 0.321*b
    im[:,:,2] = 0.212*r - 0.523*g + 0.311*b

#-------------------------------------------------------------------------------
def rotate90acw (im):
    """
    Return a copy of `im` rotated around its centre by 90 degrees anticlockwise.

    Arguments:
      im: image to be rotated
    """
    return numpy.rot90 (im, 1)

#-------------------------------------------------------------------------------
def rotate90cw (im):
    """
    Return a copy of `im` rotated around its centre by 90 degrees clockwise.

    Arguments:
      im: image to be rotated
    """
    return numpy.rot90 (im, 3)

#-------------------------------------------------------------------------------
def rotate180 (im):
    """
    Return a copy of `im` rotated around its centre by 180 degrees.

    Arguments:
      im: image to be rotated
    """
    return numpy.rot90 (im, 2)

#-------------------------------------------------------------------------------
def set_mean_sd (im, newmean, newsd):
    """
    Rescale the image to a given mean and sd.

    Arguments:
         im  image to be rescaled (modified)
    newmean  mean the image is to have after rescaling
      newsd  standard deviation that the image is to have after rescaling
    """
    oldmean = mean (im)
    oldsd = sd (im)
    im -= oldmean
    im /= oldsd
    im *= newsd
    im += newmean

#-------------------------------------------------------------------------------
def set_to_pattern (im, yfac=100, xfac=10, cfac=1):
    """
    Fill an image with a simple test pattern, wherein each pixel receives
    a unique value.

    Arguments:
         im  image to receive (modified)
       yfac  increment to make for each line (default: 100)
       xfac  increment to make for each pixel of a line (default: 10)
       cfac  increment to make for each channel of a pixel (default: 1)
    """
    ny, nx, nc = sizes (im)
    for y in range (0, ny):
        yy = yfac * y
        for x in range (0, nx):
            xx = xfac * x
            for c in range (0, nc):
                im[y,x,c] = yy + xx + cfac *c

#-------------------------------------------------------------------------------
def sd (im):
    """
    Return the standard deviation of an image.

    Arguments:
    im  image for which the standard deviation is to be found
    """
    return im.std (ddof=1)

#-------------------------------------------------------------------------------
def segment_hsv (im, hlo, hhi, slo, shi, vlo, vhi, ishsv=False):
    """
    Return a binary mask identifying pixels that fall within a region
    in HSV space.

    Arguments:
       im  image in which regions are to be found
      hlo  lowest HSV hue
      hhi  highest HSV hue
      slo  lowest HSV saturation
      shi  highest HSV saturation
      vlo  lowest HSV value
      vhi  highest HSV value
    ishsv  if True, the input image contains pixels in HSV format rather
           than RGB (default: False)
    """
    hsvim = copy (im)
    hsvim = reshape3 (hsvim)
    if not ishsv: rgb_to_hsv (hsvim)
    ny, nx, nc = sizes (hsvim)
    mask = image ((ny, nx, 1))
    h = hsvim[:,:,0]
    s = hsvim[:,:,1]
    v = hsvim[:,:,2]
    if hlo > hhi:
        m = ((h < hlo) | (hhi < h)) & (slo < s) & (s < shi) & \
            (vlo < v) & (v < vhi)          # we span 360 degrees
    else:
        m = ((hlo < h) & (h < hhi)) & (slo < s) & (s < shi) & \
            (vlo < v) & (v < vhi)
    mask[numpy.where (m)] = max_image_value
    return mask

#-------------------------------------------------------------------------------
def select_matches (scores, locs1, locs2, max_score_factor=5, max_matches=50):
    """Choose the matches with the best scores.

    Arguments:
              scores  list of scores from match_descriptors
               locs1  locations of features found on the first image
               locs2  locations of features found on the second image
    max_score_factor  ratio of the worst match to the best (default: 5)
         max_matches  maximum number of matches to return (default: 50)
    """
    n = len(scores)
    matches = []
    # If the best score is actually zero (i.e., perfect), turn off the threshold
    # by making it ludicrously big.
    if scores[0][0] != 0.0:
        thresh  = scores[0][0] * max_score_factor
    else:
        thresh = huge
    for i in range (0, n):
        if scores[i][0] <= thresh:
            i1 = scores[i][1] # first image
            i2 = scores[i][2] # second image
            y1 = locs1[i1,0]
            x1 = locs1[i1,1]
            y2 = locs2[i2,0]
            x2 = locs2[i2,1]
            matches.append ([y1, x1, y2, x2, i1 * i2])
            if len (matches) >= max_matches: break
        else:
            break
    return matches

#-------------------------------------------------------------------------------
def select_pixels_above (im, threshold):
    """
    Return a list of the location of all pixels in `im` above `threshold`.

    Arguments:
           im  image to be examined
    threshold  threshold above which pixels are identified
    """
    locs = []
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    for y in range (0, ny):
        for x in range (0, nx):
            for c in range (0, nc):
                if im[y,x,c] > threshold:
                    locs += [[y,x,c]]
    return locs

#-------------------------------------------------------------------------------
def select_pixels_below (im, threshold):
    """
    Return a list of the location of all pixels in `im` below `threshold`.

    Arguments:
           im  image to be examined
    threshold  threshold below which pixels are identified
    """
    locs = []
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    for y in range (0, ny):
        for x in range (0, nx):
            for c in range (0, nc):
                if im[y,x,c] < threshold:
                    locs += [[y,x,c]]
    return locs

#-------------------------------------------------------------------------------
def set (im, v):
    """
    Set all the pixels of an image to a constant value.

    Arguments:
    im  image to be set (modified)
     v  value to which the pixels are to be set
    """
    im = reshape3 (im)
    im[:,:,:] = v

#-------------------------------------------------------------------------------
def set_channel (im, c, chim):
    """
    Set a channel of an image to a constant value.

    Arguments:
      im  image in which the channel is to be inserted (modified)
       c  number of the channel which is to be set
    chim  single-channel image which is to be inserted in channel c
    """
    im[:,:,c] = chim[:,:,0]

#-------------------------------------------------------------------------------
def set_region (im, yfrom, xfrom, yto, xto, v):
    """
    Set a region of an image to a constant value.

    Arguments:
     im  image in which the region is to be set (modified)
    ylo  lower y-value (row) of the region
    yhi  higher y-value (row) of the region
    xlo  lower x-value (column) of the region
    xhi  higher x-value (column) of the region
      v  value to which the region is to be set
    """
    im = reshape3 (im)
    im[int(yfrom):int(yto), int(xfrom):int(xto), :] = v

#-------------------------------------------------------------------------------
def shift_and_wrap (im, y, x):
    """
    Shift image `im` by `y` and `x`, returning the result.

    Arguments:
    im  image to be processed
     y  y-shift (positive for down)
     x  x-shift (positive for right)
    """
    res = numpy.roll (im, int (y), axis=0)
    res = numpy.roll (res, int (x), axis=1)
    return res

#-------------------------------------------------------------------------------
def shrink (im, mask):
    """
    Perform a morphological shrink on `im` with `mask`, returning the result.

    Arguments:
      im  image to be processed
    mask  mask image
    """
    return convolve (im, mask, "min")

#-------------------------------------------------------------------------------
def sift (fn):
    """
    Return the SIFT keypoints of the image stored in file `fn`.  If the
    keypoints do not already exist, run the SIFT program from the popular
    VLfeat package (from http://www.vlfeat.rg/) to create them.
    The keypoints are stored for future use in a file whose name depends
    on `fn` but with the '.sift' extension.

    Arguments:
    fn  image file for which the keypoints are to be returned.
    """
    # The first thing to do is ascertain whether fn has a corresponding file
    # of SIFT keypoints already calculated.  We try the filename + ".sift" and
    # the filename a ".sift" extension instead.
    for xfn in [fn + ".sift", os.path.splitext(fn)[0] + '.sift']:
        if os.path.exists (xfn):
            return sift_keypoints (xfn)
    
    # There are no keypoints ready and waiting for us, so we have to run SIFT
    # to generate them.  However, VLfeat's SIFT requires PGM-format input
    # files, so if fn isn't in that format, we convert it to that.
    if fn.endswith (".pgm"):
        ifn = fn
    else:
        ifn = os.path.splitext(fn)[0] + '.pgm'
        im = image (fn)
        ny, nx, nc = sizes (im)
        if nc != 1:
            im = mono (im)
        print ("[Generating PGM-format file " + ifn + "]", file=sys.stderr)
        write (im, ifn)
        
    ofn = os.path.splitext(fn)[0] + '.sift'
    cmd = 'sift %s -o %s' % (ifn, ofn)
    print ("Executing " + cmd, file=sys.stderr)
    os.system (cmd)
    if ifn != fn:
        os.remove (ifn)
    return sift_keypoints (ofn)
    
#-------------------------------------------------------------------------------
def sift_keypoints (fn):
    """
    Return the SIFT keypoints from file fn.

    Two arrays are returned, the first giving the locations (and
    corresponding scales and orientations) of the feature points found,
    while the second contains the associated descriptors.

    Arguments:
    im  name of a file containing the SIFT keypoints
    """
    import scipy.linalg
    # Read in the keypoints from the file.  We read the entire file into memory,
    # where it ends up as a list with one line of the file in each element.
    # Each line contains exactly 132 elements which give the position and
    # orientation of the feature and its descriptor; we split these out into
    # the arrays called locs and descs, normalising the latter en route.  We
    # ultimately return locs and descs.
    fd = open (fn)
    lines = fd.readlines()
    fd.close()
    lf = 128          # length of each descriptor
    nf = len (lines)  # number of features
    if nf == 0: return None, None
    locs = numpy.zeros ((nf, 4))
    descs = numpy.zeros ((nf, lf))
    for f in range (0, nf):
        v = lines[f].split()
        p = 0
        # row, col, scale, orientation
        locs[f,1] = float (v[p])
        locs[f,0] = float (v[p+1])
        locs[f,2] = float (v[p+2])
        locs[f,3] = float (v[p+3])
        p += 4
        for i in range (0, lf):
            descs[f,i] = float (v[p+i])
        descs[f] = descs[f] / scipy.linalg.norm (descs[f])
    return locs, descs

#-------------------------------------------------------------------------------
def skeleton (im, mask, verbose=True):
    """
    Return the skeleton (medial axis transform) of an image with the structure
    element in mask.

    Arguments:
         im  image to be processed
       mask  structuring element (often a 3x3 '+' shape)
    verbose  if True, output the iteration number to stdout while running
             (default: True)
    """
    nits = 0
    ny, nx, nc = sizes (im)
    skel = image ((ny, nx, nc), type="uint8")
    while True:
        nits += 1
        if verbose: print (nits, end="\r")
        eroded = convolve (im, mask, statistic="min")
        opened = convolve (eroded, mask, statistic="max")
        temp = (im - opened).astype ("uint8")
        eroded = eroded.astype ("uint8")
        skel = numpy.bitwise_or (skel, temp)
        if eroded.sum () == 0:
            break
        else:
            im = copy (eroded)

    if verbose: print ()
    return skel

#-------------------------------------------------------------------------------
def sizes (im):
    """
    Return the dimensions of an image as a list.

    Arguments:
    im  the image whose dimensions are to be returned
    """
    return im.shape

#-------------------------------------------------------------------------------
def snr (im1, im2):
    """
    Return the signal-to-noise ratio between two images.

    Arguments:
    im1  first image to be used in calculating the SNR
    im2  first image to be used in calculating the SNR
    """
    r = correlation_coefficient (im1, im2)
    if r <= 0.0: return 0.0
    if r >= 1.0: return math.inf
    return math.sqrt (r / (1.0 - r))

#-------------------------------------------------------------------------------
def sobel (im):
    """
    Perform edge detection in im using the Sobel operator, returning the
    result.

    Arguments:
    im  image in which the edges are to be found
    """
    import scipy
    import scipy.ndimage as ndimage

    # Convert the EVE-format image into one compatible with scipy, run its
    # Sobel routine, then convert the result back into EVE format and return it.
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    if nc == 1: sci_im = im[:,:,0]
    else:       sci_im = mono(im)[:,:,0]
    grad_x = ndimage.sobel (sci_im, 0)
    grad_y = ndimage.sobel (sci_im, 1)
    grad_mag = scipy.sqrt (grad_x**2 + grad_y**2)
    gm = image ((ny,nx,1))
    gm[:,:,0] = grad_mag[:,:]
    return gm

#-------------------------------------------------------------------------------
def ssd (im1, im2):
    """
    Return the sum-squared difference between two images.

    Arguments:
    im1  image form which im2 is to be subtracted
    im2  image to be subtracted from im1
    """
    return ((im1 - im2)**2).sum()

#-------------------------------------------------------------------------------
def statistics (im, output=False, prefix="   ", fd=sys.stdout):
    """
    Return important statistics of an image.

    This routine returns the minimum, maximum, mean and standard
    deviation as a list.

    Arguments:
        im  image for which the statistics are to be calculated
    output  if True, output the statistics to fd (default: False)
    prefix  text to be output before each line of output (default: "   ")
        fd  file to which the output should be written
            (default: sys.stdout)

    """
    lo, hi = extrema (im)
    ave = mean (im)
    sdev = sd (im)
    if output:
        print (prefix + "mean:", ave, file=fd)
        print (prefix + "s.d.:", sdev, file=fd)
        print (prefix + "min: ", lo, file=fd)
        print (prefix + "max: ", hi, file=fd)
    return [lo, hi, ave, sdev]

#-------------------------------------------------------------------------------
def subsample (im, inc=2):
    """
    Sub-sample an image by selecting every inc-th pixel from every
    inc-th line.  The sub-sampled image is returned.

    Arguments:
     im  image to be sub-sampled
    inc  number of pixels between sub-samples (default: 2)
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    ny2 = ny // inc
    nx2 = nx // inc
    im2 = image ((ny2, nx2, nc))
    for y in range (0, ny2):
        for x in range (0, nx2):
            im2[y,x,:] = im[inc*y,inc*x,:]
    return im2

#-------------------------------------------------------------------------------
def sum (im):
    """
    Return the sum of all the values of an image.

    Arguments:
    im  image for which the sum is to be found
    """
    return im.sum()

#-------------------------------------------------------------------------------
def sum_elements (v):
    "Return the sum of elements of a list."
    if isinstance (v, list) or isinstance (v, tuple):
        sum = 0
        for el in v:
            sum += el
    else:
        sum = v
    return sum

#-------------------------------------------------------------------------------
def susan (im):
    """
    Process image `im` using the SUSAN feature point detector, returning
    the result.  An external program is used to perform the processing.

    Arguments:
         im  the image for which the keypoints are to be found
    """
    global PROGRAMS

    # SUSAN requires a single-channel image.
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    if nc == 1:
        im1 = im
    else:
        im1 = mono (im)

    susim = invoke_image_program ("susan", im=im1)
    return susim

#-------------------------------------------------------------------------------
def swap_channels (im):
    """
    Return an image with the red and blue channels interchanged, intended
    for converting OpenCV BGR images to the more conventional RGB order.

    Arguments:
      im  the image whose channels are to be swapped

    """
    res = im.copy ()
    ch0 = im[:,:,0]
    ch2 = im[:,:,2]
    res[:,:,2] = ch0[:,:]
    res[:,:,0] = ch2[:,:]
    return res
    
#-------------------------------------------------------------------------------
def thong (im, scale=64.0, offset=128.0):
    """
    Fill an image with Tran Thong's zone-plate-like test pattern.

    Arguments:
       im  image to contain the pattern (modified)
     scale  maximum deviation of the pattern from the mean
    offset  mean of the resulting pattern
    """
    # Work out the centre of the region and the various fiddle factors.
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    xc = nx // 2
    yc = ny // 2
    nmax = ny
    if nx > ny: nmax = nx
    rad = 0.4 * nmax
    rad2 = rad / 2.0
    radsqd = rad * rad
    radsq4 = radsqd / 4.0
    fac = 2 * math.pi * 0.496

    # Fill the region with the pattern.
    for y in range (0, ny):
        yy = (y - yc) **2
        for x in range (0, nx):
            rsqd = (x - xc) **2 + yy
            if rsqd <= radsq4:
                v = scale * math.cos (fac*rsqd/rad) + offset
                for c in range (0, nc):
                    im[y,x,c] = v
            else:
                r = math.sqrt (rsqd)
                v = scale * math.cos (fac * (2*r - rsqd/rad - rad2)) + offset
                im[y,x,:] = v

#-------------------------------------------------------------------------------
def transpose (im):
    """
    Transpose an image in the leading diagonal, returning the result.

    Arguments:
    im  image to be transposed
    """
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    tr = image ((nx, ny, nc))
    return numpy.transpose (im, axes=(1, 0, 2))

#-------------------------------------------------------------------------------
def variance (im):
    """
    Return the variance of an image.

    Arguments:
    im  image for which the standard deviation is to be found
    """
    return im.var (ddof=1)

#-------------------------------------------------------------------------------
def version ():
    """
    Return our version, which is a date.
    """
    # The content of timestamp is updated every time Emacs saves the file.
    return timestamp[13:32]

#-------------------------------------------------------------------------------

def version_info (caller_modules=[], caller_programs=[]):
    """
    Return a string of version information.

    Arguments:
    caller_versions  list of information to put at the top of the left column
    """
    fmt = "%-15s %-22s"
    nope = "[not available]"
    # Do the caller's version information first.
    mtext = []
    for code, v in caller_modules:
        mtext += [fmt % (code, v)]

    # Append the versions of the Python modules we use.  Sadly, we cannot do
    # this in a neat loop.
    mtext += [fmt % ("EVE library:", version ())]
    mtext += [fmt % ("Python:", platform.python_version ())]

    try:
        from PIL import Image
        v = Image.__version__
    except:
        v = nope
    mtext += [fmt % ("Image:", v)]

    try:
        import numpy
        v = numpy.__version__
    except:
        v = nope
    mtext += [fmt % ("numpy:", v)]

    try:
        import matplotlib
        v = matplotlib.__version__
    except:
        v = nope
    mtext += [fmt % ("matplotlib:", v)]

    try:
        import scipy
        v = scipy.__version__
    except:
        v = nope
    mtext += [fmt % ("scipy:", v)]

    # Do the caller's version information first.
    ptext = []
    for code, v in caller_programs:
        ptext += [fmt % (code, v)]

    # Now do the external programs that we use.
    for task in sorted (PROGRAMS.keys ()):
        for prog, cmd in PROGRAMS[task]:
            if find_in_path (prog):
                v = prog
                break
            else:
                v = nope
        ptext += [fmt % (task + ":", v)]

    # Now construct the actual text to return.
    mlen = len (mtext)
    plen = len (ptext)
    text = ""
    for i in range (0, max (mlen, plen)):
        if i >= mlen:
            text += " " * 38
        else:
            text += mtext[i]
        
        if i >= plen:
            pass
        else:
            text += ptext[i]
        text += "\n"
    return text[:-1]

#-------------------------------------------------------------------------------
def write (xim, fn, bgr=False):
    """
    Write an image to a file, the format being determined by the
    filename extension.

    Arguments:
    im  image to be written
    fn  name of the file to be written, ending in:
        ".jpg" for JPEG format
        ".png" for PNG format
        ".pnm" or ".pgm" or ".ppm" for PBMPLUS format
    bgr  the channels are ordered B-G-R as in OpenCV (default: False)
    """
    # Make sure the channels are in the right order.
    im = xim.copy ()
    if bgr:
        b = xim[:,:,0]
        g = xim[:,:,1]
        r = xim[:,:,2]
        im[:,:,0] = r
        im[:,:,1] = g
        im[:,:,2] = b

    # Determine the output format from the filename and invoke the appropriate
    # routine to do the saving.
    extn = fn[-3:]
    if   extn == 'jpg': write_pil (im, fn, 'JPEG')
    elif extn == 'png': write_pil (im, fn, 'PNG')
    elif extn == 'bmp': write_pil (im, fn, 'BMP')
    elif extn == 'pnm': write_pnm (im, fn)
    elif extn == 'pgm': write_pnm (im, fn)
    elif extn == 'ppm': write_pnm (im, fn)
    else:
        ValueError, 'Unsupported file extension'

#-------------------------------------------------------------------------------
def write_bmp (im, fn):
    """
    Write an image to a file in BMP format.

    Arguments:
    im  image to be written
    fn  name of the file to be written
    """
    write_pil (im, fn, 'BMP')

#-------------------------------------------------------------------------------
def write_jpeg (im, fn):
    """
    Write an image to a file in JPEG format.

    Arguments:
    im  image to be written
    fn  name of the file to be written
    """
    write_pil (im, fn, 'JPEG')

#-------------------------------------------------------------------------------
def write_jpg (im, fn):
    """
    Write an image to a file in JPEG format.

    Arguments:
    im  image to be written
    fn  name of the file to be written
    """
    write_pil (im, fn, 'JPEG')

#-------------------------------------------------------------------------------
def write_pil (im, fn, format='PNG'):
    """
    Write an image to a file using PIL.

    Arguments:
       im  image to be written
       fn  name of the file to be written
    format  the format of the file to be written (default: 'PNG')
    """
    from PIL import Image
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    bim = im.astype ('B')
    if nc == 3:
        pilImage = Image.fromarray (bim, 'RGB')
    elif nc == 4:
        pilImage = Image.fromarray (bim, 'RGBA')
    else:
        pilImage = Image.fromarray (bim[:,:,0], 'L')
    if format == 'display': pilImage.show ()
    else: pilImage.save (fn, format)

#-------------------------------------------------------------------------------
def write_png (im, fn):
    """
    Write an image to a file in PNG format.

    Arguments:
    im  image to be written
    fn  name of the file to be written
    """
    write_pil (im, fn, 'PNG')

#-------------------------------------------------------------------------------
def write_pnm (im, fn, binary=True, stretch=False, biggreys=False):
    """
    Write an image in PBMPLUS format to a file or stdout.

    Arguments:
          im  image to be written
          fn  name of the file to be written
      binary  if True, output binary, rather than text, data
              (default: True)
     stretch  if True, contrast-stretch the image during output
              (default: False)
    biggreys  if True, output 16-bit pixels (default: False)
    """

    # First, make sure we know the range of the data we are to output and
    # work out the necessary scaling factor.
    if biggreys: opmax = 62235; fmt = "%6d"
    else:        opmax = max_image_value; fmt = "%4d"
    if stretch:
        lo, hi = extrema (im)
        opmin = 0
        fac = opmax / (hi - lo)

    im = reshape3 (im)
    ny, nx, nc = sizes (im)

    # We have to do things differently if we're writing out a binary or textual
    # version of the image; the binary one is faster to do and much smaller.
    if binary:
        # Decide on the identifier that goes into the header.
        if nc == 1:
            pbmtype = "P5"
        elif nc == 3:
            pbmtype = "P6"
        else:
            pbmtype = "P? (%d channels as binary)" % nc
        # Open the output file.
        if fn == "-":
            f = sys.stdout
        else:
            f = open (fn, "wb")
        # Write the header.
        f.write ((pbmtype + "\n%d %d\n%d\n" % (nx, ny, opmax)).encode())
        # Scale the data if necessary and write them out.
        temp = im
        if stretch: temp = (temp - lo) * fac
        # Ensure the image is in C-contiguous order for easy writing.
        temp = numpy.asarray(temp, order="C")
        f.write (temp.astype("B"))

    else:
        # Decide on the identifier that goes into the header.
        if nc == 1:
            pbmtype = "P2"
        elif nc == 3:
            pbmtype = "P3"
        else:
            pbmtype = "P? (%d channels as ASCII)" % nc
        # Open the output file.
        if fn == "-":
            f = sys.stdout
        else:
            f = open (fn, "w")
        # Write the header.
        f.write (pbmtype + "\n%d %d\n%d\n" % (nx, ny, opmax))
        # Scale the data if necessary and write them out.
        for y in range (0, ny):
            for x in range (0, nx):
                for c in range (0, nc):
                    if stretch:
                        v = (im[y,x,c] - lo) * fac
                        if v > opmax: v = opmax
                        if v < opmin: v = opmin
                    else:
                        v = im[y,x,c]
                    if binary:
                        byte = struct.pack ("B", v)
                        f.write (byte)
                    else:
                        f.write (fmt % v)
        f.write ("\n")

    # Close the file.
    if fn != "-": f.close ()

#-------------------------------------------------------------------------------
def zero (im):
    """
    Set all pixels of an image to zero.

    Arguments:
    im  image to be zeroed (modified)
    """
    set (im, 0.0)

#-------------------------------------------------------------------------------
# Internal routines.
#-------------------------------------------------------------------------------
def _pnm_header (content, header_size):
    """
    Internal routine used by `read_pnm` to parse the header (containing
    `header_size` values) from binary PBMPLUS-format data in `content`.  We
    return the header and the next location in `content` to be read, which
    should be the image data themselves.
    """
    # We step forward from the start of the file, ignoring comments, until
    # we have read header_size whitespace-delimited values.  Note that the
    # end-of-line can be any of "^M" (ASCII 13, as found on old Macs),
    # "^J" (ASCII 10, as used under Unix) or both (as used in Windows, DOS
    # and OpenVMS if it every reappears).
    header = []
    loc = 0
    while len (header) < header_size:
        line = []
        comment = False
        while content[loc] != 13 and content[loc] != 10:
            if content[loc] == ord ("#"):
                comment = True
            line += chr (content[loc])
            loc += 1
        if not comment:
            line = "".join (line)
            header.extend (line.split ())
        if content[loc] == 10:
            loc += 1

    return header, loc

#-------------------------------------------------------------------------------
def _pnm_tokenize (content):
    """
    Internal routine used by `read_pnm` to convert the content that has been
    read from a file into text and then split the text up into "words".

    Arguments:
    content  content read directly from an image file
    """
    # Convert from binary to text.  We use ASCII here but one might argue that
    # UTF-8 might be better in the 21st century.  Split the result into lines.
    lines = str (content, "ascii").split ("\n")

    # Now tokenize the input lines.  Any word that starts with a "#" means the
    # rest of the line is a comment.
    tokens = []
    for line in lines:
        words = line.split ()
        for w in words:
            if w[0] == "#": break    # rest of line is a comment
            tokens += [w]

    return tokens

#-------------------------------------------------------------------------------
# Graphical output routines.
#-------------------------------------------------------------------------------
def graphics (what="default", quant=256):
    """
    Choose the graphics subsystem that EVE is to use.

    Arguments:
     what  one of "default", "sixel", "tty" (default: "default")
    quant  number of levels into which sixel graphics are quantised
           (default: 256)
    """
    global use_graphics, sixel_quant

    what = what.lower ()
    if what == "default":
        use_graphics = "default"
    elif what == "sixel":
        use_graphics = "sixel"
        sixel_quant = quant
    elif what == "tty" or what == "terminal" or what == "vdu":
        use_graphics = "tty"
    elif what == "lp":
        use_graphics = "lp"
    else:
        m = "I don't known what '%s' graphics are, so using default"
        print (m % what, file=sys.stderr)
        use_graphics = "default"
    #print ("[%s graphics selected]" % use_graphics, file=sys.stderr)

#-------------------------------------------------------------------------------
def select_graphics_type ():
    "Determine the type of graphical output tho generate."
    key =  "EVE_GRAPHICS"
    if key in os.environ:
        val = os.environ["EVE_GRAPHICS"].lower ()
        if val == "sixel16":
            graphics ("sixel", 16)
        elif val == "sixel256":
            graphics ("sixel", 256)
        else:
            graphics (val)
    else:
        graphics ("default")
        
#-------------------------------------------------------------------------------
def display (im, stretch=False, bgr=False, name="Eve image", wait=False):
    """
    Display an image using whatever graphics subsystem has been chosen.

    Note that this routine does not provide the fine level of control of
    output that is available when calling a device-specific routine.

    Arguments:
         im  image to be displayed
    stretch  if True, displayed image will be contrast-stretched
        bgr  the channels are ordered B-G-R as in OpenCV (default: False)
       name  the name given to the window
       wait  if set (and it makes sense), wait for the user to view the image
             before returning
    """
    global use_graphics, sixel_quant

    # Ensure the graphics type is set.
    if use_graphics is None:
        select_graphics_type ()

    if use_graphics == "default":
        display_external (im, stretch=stretch, bgr=bgr, name=name, wait=wait)
    elif use_graphics == "sixel":
        display_sixel (im, stretch=stretch, bgr=bgr, name=name)
    elif use_graphics == "tty":
        lppic (im, [" ./WX#"], width=80)
    elif use_graphics == "lp":
        lppic (im)
    else:
        print ("Internal error in display ('%s')!" % use_graphics,
               file=sys.stderr)
        sys.exit (99)

#-------------------------------------------------------------------------------
def graph (x, y, xlabel, ylabel, title="", logx=False, logy=False,
           style="linespoints"):
    """
Plot a graph using whatever graphics subsystem has been chosen.

    Note that this routine does not provide the fine level of control
    of output that is available when calling a device-specific
    routine.

    Arguments:
         x  a list of values to form the abscissa
         y  either a list of values to be plotted on the ordinate axis
            or a list of lists of values to be plotted as a series of
            separate curves
    xlabel  label to appear on the abscissa
    ylabel  label to appear on the ordinate axis
     title  the title of the graph (default: None)
      logx  if True, make the x-axis logarithmic (default: False)
      logy  if True, make the y-axis logarithmic (default: False)
     style  the method used to plot the data, a valid Gnuplot line-type
            or "histogram" (default: "linespoints")

    """
    global use_graphics

    # Ensure the graphics type is set.
    if use_graphics is None:
        select_graphics_type ()

    if use_graphics == "default":
        graph_gnuplot (x, y, xlabel, ylabel, title, logx=logx, logy=logy,
                       style=style)
    elif use_graphics == "sixel":
        graph_gnuplot (x, y, xlabel, ylabel, title, logx=logx, logy=logy,
                       terminal="sixel", wait=False, style=style)
        print ()
    elif use_graphics == "tty":
        lpgraph (x, y, xlabel, ylabel, title, logx=logx, logy=logy, style=style)
    else:
        print ("Internal error in graph ('%s')!" % use_graphics,
               file=sys.stderr)
        sys.exit (99)

#-------------------------------------------------------------------------------
def display_external (im, stretch=False, wait=False, bgr=False,
                      name="EVE image", program=None, hint=False):
    """
    Display an image using an external program.

    Arguments:
         im  image to be displayed
    stretch  if True, displayed image will be contrast-stretched
       wait  when True, allow the display program to exit before returning
        bgr  the channels are ordered B-G-R as in OpenCV (default: False)
       name  the name given to the window
    program  external program to be used for display
             (default: system-dependent)
       hint  output a line saying how to close the display (default: False)
    """
    # Make sure the channels are in the right order.
    if bgr: im = swap_channels (im)

    # We do different things on different operating systems.
    if systype == 'Windows':
        if stretch:
            copy = im.copy()
            contrast_stretch (copy)
        else:
            copy = im
        write_pil (copy, '', 'display')    # temporary kludge
        return

        # According to the website (spread over two lines of comment here):
        #   http://www.velocityreviews.com/forums/
        #     t707158-python-pil-and-vista-windows-7-show-not-working.html
        # Windows Vista needs the following workaround for image display via
        # Pil to work properly:
        # Edit (e.g., with TextPad) the file
        #   C:\Python26\lib\site-packages\PIL\ImageShow.py
        # Around line 99, edit the existing line to include a ping command,
        # as follows (spread over two lines of comments here):
        #   return "start /wait %s && PING 127.0.0.1 -n 5
        #      > NUL && del /f %s" % (file, file)
        if program is None: program = 'mspaint'
        handle, fn = tempfile.mkstemp (suffix='.bmp')
        write_bmp (copy, fn)
        if hint:
            print ('Type "Control-q" in the image window to close it.',
                   file=sys.stderr)
        line = "%s %s && ping 127.0.0.1 -n 15 > NUL && del /f %s" % \
               (program, fn, fn)
        if wait:
            line = 'start /wait ' + line
        else:
            line = 'start /wait' + line   # must wait on Windows, I think
        os.system (line)
    else:
        # Phew!  We're in the Unix world.
        if program is None:
            if find_in_path ('xv'):
                program = 'xv -name "' + name + '"'
                if hint:
                    print ('Type "q" in the image window to close it.',
                           file=sys.stderr)
            elif find_in_path ('display'):
                program = 'display'
                if hint:
                    print ('Type "q" in the image window to close it.',
                           file=sys.stderr)
            elif systype == 'Darwin':
                if stretch:
                    copy = im.copy()
                    contrast_stretch (copy)
                else:
                    copy = im
                handle, fn = tempfile.mkstemp (suffix='.png')
                write_png (copy, fn)
                if hint:
                    print ('Type "Command-q" in the image window to close it.',
                           file=sys.stderr)
                line = "%s '%s'; sleep 5; rm -f '%s'"
                if not wait:
                    line = "(" + line + ")&"
                line = line % ("open -a /Applications/Preview.app", fn, fn)
                os.system (line)
                return
            else:
                raise ValueError ('Cannot find an image display program')
        handle, fn = tempfile.mkstemp ()
        write_pnm (im, fn, stretch=stretch)
        if wait:
            line = "%s %s; rm -f %s"    % (program, fn, fn)
        else:
            line = "(%s %s; rm -f %s)&" % (program, fn, fn)
        os.system (line)
        os.close (handle)

#-------------------------------------------------------------------------------
def display_sixel (im, stretch=False, bgr=False, wait=True, name="Eve image",
                   levels=256):
    """Display `im` as sixels."""
    global PROGRAMS

    print (name + ":")
    for prog, inv in PROGRAMS["sixel display"]:
        if find_in_path (prog):
            display_external (im, stretch=stretch, bgr=bgr, wait=wait,
                              program=inv % levels)
            break
    print ()

#-------------------------------------------------------------------------------
def graph_gnuplot (x, y, xlabel='x', ylabel='y', title=' ', xlimits=None,
                   logx=False, logy=False, style='linespoints', key=None,
                   wait=True, terminal=None, fn=None):
    """
    Graph data using Gnuplot.

    Arguments:
         x  a list of values to form the abscissa
         y  either a list of values to be plotted on the ordinate axis
            or a list of lists of values to be plotted as a series of
            separate curves
     title  the title of the graph
    xlabel  the text used to annotate the abscissa
    ylabel  the text used to annotate the ordinate
   xlimits  if supplied, a list of [xlow, xhigh]
      logx  if True, make the x-axis logarithmic (default: False)
      logy  if True, make the y-axis logarithmic (default: False)
     style  the method used to plot the data, one of "lines", "linespoints"
            or "histogram" (default: "linespoints")
       key  if supplied, a list of the same length as the number of plots
            giving the name for each curve (default: None)
      wait  if True, allow the user to view the plot (and optionally save
            the data to file) before continuing (default: True)
        fn  if supplied, the name of a PDF file into which the graph is
            saved (in which case, wait is set to False)
    """
    # Work out and organise what we're plotting.
    if isinstance (y, numpy.ndarray):
        shape = y.shape
        if len (shape) == 1:
            y = y.reshape (1, shape[0])
        nplots, npoints = y.shape
    else:    
        nydims = 2 if isinstance (y[0], list) else 1
        if nydims == 1: y = [y]
        nplots = len (y)
        npoints = len (y[0])
        y = numpy.array (y)

    # If no x-values were given, provide some.
    if x is None:
        x = numpy.ndarray (npoints)
        for ix in range (0, npoints):
            x[ix] = ix
    else:
        x = numpy.array (x)

    # Start talking to Gnuplot and tell it how we'd like the graph to appear.
    if wait:
        p = os.popen ("gnuplot --persist", "w")
    else:
        p = os.popen ("gnuplot", "w")
    if terminal is not None:
        print ("set term " + terminal, file=p)
    if key is None:
        print ("set nokey", file=p)
    print ("set grid", file=p)
    print ('set title "%s"' % title, file=p)
    print ('set xlabel "%s"' % xlabel, file=p)
    print ('set ylabel "%s"' % ylabel, file=p)
    if xlimits: print ("set xrange [%f:%f]" % (xlimits[0], xlimits[1]), file=p)
    if logx: print ("set log x", file=p)
    if logy: print ("set log y", file=p)

    if fn is not None:
        print ("set term pdf", file=p)
        print ('set output "%s"' % fn, file=p)
        wait = False

    if style == "histogram":
        print ("set style data histogram", file=p)
        print ("binwidth=0.9", file=p)
        print ("set boxwidth binwidth", file=p)
        print ("set style fill solid", file=p)
        extra = "with boxes"
    else:
        print ("set style data " + style, file=p)
        extra = ""

    # Produce the plot command.
    text = "plot "
    for dataset in range (0, nplots):
        text += '"-"'
        if not key is None:
            text += 'title "%s" ' % key[dataset]
        text += extra + ", "
    print (text[:-2], file=p)

    # Output the points to plot.
    for dataset in range (0, nplots):
        for i in range (0, npoints):
            print (x[i], y[dataset,i], file=p)
        print ("e", file=p)
    p.flush () 

    """
    # Exit if the user types <EOF>; give (minimal) instructions if they type
    # "?"; simply continue if they type <return>.  Anything else typed in
    # response to the prompt is assumed to be a filename and we save the data
    # that file. (And yes, that can result in silly filenames...)
    if wait:
        looping = True
    else:
        looping = False

    while looping:
        sys.stderr.write ("CR> ")
        sys.stderr.flush()
        fn = sys.stdin.readline()
        if len(fn) < 1:
            print ("Exiting...", file=sys.stderr)
            sys.exit (1)
        if len(fn) > 0 and fn == "?\n":
            print ('fn to save data to "fn" else <return>', file=sys.stderr)
            continue
        if len(fn) > 1 and fn != "":
            f = open (fn[:-1],"w")
            nx = len (x)
            for i in range (0, nx):
                print (x[i], file=f, end=" ")
                for dataset in range (0, nplots):
                    print (y[dataset,i], file=f, end=" ")
                print (file=f)
            f.close ()
        looping = False
    """
    p.close ()

#-------------------------------------------------------------------------------
def graph_matplotlib (x, y, xlabel='x', ylabel='y', title=' ', logx=False,
                      logy=False, style="linespoints", key=None):
    """
    Graph data using Matplotlib.

    Arguments:
         x  a list of values to form the abscissa
         y  either a list of values to be plotted on the ordinate axis
            or a list of lists of values to be plotted as a series of
            separate curves
     title  the title of the graph
    xlabel  the text used to annotate the abscissa
    ylabel  the text used to annotate the ordinate
      logx  if True, make the x-axis logarithmic (default: False)
      logy  if True, make the y-axis logarithmic (default: False)
     style  the method used to plot the data, one of "lines", "linespoints"
            or "histogram" (default: "linespoints")
       key  if supplied, a list of the same length as the number of plots
            giving the name for each curve (default: None)
    """
    import pylab as p

    # Possible line styles.
    linestyles = {
        "lines": "-",
        "linespoints": "-o",
    }

    # Work out whether we're plotting one or several lines.
    nydims = 2 if isinstance (y[0], list) else 1
    if nydims == 1: y = [y]
    nplots = len (y)
    ny = len (y[0])

    # If no x-values were given, provide some.
    if x is None:
        x = []
        for ix in range (0, ny):
            x += [ix]

    # We want everything to be numpy arrays.
    x = numpy.array (x)
    y = numpy.array (y)

    # Set up pylab.
    p.figure ()
    p.grid ()
    p.title (title)
    p.xlabel (xlabel)
    p.ylabel (ylabel)

    # Set the plotting style of a conventional histogram.
    if style in linestyles:
        ls = linestyles[style]
    else:
        ls = "-o"

    # Plot the data.
    for dataset in range (0, nplots):
        if key is not None:
            lab = key[dataset]
        else:
            lab = None
        if style == "histogram":
            p.bar  (x, y[dataset], label=lab, align='center')
        else:
            p.plot (x, y[dataset], ls, label=lab)
    if not key is None: p.legend ()
    p.show ()

#-------------------------------------------------------------------------------
def graph_pgfplots (x, y, fn, xlabel='x', ylabel='y', title=' ', logx=False,
                   logy=False, style='linespoints', key=None, preamble=True):
    """
    Graph data using LaTeX's PGFplot system.

    Arguments:
         x  a list of values to form the abscissa
         y  either a list of values to be plotted on the ordinate axis
            or a list of lists of values to be plotted as a series of
            separate curves
        fn  the name of the file to receive the LaTeX commands for
            the plot
     title  the title of the graph
    xlabel  the text used to annotate the abscissa
    ylabel  the text used to annotate the ordinate
      logx  if True, make the x-axis logarithmic (default: False)
      logy  if True, make the y-axis logarithmic (default: False)
     style  the method used to plot the data, one of "lines",
            "linespoints" or "histogram" (default: "linespoints")
       key  if supplied, a list of the same length as the number of plots
            giving the name for each curve (default: None)
  preamble  if True, write out the document preamble at the top of the `fn`
    """
    # Work out whether we're plotting one or several lines.
    nydims = 2 if isinstance (y[0], list) else 1
    if nydims == 1: y = [y]
    nplots = len (y)
    ny = len (y[0])

    # If no x-values were given, provide some.
    if x is None:
        x = []
        for ix in range (0, ny):
            x += [ix]

    # We want everything to be numpy arrays.
    x = numpy.array (x)
    y = numpy.array (y)

    # Open the file and write out the preamble.
    f = open (fn, "w")
    if preamble: print (r"""
%\usepackage{pgfplots}  % <-- in the document preamble
\pgfplotsset{compat=newest}
\pgfplotsset{eve/.style={
    y tick label style={
      /pgf/number format/.cd,
      fixed,
      fixed zerofill,
      precision=1,
      /tikz/.cd
    },
    x tick label style={
      /pgf/number format/.cd,
      fixed,
      fixed zerofill,
      precision=1,
      /tikz/.cd
    },
    tick label style = {font=\sffamily\small},
    every axis label = {font=\sffamily\small},
    legend style = {font=\sffamily},
    label style = {font=\sffamily\small}}
}""", file=f)

    # Work out the line style.
    mark = "none" if style == "lines" else "*"

    # Work out the axis type and write out the beginning of the plot.
    if logx and logy: axis = "loglogaxis"
    elif logx: axis = "semilogxaxis"
    elif logx: axis = "semilogyaxis"
    else: axis = "axis"
    print (r"\begin{figure}", file=f)
    print (r"  \begin{center}", file=f)
    print (r"    \begin{tikzpicture}", file=f)
    print (r"      \begin{%s}[eve, xlabel=%s, ylabel=%s," % \
           (axis, xlabel, ylabel), file=f)
    print (r"         width=0.8\textwidth, height=0.45\textheight]", file=f)

    # Output the data to be plotted.
    for dataset in range (0, nplots):
        if style == "histogram":
            print (r"         \addplot[ybar interval] coordinates {", file=f)
        else:
            print (r"         \addplot[mark=%s] coordinates {" % mark, file=f)
        for i in range (0, ny):
            print ('          (%f, %f)' % (x[i], y[dataset,i]), file=f)
        print ('        };', file=f)

    # Finish the plot off.
    print (r"      \end{%s}" % axis, file=f)
    print (r"""    \end{tikzpicture}
  \end{center}
  \caption{%s}
  \label{fig:%s}
\end{figure}""" % (title, title), file=f)
    f.close()

#-------------------------------------------------------------------------------
def lpaxis (vmin, vmax, nincs):
    "Calculate information to use on the axis of a line printer plot."
    preferred = [1, 1.2, 1.5, 1.8, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10]
    npref = len (preferred)

    # Make an initial guess of step, reducing it into the range of
    # preferred values.
    step = (vmax - vmin) / nincs
    order = int (math.log10 (step))
    if step < 1.0:
        order -= 1
    step = step / 10.0**order

    # Find the smallest value in preferred greater than step.
    for p in range (0, npref):
        if preferred[p] > step:
            break

    while True:
        # Regenerate step.
        step = preferred[p] * 10**order

        # Set vlow to the next multiple of step below vmin.  This is the
        # smallest value which will appear on the axis, and we return it.
        offset = 0.04 if vmin >= 0 else -0.96
        vlow = int (vmin/step + offset) * step

        # Check that vmax lies in the range of values.
        if (vmax - vlow) / nincs <= step:
            vv = 0.0 if vmin < 0 and vmax > 0 else vlow
            origin = int ((vv - vmin) / step)
            break

        # It didn't fit so increase step and try again.
        p += 1
        if p >= npref:
            p = 0
            order += 1
    
    return origin, step, vlow

#-------------------------------------------------------------------------------
def lpgraph (x, y, xlabel, ylabel, title="", width=80, height=23,
             fd=sys.stdout, logx=False, logy=False, style="linespoints",
             points="X*AXCDEFGHIJKLMNPQRSTUVWXYZ"):
    """Plot a graph on a line printer (as "ASCII art").

    Arguments:
         x  a list of values to form the abscissa
         y  either a list of values to be plotted on the ordinate axis
            or a list of lists of values to be plotted as a series of
            separate curves
    xlabel  label to appear on the abscissa
    ylabel  label to appear on the ordinate axis
     title  the title of the graph (default: None)
     width  number of columns in the output (default: 80)
    height  number of lines in the output (default: 23)
        fd  output file stream
      logx  if True, make the x-axis logarithmic (default: False)
      logy  if True, make the y-axis logarithmic (default: False)
     style  the method used to plot the data, a valid Gnuplot line-type
            or "histogram" (default: "linespoints")
    points  plotting points (default: "X*AXCDEFGHIJKLMNPQRSTUVWXYZ")
    """
    # Set-up.
    nowid = 9
    fmt = "%" + "%d.2g" % nowid

    # Work out whether we're plotting one or several lines.
    nydims = 2 if isinstance (y[0], list) else 1
    if nydims == 1: y = [y]
    nplots = len (y)
    ny = len (y[0])

    # If no x-values were given, provide some.
    if x is None:
        x = []
        for ix in range (0, ny):
            x += [ix]

    # Determine the extrema of the axes.
    xmin = x[0]
    xmax = x[0]
    for xx in x:
        if logx: xx = math.log10 (xx)
        if xx < xmin: xmin = xx
        if xx > xmax: xmax = xx

    ymin = y[0][0]
    ymax = y[0][0]
    for p in range (0, nplots):
        for i in range (0, ny):
            yy = y[p][i]
            if logy: yy = math.log10 (yy)
            if yy < ymin: ymin = yy
            if yy > ymax: ymax = yy

    # Work out the increments and origin positions on the axes.
    fstcol = nowid + 3
    lstrow = 3

    ncols = width - fstcol
    colyax, xinc, xmin = lpaxis (xmin, xmax, ncols)
    colyax += fstcol

    # Output any title across the top of the graph.
    if title != "" and title != " ":
        nb = (width - fstcol) - len (title)//2
        gap = " " * (nb // 2)
        print (gap + title, file=fd)
        height -= 1

    nlines = height - lstrow
    linxax, yinc, ymin = lpaxis (ymin, ymax, nlines)
    linxax += lstrow

    # Calculate those values that depend on the increments and axis positions.
    colyno = colyax - nowid
    colytl = colyno - 1
    ylen = min (len (ylabel), height)
    ytfst = height - (height-ylen) // 2
    ych = 0

    # We can now start to output the graph line by line.
    top = ymin + (nlines+1) * yinc
    for L in range (height, 0, -1):
        bot = top - yinc
        # Clear the output buffer.
        lpbuf = [" "] * width
        if L == linxax:
            for i in range (fstcol, width):
                lpbuf[i] = "-" if (i-colyax) % 10 else "+"
        elif linxax > lstrow or L > linxax:
            lpbuf[colyax] = "|"

        # Add the y-axis numbering and annotation, if required.
        if (L-linxax) % 5 == 0:
            lpbuf[colyax] = "+"
            yy = 10**bot if logy else bot
            number = fmt % yy
            for i in range (0, nowid):
                if number[i] != " ": lpbuf[i+colyno] = number[i]

        if L <= ytfst and ych < ylen:
            lpbuf[colytl] = ylabel[ych]
            ych += 1

        # Add the x-axis numbering and annotation, if required.
        if L == (linxax-1):
            for i in range (fstcol, width - (nowid//2)):
                if (i-colyax) % 10 == 0:
                    fch = i - nowid + 1
                    number = (i-fstcol)*xinc + xmin
                    xx = 10**number if logx else number
                    number = fmt % xx
                    for n in range (0, nowid):
                        lpbuf[fch+n] = number[n]
        elif L == (linxax-2):
            nch = min (len (xlabel), ncols)
            fch = (ncols - nch) // 2 + fstcol
            for n in range (0, nch):
                lpbuf[fch+n] = xlabel[n]

        # lpbuf now holds any axes, annotation and numbering.  Now scan
        # through all the y-values, inserting the relevant plotting point
        # into lpbuf if a value lines in the range [bot, top).
        if top >= ymin:
            for p in range (0, nplots):
                for i in range (0, ny):
                    val = y[p][i]
                    if logy: val = math.log10 (val)
                    if style == "histogram" and val >= top:
                        xx = math.log10 (x[i]) if logx else x[i]
                        nch = int (round ((xx - xmin)/xinc) + fstcol)
                        lpbuf[nch] = points[p]
                    elif val >= bot and val < top:
                        xx = math.log10 (x[i]) if logx else x[i]
                        nch = int (round ((xx - xmin)/xinc) + fstcol)
                        lpbuf[nch] = points[p]

        # Now output the line and prepare for the next one.
        line = "".join (lpbuf)
        print (line.rstrip (), file=fd)
        top = bot

#-------------------------------------------------------------------------------
def lppic (im, using=["*       ", "@#+-    ", "#XXXX/' "], fd=sys.stdout,
           ff=False, aspect_ratio=1.65, width=132, border="tblr",
           reverse=False, limits=None):
    """
    Output an image as characters, optionally with overprinting.

    Arguments:
              im  image to be printed
           using  list of characters, each defining one layer of
                  overprinting in order of increasing blackness
                  (default: ["*       ", "@#+-    ", "#8XXX/' "])
             fd   file on which the output is written (default: sys.stdout)
             ff   if True, output a form-feed before printing the image
                  (default: false)
    aspect_ratio  ratio of character height to width (default: 1.65, which
                  matches that of Courier)
           width  number of characters in each line of output (default: 132)
          border  how the image should have its border drawn, a string of
                  characters from 'news' or 'tblr'
         reverse  if True, reverse the contrast (default: False)
          limits  if supplied, a list comprising the minimum and maximum
                  image values that should be used for determining the
                  mapping onto characters; image values outside this are
                  clipped
    """
    # If using is a string, we don't overprint.  A good set of characters in
    # that case is using="#@X+/' " for terminal windows with a light
    # background.  If using is a list, we assume it's a list of strings,
    # giving the different levels of overprinting.  In that case, the default
    # set looks OK.
    chars = []
    if isinstance (using, str):
        nover = 1
        chars.append(using)
    elif isinstance (using, list):
        chars = using
        nover = len(using)
    else:
        raise ValueError ('Illegal argument type')
    nvals = len (chars[0])

    # Work out how many pixels we're to print across the output.
    im = reshape3 (im)
    ny, nx, nc = sizes (im)
    if nx < width:
        xmax = nx
    else:
        xmax = width - 2
    xinc = nx / xmax
    yinc = xinc * aspect_ratio
    ymax = int (ny / yinc + 0.5)

    # Work out the grey-level scaling factor.
    if limits is None:
        lo, hi = extrema (im)
    else:
        lo, hi = limits
    if hi == lo:
        hi += 1
    fac = (nvals - 1) / (hi - lo)

    # Decide which borders we're to print.  If we're to print the top or
    # bottom border, work it out.
    doN = doE = doS = doW = False
    if border.find ('n') >= 0 or border.find ('t') >= 0:
        doN = True
    if border.find ('e') >= 0 or border.find ('r') >= 0:
        doE = True
    if border.find ('s') >= 0 or border.find ('b') >= 0:
        doS = True
    if border.find ('w') >= 0 or border.find ('l') >= 0:
        doW = True
    if doN or doS:
        sep = '+'
        for x in range (0, xmax):
            if x % 5 == 4:
                sep += '+'
            else:
                sep += '-'
        sep += '+'

    # Arrange to print a form-feed if necessary, and print the top border.
    if ff:
        ffc = ''
    else:
        ffc = ''
    if doN:
        print (ffc + sep, file=fd)
        ffc = ''

    # Print the image.
    buf = numpy.zeros (xmax, int)
    fy = 0
    for y in range (0, ymax):
        dy = fy - int(fy)
        dy1 = 1.0 - dy
        ylo = int(fy) % ny
        yhi = (ylo + 1) % ny
        ib = -1
        # For each pixel along a line, average all the channels to one and
        # scale the value into the appropriate number of levels.
        fx = 0
        for x in range (0, xmax):
            dx = fx - int(fx)
            dx1 = 1.0 - dx
            xlo = int(fx) % nx
            xhi = (xlo + 1) % nx
            v = 0
            for c in range (0, nc):
                vc = dx1 * dy1 * im[ylo,xlo,c] + \
                     dx  * dy1 * im[ylo,xhi,c] + \
                     dx1 * dy  * im[yhi,xlo,c] + \
                     dx  * dy  * im[yhi,xhi,c]
                v += vc
            v /= nc
            v = int ((v - lo) * fac + 0.5)
            if v < 0:      v = 0
            if v >= nvals: v = nvals - 1
            ib += 1
            buf[ib] = v
            fx += xinc
        fy += yinc

        # Print the line, including the borders if appropriate.  The traditional
        # way of doing this is as a series of lines using the carriage return
        # character '\r' so that subsequent lines over-print the first; but '\r'
        # is the end-of-line delimiter on Macintoshes, which confuses things.
        # So we have to backspace after each character in order to over-print.
        # And so technology moves on...
        if doW or doE:
            if y % 5 == 4: mark = '+'
            else:          mark = '|'
        else:              mark = ' '
        line = ' '
        if doW: line = mark
        for x in range (0, len(buf)):
            for ov in range (0, nover-1):
                line += chars[ov][buf[x]] + '\b'
            line += chars[nover-1][buf[x]]
        if doE: line += mark
        print (ffc + line, file=fd)
        ffc = ''

    # Print the bottom border.
    if doS: print (sep, file=fd)


#-------------------------------------------------------------------------------
# Main program
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    text = \
"""
There is a separate test script to check that EVE works correctly on your
platform, available from:
   http://vase.essex.ac/uk/software/eve/
""".strip ()

timestamp = "Time-stamp: <2020-10-05 16:49:14 Adrian F Clark (alien@essex.ac.uk)>"

# Local Variables:
# time-stamp-line-limit: -10
# End:
#-------------------------------------------------------------------------------
# End of EVE
#-------------------------------------------------------------------------------
