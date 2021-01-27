#!/usr/bin/env python3
"Summarize the content of images by showing their statistics and histogram"
import sys, cv2, numpy, pylab

#-------------------------------------------------------------------------------
# Routines.
#-------------------------------------------------------------------------------
def iround (x):
    "Convert x to the nearest whole number."
    return int (round (x))

def mean (im):
    "Calculate the mean of the image im."
    ny, nx, nc = im.shape
    total = 0
    for y in range (0, ny):
        for x in range (0, nx):
            for c in range (0, nc):
                total += im[y,x,c]
    return total / ny / nx / nc

def hist (im):
    "Return the grey-level histogram of an image, ready for plotting."
    # The maximum image value that we support.
    maxgrey = 256

    # Create the values for the abscissa (x-axis).
    ab = numpy.ndarray (maxgrey)
    for i in range (0, maxgrey):
        ab[i] = i

    # Create the histogram array and set it from the image.
    h = numpy.zeros (maxgrey)
    for y in range (0, ny):
        for x in range (0, nx):
            for c in range (0, nc):
                v = im[y,x,c]
                h[v] += 1

    # Return the x and y values.
    return ab, h

def plot_hist (x, y, fn):
    "Plot the histogram of image fn."
     # Set up pylab.
    pylab.figure ()
    pylab.xlim (0, 255)
    pylab.grid ()
    pylab.title ("Histogram of " + fn)
    pylab.xlabel ("grey level")
    pylab.ylabel ("number of occurrences")
    pylab.bar (x, y, align="center")
    pylab.show ()




#-------------------------------------------------------------------------------
# Main program.
#-------------------------------------------------------------------------------
# Set-up.
maxdisp = 800

# Ensure the command line is sensible.
if len (sys.argv) < 2:
    print ("Usage:", sys.argv[0], "<image>...", file=sys.stderr)
    sys.exit (1)

# Process the files given on the command line.
for fn in sys.argv[1:]:
    # Read in the image and print out its dimensions.
    im = cv2.imread (fn)
    ny, nx, nc = im.shape
    print (fn + ":")
    print ("  Dimensions:", nx, "pixels,", ny, "lines,", nc, "channels.")

    # Calculate and output some important statistics.
    print ("  Range: %d to %d" % (im.min (), im.max ()))
    print ("  Mean: %.2f (using mean)" % mean (im))
    print ("  Mean: %.2f (using numpy method)" % im.mean ())
    print ("  Standard deviation: %.2f" % im.std ())

    # Work out and display the histogram.
    x, h = hist (im)
    plot_hist (x, h, fn)

    # For display, ensure the image is no more than maxdisp pixels in x or y.
    if ny > maxdisp or nx > maxdisp:
        nmax = max (ny, nx)
        fac = maxdisp / nmax
        nny = iround (ny * fac)
        nnx = iround (nx * fac)
        print ("  [re-sizing to %d x %d pixels for display]" % (nnx, nny))
        im = cv2.resize (im, (nnx, nny))

    # Display the image.
    cv2.imshow (fn, im)
    cv2.waitKey (0)
    cv2.destroyWindow (fn)
    print ()

#-------------------------------------------------------------------------------
# End of summarize.
#-------------------------------------------------------------------------------