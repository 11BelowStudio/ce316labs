#!/usr/bin/env python3

import cv2, sys, numpy, matplotlib.pyplot

# histogram: number of times each individual picture value occurs in an image


def my_histo(im):
    "Return the histogram of an image"
    ny, nx, nc = im.shape
    print(ny,nx,nc)

    # creates an array full of 0s for everything we see
    hist = numpy.zeros((256))
    for y in range(0,ny):
        for x in range(0,nx):
            for c in range (0,nc):
                v = im[y,x,c] # cycles through the 3 channels, puts it into the histogram
                hist[v] = hist[v] + 1 # adds 1, keeping count of times the grey level occurs
    return hist


# Read an image and store it as a numpy array
# Has a type, can extract information from it
#im = cv2.imread(sys.argv[1])
im = cv2.imread("img.png")

# calculate the histogram
hist = my_histo(im)

# print out what has been calculated
for i in range(0, 256):
    print(i, hist[i])

matplotlib.pyplot.plot(hist)



hist2 = cv2.calcHist([im],[0],None,[256],[0,256])

# print out what has been calculated
for i in range(0, 256):
    print(i, hist2[i])

matplotlib.pyplot.plot(hist2)



hist3, _ = numpy.histogram(im.ravel(), 256, [0,256])

# print out what has been calculated
for i in range(0, 256):
    print(i, hist3[i])

matplotlib.pyplot.plot(hist3)



