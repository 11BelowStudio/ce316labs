#!/usr/bin/env python3

import cv2

# has made opencv accessible.
# called cv2 because it was the 2nd attempt at a python version of opencv

print(cv2.__version__)

# we want to do the closest thing to a 'hello world' in opencv

# routine that reads an image
im = cv2.imread("img.png")

# routine that will show an image
# arg 1: window name
# arg 2: image
cv2.imshow("nice", im)

# shows the image, and waits for a key to be pressed (before then closing the image)
cv2.waitKey(0)
print("done")
