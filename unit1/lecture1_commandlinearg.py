#!/usr/bin/env python3

import cv2, sys

im = cv2.imread(sys.argv[1])
cv2.imshow("nice", im)
cv2.waitKey(0)
print("done")