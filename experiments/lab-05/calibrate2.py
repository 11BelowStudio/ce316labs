#!/usr/bin/env python3
"Calibrate stereo cameras using POV images."
import sys, os, math, cv2
import numpy as np

# We are looking for 8 horizontal and 7 vertical lines.
objpoints = np.zeros ((8*7,3), np.float32)
objpoints[:,:2] = np.mgrid[0:8,0:7].T.reshape(-1,2)

# Termination criterion.
term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0)

# Arrays for real-world object points and the corresponding image locations.
points3d = []
points2d = []

# Loop over the calibration images.
for frame in range (0, 30):
    im = cv2.imread ("calib-%3.3d.png" % frame)
    grey = cv2.cvtColor (im, cv2.COLOR_BGR2GRAY)

    # Look for the corners inside the board.  Record any that we find for use
    # in calibrating the camera.
    found, corners = cv2.findChessboardCorners (grey, (8,7), None, 0)
    if found:
        points3d.append (objpoints)
        cv2.cornerSubPix (grey, corners, (11,11), (-1,-1), term_crit)
        points2d.append (corners)
        
        # Mark and display the corners.
        cv2.drawChessboardCorners (im, (8,7), corners, found)
        cv2.imshow ("Corners found", im)
        cv2.waitKey (125)

# Calibrate the camera using the points and image locations.
found, cam, D, R, T = cv2.calibrateCamera (points3d, points2d,
                         grey.shape[::-1], None, None)
            
# Take the mean of the x and y focal lengths as the overall one, and their
# difference as a (poor) estimate of the error.
fx = cam[0,0]
fy = cam[1,1]
F = (cam[0,0] + cam[1,1]) / 2
Ferr =  max (abs (F-fx), abs(F-fy))
print ("Focal length:", F, "+/-", Ferr)

# Z = (f * B) / D
# dist = (focal * baseline) / disparity

# disparity: x'L - x'R
#   x': x location - (total x/2)
#   y': y location - (total y/2)
#

# images both 640 * 480
# tip of nose (left):
#     x: 389 +/- 1
#       x' : 389 - 320 = 69 +/- 1
#     y: -229±1
# tip of nose (right)
#     x: 252±1
#       x' : 252 - 320 = -68 +/- 1
#     y:-229±1
# both y are equal, no y disparity.
# x disparity: x'L - x'R = 69 + 68 = 137

# rough estimate of camera stuff:
#     actual tip of nose at <0, 289, 280>
#     cams at (+=75,300,800)
#         baseline of 150
#     distance from cams approx 520
#         800 (cam Z) - 280 (nose Z) = 520
#         approx because cams are in different locations
#     disparity
#           left transformed x
#   520 = (focal * 150) / 137
#   disp * dist = (focal * 150)
#       137 * 520 = 71240
#       71240/150 = 474.93r approx focal length

# **factoring in error of +- 1/640**
#   B = (150 +- 0.1)mm
#   f = (477.35 +- 1.2)mm
#   xl = (69+-1)p
#   xr = (-68 +- 1)p
# suppose sensor = 10mm square
#   1 pixel = 10/640mm = 0.015625mm
#   xL (1.078125 +- 0.015625mm)
#   xR (-1.061250 +- 0.015625mm)
#
# error in Z (dist) = sum of errors in B, F, and disparity
# eZ/Z = (eB/B) + (eF/F) + ((exL + exR)/(xL-xR))
# (0.1/150) + (1.2/477.35) + (0.03125/2.140625) = 0.01777908552...
# allegedly means an error of 1.7mm
#   but I got 2.413007691 on my calculator.


# actual focal length:
#   484.65535083893656 +/- 1.3106203765603368
# lecture slides say 477.35

# using this to get actual dist
#   (150 * 484) / (69 + 68) = 522.6
print("")
def distChecker(origin, cam):

    x = (cam[0] - origin[0]) ** 2
    y = (cam[1] - origin[1]) ** 2
    z = (cam[2] - origin[2]) ** 2
    return math.sqrt(x + y + z)

noseTip = (0,289,280)
leftCam = (-75,300,800)
leftDist = distChecker(noseTip, leftCam)
rightCam = (75,300,800)
rightDist = distChecker(noseTip,rightCam)

meanDist = (leftDist + rightDist)/2
#print("l " + str(leftDist))
#print("r " + str(rightDist))
print("dist from actual coordinates: " + str(meanDist))
# 525.4959562166011 mm


print("")

print("Dist from the focal length and disparity and such")

eBase = 0.1
base = 150

dispPx = 69 + 68

pxSize = 10 # 10mm/pixel
imSize = 640 # 640px

eDisp = (pxSize * 2) / imSize
dispMM = (dispPx * pxSize) / imSize

dist = (F * base)/dispPx

print(dist) # 530.6445447141641

eDist = (Ferr/F) + (eBase/base) + (eDisp/dispMM)

print(eDist) # 0.017969438539985023

print("")
print("distance:", dist, "±", eDist, "mm")
# 530.6445447141641 ± 0.017969438539985023 mm
