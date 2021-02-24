import sys
import cv2
import numpy as np
import scipy as sp
from scipy import ndimage as si
import typing
from typing import Tuple, List

"""
opencv version 4.3.0

cam info:
    focal length: 12m
    dist (baseline): 3500m
    pixel spacing: 10 microns = 1*10e-5m =  0.00001m
        scale in radians: ps/focal
            1/1200000 = (8.33r*10e-7)
            radians -> arc = (180/pi) * 3600
            0.1718873385 micron scale
        scale in m: 1.718873385*10e-7
    680 * 480 img
    
depth = f m b/disparity

f[12m] = sqrt(640^2 + 480^2)/
"""

testing: bool = False

# order in which the masks of objects are returned
objectOrder: List[str] = \
    ["cyan", "red", "white", "blue", "green", "yellow", "orange"]

# HSV values to be used to extract the objects from the image.
min_cyan: Tuple[int, int, int] = (90, 0, 0)
max_cyan: Tuple[int, int, int] = (90, 255, 255)

min_red: Tuple[int, int, int] = (0, 1, 0)
max_red: Tuple[int, int, int] = (0, 255, 255)

min_white: Tuple[int, int, int] = (0, 0, 1)
max_white: Tuple[int, int, int] = (0, 0, 255)

min_blue: Tuple[int, int, int] = (120, 1, 1)
max_blue: Tuple[int, int, int] = (120, 255, 255)

min_green: Tuple[int, int, int] = (60, 1, 1)
max_green: Tuple[int, int, int] = (60, 255, 255)

min_yellow: Tuple[int, int, int] = (30, 1, 0)
max_yellow: Tuple[int, int, int] = (30, 255, 255)

min_orange: Tuple[int, int, int] = (20, 0, 0)
max_orange: Tuple[int, int, int] = (29, 255, 255)

kernel33: np.ndarray = np.ones((3, 3), np.uint8)


# gets the masks for each object in the given image.
def getObjectMasks(hsvIn: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray,
                                               np.ndarray]:
    """
    Generates the object masks for a given HSV image.
    :param hsvIn: the hsv image that contains the objects
    :return: masks for each of the 7 coloured objects that are in it.
    (cyan, red, white, blue, green, yellow, orange)
    """
    cyan_mask: np.ndarray = cv2.inRange(hsvIn, min_cyan, max_cyan)
    cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel33)
    red_mask: np.ndarray = cv2.inRange(hsvIn, min_red, max_red)
    white_mask: np.ndarray = cv2.inRange(hsvIn, min_white, max_white)
    white_mask = cv2.subtract(white_mask, cyan_mask)
    blue_mask: np.ndarray = cv2.inRange(hsvIn, min_blue, max_blue)
    green_mask: np.ndarray = cv2.inRange(hsvIn, min_green, max_green)
    yellow_mask: np.ndarray = cv2.inRange(hsvIn, min_yellow, max_yellow)
    orange_mask: np.ndarray = cv2.inRange(hsvIn, min_orange, max_orange)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel33)

    yellow_mask = cv2.subtract(yellow_mask, orange_mask)

    return (cyan_mask, red_mask, white_mask, blue_mask,
            green_mask, yellow_mask, orange_mask)


def getStereoMasks(left_in: np.ndarray, right_in: np.ndarray) -> \
        List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates a list of masks for the left and right stereo images
    :param left_in: the left image.
    :param right_in: the right image.
    :return: a list of tuples containing the masks for each of the images.
    Tuples are in the form (left, right).
    The list itself is in the order
    cyan->red->white->blue->green->yellow->orange.
    """
    leftMasks = getObjectMasks(cv2.cvtColor(left_in, cv2.COLOR_BGR2HSV))
    rightMasks = getObjectMasks(cv2.cvtColor(right_in, cv2.COLOR_BGR2HSV))

    theMasks: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(0, 7):
        theMasks.append((leftMasks[i], rightMasks[i]))

    return theMasks


def showAllMasksForTesting(i_left: np.ndarray, i_right:np.ndarray, f: int) -> \
        None:
    """
    This is here mostly for testing purposes, showing the masks of each image
    :param i_left: left image
    :param i_right: right image
    :param f: frame number
    :return: nothing.
    """
    i = 0
    testMasks = getStereoMasks(i_left, i_right)
    for m in testMasks:
        showLeft = m[0]
        label = str(objectOrder[i]) + " " + str(f)
        print(label)
        cv2.putText(showLeft, label, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
        cv2.imshow("left", showLeft)
        cv2.imshow("right", m[1])
        cv2.waitKey(0)

        showLeft = cv2.bitwise_and(i_left, i_left, mask=m[0])
        showRight = cv2.bitwise_and(i_right, i_right, mask=m[1])
        cv2.putText(showLeft, label, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow("left", showLeft)
        cv2.imshow("right", showRight)
        cv2.waitKey(0)
        i += 1

params : cv2.SimpleBlobDetector_Params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 1
params.maxThreshold = 255
params.filterByArea = False
params.filterByColor = False
params.filterByCircularity = False
params.filterByInertia = False
params.filterByInertia = False
params.minArea = 1
params.minDistBetweenBlobs = 0

blobby : cv2.SimpleBlobDetector = cv2.SimpleBlobDetector_create(params)


def getKeypoints(leftMask: np.ndarray,
                 rightMask: np.ndarray) -> Tuple[cv2.KeyPoint, cv2.KeyPoint]:
    leftDilated = cv2.blur(leftMask, kernel33)
    leftKP = blobby.detect(leftDilated)






class FrameData:
    masks_cyan: Tuple[np.ndarray, np.ndarray]
    kp_cyan: Tuple[cv2.KeyPoint, cv2.KeyPoint]
    dist_cyan: float
    masks_red: Tuple[np.ndarray, np.ndarray]
    kp_red: Tuple[cv2.KeyPoint, cv2.KeyPoint]
    dist_red: float
    masks_white: Tuple[np.ndarray, np.ndarray]
    kp_white: Tuple[cv2.KeyPoint, cv2.KeyPoint]
    dist_white: float
    masks_blue: Tuple[np.ndarray, np.ndarray]
    kp_blue: Tuple[cv2.KeyPoint, cv2.KeyPoint]
    dist_blue: float
    masks_green: Tuple[np.ndarray, np.ndarray]
    kp_green: Tuple[cv2.KeyPoint, cv2.KeyPoint]
    dist_green: float
    masks_yellow : Tuple[np.ndarray, np.ndarray]
    kp_yellow: Tuple[cv2.KeyPoint, cv2.KeyPoint]
    dist_yellow: float
    masks_orange = Tuple[np.ndarray, np.ndarray]
    kp_orange: Tuple[cv2.KeyPoint, cv2.KeyPoint]
    dist_orange: float
    placeholderOutString: str = "{:5D} {:8} {:8.2E}"

    def __init__(self, frameNumber: int, left: np.ndarray, right: np.ndarray):
        self.frameNum: int = frameNumber
        self.left: np.ndarray = left
        self.right: np.ndarray = right
        self.has_cyan: bool = False
        self.has_red: bool = False
        self.has_white: bool = False
        self.has_blue: bool = False
        self.has_green: bool = False
        self.has_yellow: bool = False
        self.has_orange: bool = False

    def generateTheMasks(self):

        if testing:
            showAllMasksForTesting(self.left, self.right, self.frameNum)

        theMasks: List[Tuple[np.ndarray, np.ndarray]] =\
            getStereoMasks(self.left, self.right)

        if theMasks[0][0].any() or theMasks[0][1].any():
            self.has_cyan = True
            self.masks_cyan = theMasks[0]
        if theMasks[1][0].any() or theMasks[1][1].any():
            self.has_red = True
            self.masks_red = theMasks[1]
        if theMasks[2][0].any() or theMasks[2][1].any():
            self.has_white = True
            self.masks_white = theMasks[2]
        if theMasks[3][0].any() or theMasks[3][1].any():
            self.has_blue = True
            self.masks_blue = theMasks[3]
        if theMasks[4][0].any() or theMasks[4][1].any():
            self.has_green = True
            self.masks_green = theMasks[4]
        if theMasks[5][0].any() or theMasks[5][1].any():
            self.has_yellow = True
            self.masks_yellow = theMasks[5]
        if theMasks[6][0].any() or theMasks[6][1].any():
            self.has_orange = True
            self.masks_orange = theMasks[6]

    #def getKeypoints(self):


    def printIndividualFrameData(self):
        if self.has_cyan:
            print(self.placeholderOutString.format(self.frameNum, "cyan",
                                                   self.dist_cyan))
            if self.has_red:
                print(self.placeholderOutString.format(self.frameNum, "red",
                                                       self.dist_cyan))
            if self.has_white:
                print(self.placeholderOutString.format(self.frameNum, "white",
                                                       self.dist_white))
            if self.has_blue:
                print(self.placeholderOutString.format(self.frameNum, "blue",
                                                       self.dist_blue))
            if self.has_green:
                print(self.placeholderOutString.format(self.frameNum, "green",
                                                       self.dist_green))
            if self.has_yellow:
                print(self.placeholderOutString.format(self.frameNum, "yellow",
                                                       self.dist_yellow))
            if self.has_orange:
                print(self.placeholderOutString.format(self.frameNum, "orange",
                                                       self.dist_orange))



# print(cv2.__version__)

if len(sys.argv) < 3:
    print("Usage:", sys.argv[0],
          "<frame count> ",
          "<left-hand frame filename template> ",
          "<right-hand frame filename template>",
          file=sys.stderr)
    sys.exit(1)

print("deez nutz lmao gottem")

orb : cv2.ORB = cv2.ORB_create()




nframes: int = int(sys.argv[1])
for frame in range(0, nframes):
    fn_left = sys.argv[2] % frame
    im_left: np.ndarray = cv2.imread(fn_left)
    fn_right = sys.argv[3] % frame
    im_right: np.ndarray = cv2.imread(fn_right)

    if testing:
        print(fn_left)
        print(fn_right)
        cv2.imshow("left", im_left)
        cv2.imshow("right", im_right)
        cv2.waitKey(0)
        showAllMasksForTesting(im_left, im_right, frame)

    drawLeft = im_left
    drawRight = im_right
    masks = getStereoMasks(im_left, im_right)

    needToWaitAtTheEnd: bool = True
    for m in masks:
        needToWait = False
        needToWaitAtTheEnd = True
        maskedLeft = cv2.bitwise_and(im_left, im_left, mask = m[0])
        kpLeft = orb.detect(im_left, m[0])
        kpLeft, desLeft = orb.compute(im_left, kpLeft)
        print("left")

        whiteArea = []
        nx, ny = m[0].shape
        for x in range(0, nx):
            for y in range(0, ny):
                #print(m[0][x][y])
                if m[0][x][y] != 0:
                    whiteArea.append((x,y))
        print(whiteArea)
        # TODO: work out what I can actually do with this list of white pixels

        labelled, lCount = si.label(m[0])
        #print(labelled)
        #cv2.imshow("labelled", labelled)
        #cv2.waitKey(0)
        print(lCount)
        leftMaskBlurred = cv2.dilate(m[0], kernel33)

        #cv2.imshow("lBlurred", leftMaskBlurred)

        leftBlobby = blobby.detect(m[0])
        #leftBlobby = blobby.detect(leftMaskBlurred)

        #leftBlobby, desLeftBlobby = blobby.compute(im_left, leftBlobby)
        if len(leftBlobby) >= 1:
            print(leftBlobby[0].pt)
        else:
            print("no blobby")
            needToWait = True
        #print(leftBlobby)

        #print("desc")
        #print(desLeft)



        drawLeft = cv2.drawKeypoints(drawLeft, leftBlobby, None,
                                     color = (255,0,255), flags = 0)




        maskedRight = cv2.bitwise_and(im_right, im_right, mask=m[1])



        #kpRight = orb.detect(im_right, m[1])
        #kpRight, desRight = orb.compute(im_right, kpRight)

        rightMaskBlurred = cv2.dilate(m[1], kernel33, iterations=2)

        cv2.imshow("rBlurred", rightMaskBlurred)
        rightBlobby = blobby.detect(rightMaskBlurred)
        print("right")
        if len(rightBlobby) >= 1:
            print(rightBlobby[0].pt)
        else:
            print("no blobby")
            needToWait = True
        #rightBlobby, desRightBlobby = blobby.compute(im_right, rightBlobby)

        rightContours = cv2.findContours(m[1], cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        #drawRight = cv2.drawContours(drawRight, rightContours, 0,
         #                           (255, 0, 255), 1)
        #print(desRight)

        drawRight = cv2.drawKeypoints(drawRight, rightBlobby, None,
                                     color=(255, 0, 255), flags=0)
        cv2.imshow("left", drawLeft)
        cv2.imshow("right", drawRight)

        #disparityMap = stereo.compute(maskedLeft, maskedRight)
        #disparityMap = stereo.compute(im_left, im_right)
        #cv2.imshow("disparity map", disparityMap)
        if needToWait:
            cv2.waitKey(0)
            needToWaitAtTheEnd = False
    if needToWaitAtTheEnd:
        cv2.waitKey(0)

    """
    kpLeft = orb.detect(im_left)
    kpLeft, desLeft = orb.compute(im_left, kpLeft)

    print(desLeft)

    drawLeft = cv2.drawKeypoints(drawLeft, kpLeft, None,
                                 color=(255, 0, 255), flags=0)

    kpRight = orb.detect(im_right)
    kpRight, desRight = orb.compute(im_left, kpRight)

    print(desRight)

    drawRight = cv2.drawKeypoints(drawRight, kpRight, None,
                                  color=(255, 0, 255), flags=0)
    """
    cv2.imshow("left", drawLeft)
    cv2.imshow("right", drawRight)
    cv2.waitKey(0)



    #testMasks(im_left)




    #_, leftMask = cv2.threshold(leftGreyscale, 0, 255, cv2.THRESH_BINARY)

    #cv2.imshow("left greyscale", leftGreyscale)
    #cv2.waitKey(0)
    #cv2.imshow("left mask", leftMask)
    #acv2.waitKey(0)

    #leftHSV = cv2.cvtColor(im_left, cv2.COLOR_BGR2HSV)

    #left_m_cyan = cv2.inRange(leftHSV, min_cyan, max_cyan)
    #left_m_red = cv2.inRange(leftHSV, min_red, max_red)

    #print(type(leftHSV))

    #getObjectMasks(leftHSV)

    # print(type(leftHSV[0,0]))
    # print(leftHSV[0,0])

    #keypoints = detector.detect(im_left, mask=leftMask)
    # kp, des = detector.compute(im_left, mask = leftMask)

    #print(keypoints)
    # print(kp)
    # print(des)

    # TODO
    # we know background is exactly black
    # each object is a different colour
    # if spot the red one in the left,
    #   know that the red one in the right is the same
    #   look for leftmost, rightmost, highest,
    #   and lowest pixel that is red?
    # make each pixel above a certain value into a region,
    #   see what regions match?
    # can threshold a colour image, but most stuff so far is for greyscale
    #   make it greyscale
    #   find the regions from the greyscale image
    #   look at those regions in the colour image
    #   identify the colour
    #   ta-daa

print("sugma")



