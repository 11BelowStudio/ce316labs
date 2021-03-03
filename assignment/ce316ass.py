import sys
import cv2
import numpy as np
import scipy as sp
from scipy import ndimage as si
import typing
from typing import Tuple, List, Dict

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




def handleShowingStuff() -> None:
    """
    This is here to handle actually showing stuff when testing the program.
    Prints something to console to let me know that I need to press something,
    handles doing the cv2.waitKey(0) call,
    and, if I press q or escape, it will close the program.
    :return: nothing is returned.
    """
    print("pls to press")
    key = cv2.waitKey(0)
    # quit if escape (27) or q (113) are pressed
    if key == 27 or key == 113:
        cv2.destroyAllWindows()
        print("quitting!")
        sys.exit(0)


testing: bool = False

# order in which the masks of objects are returned
objectOrder: List[str] = \
    ["cyan", "red", "white", "blue", "green", "yellow", "orange"]

# HSV values to be used to extract the objects from the image.
min_cyan: Tuple[int, int, int] = (90, 0, 0)
"""minimum HSV threshold for the cyan object"""
max_cyan: Tuple[int, int, int] = (90, 255, 255)
"""maximum HSV threshold for the cyan object"""

min_red: Tuple[int, int, int] = (0, 1, 0)
"""minimum HSV threshold for the red object"""
max_red: Tuple[int, int, int] = (0, 255, 255)
"""maximum HSV threshold for the red object"""

min_white: Tuple[int, int, int] = (0, 0, 1)
"""minimum HSV threshold for the white object"""
max_white: Tuple[int, int, int] = (0, 0, 255)
"""maximum HSV threshold for the white object"""

min_blue: Tuple[int, int, int] = (120, 1, 1)
"""minimum HSV threshold for the blue object"""
max_blue: Tuple[int, int, int] = (120, 255, 255)
"""maximum HSV threshold for the blue object"""

min_green: Tuple[int, int, int] = (60, 1, 1)
"""minimum HSV threshold for the green object"""
max_green: Tuple[int, int, int] = (60, 255, 255)
"""maximum HSV threshold for the green object"""

min_yellow: Tuple[int, int, int] = (30, 1, 0)
"""minimum HSV threshold for the yellow object"""
max_yellow: Tuple[int, int, int] = (30, 255, 255)
"""maximum HSV threshold for the yellow object"""

min_orange: Tuple[int, int, int] = (20, 0, 0)
"""minimum HSV threshold for the orange object"""
max_orange: Tuple[int, int, int] = (29, 255, 255)
"""maximum HSV threshold for the orange object"""

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



def getObjectMidpoint(objectMask: np.ndarray) -> Tuple[float, float]:
    """
    Returns the midpoint of the region of 1s in the given binary image array
    :param objectMask: binary image, with 1s in the area where the object
    with the midpoint being looked for is, and 0s everywhere else.
    :return: A tuple containing:
        * Tuple with the midpoint of the object (in middle of middle pixel)
    """
    if not objectMask.any():
        # if nothing in the objectMask is a 1, we return -1s.
        return (-1,-1)

    yx: Tuple[int,int] = objectMask.shape
    ny: int = yx[0]
    nx: int = yx[1]

    minX: int = -1
    maxX: int = -1
    minY: int = -1
    maxY: int = -1
    notFoundFirst: bool = True

    for y in range(0, ny):
        for x in range(0, nx):
            if objectMask[y][x] != 0:
                if notFoundFirst:
                    notFoundFirst = False
                    minX = maxX = x
                    minY = maxY = y
                else:
                    if x < minX:
                        minX = x
                    elif x > maxX:
                        maxX = x
                    maxY = y
                    # y wont get smaller.
                    # and assignment has same complexity as checking a single
                    # condition so I may as well just reassign y anyway.


    w: int = (maxX - minX)
    h: int = (maxY - minY)
    # the +1s are here because I'm considering the width and height to be
    # between the far edges of the minimum/maximum pixels,
    # to account for the error that may arise in the pixel stuff.
    #   pixel coords correspond to top-left corner of pixel,
    #   so I'm using the midpoints of the pixel instead.
    #
    # here's some examples.
    #   Outer numbers: pixel positions.
    #   inner: X = in area, 0: not in area
    #       * : where the midpoint should be.
    #           if on a pixel, it's at the midpoint of that pixel.
    #           if out of the grid, it's in that part of the grid.
    #
    # 0↴1↴|
    # ↳*|0|
    # 1-|-|
    # ↳0|0|
    # |-|-|
    #
    # Without the +1:
    #   max would be n+0, min would be n+0; 0-0 = 0.
    #   midpoint would be 0/2 = n+0
    # With the +1:
    #   max would be n+0+1, min would be n+0; 1-0 = 1.
    #   midpoint would be 1/2 = n+0.5
    #       right in the middle of that one pixel.
    #
    # 0↴1↴2↴3↴
    # ↳X|X|X|0|
    # 1-|-|-|-|
    # ↳X|*|X|0|
    # 2-|-|-|-|
    # ↳X|X|X|0|
    # 3-|-|-|-|
    # ↳0|0|0|0|
    # |-|-|-|-|
    #
    # Without the +1:
    #   max would be n+2, min would be n+0; 2-0 = 2.
    #   midpoint would be 2/2 = n+1
    #       on the left edge of 2nd column/right edge of 1st; not exact.
    # With the +1:
    #   max would be n+2+1, min would be n+0; 3-0 = 3.
    #   midpoint would be 3/2 = n+1.5
    #       right in the middle of the 2nd column; exact.
    #
    # 0↴1↴2↴3↴4↴
    # ↳X|X|X|X|0|
    # 1-|-|-|-|-|
    # ↳X|X|X|X|0|
    # 2-|-*-|-|-|
    # ↳X|X|X|X|0|
    # 3-|-|-|-|-|
    # ↳X|X|X|X|0|
    # 4-|-|-|-|-|
    # ↳0|0|0|0|0|
    # |-|-|-|-|-|
    #
    #
    # Without the +1:
    #   max would be n+3, min would be n+0; 3-0 = 3.
    #   midpoint would be 3/2 = n+1.5
    #       in the middle of the 2nd column pixels; not exact.
    # With the +1:
    #   max would be n+3+1, min would be n+0; 4-0 = 4.
    #   midpoint would be 4/2 = n+2
    #       left edge of 3rd column/right edge of 2nd; exact.

    xMid: float = minX + (w/2.0)
    yMid: float = minY + (h/2.0)

    return xMid, yMid



def getObjectMidpoints(hsvIn: np.ndarray) -> Dict[str, Tuple[float,float]]:
    """
    Gets the midpoints for the objects that may be in the coloured image.
    :param hsvIn: the hsv image that contains the objects
    :return: dict with midpoints for each of the 7 coloured objects
    that might be present in the given image.
    Keys are (cyan, red, white, blue, green, yellow, orange)
    """
    # generating masks for each object.
    cyan_mask: np.ndarray = cv2.inRange(hsvIn, min_cyan, max_cyan)
    red_mask: np.ndarray = cv2.inRange(hsvIn, min_red, max_red)
    white_mask: np.ndarray = cv2.inRange(hsvIn, min_white, max_white)
    blue_mask: np.ndarray = cv2.inRange(hsvIn, min_blue, max_blue)
    green_mask: np.ndarray = cv2.inRange(hsvIn, min_green, max_green)
    yellow_mask: np.ndarray = cv2.inRange(hsvIn, min_yellow, max_yellow)
    orange_mask: np.ndarray = cv2.inRange(hsvIn, min_orange, max_orange)

    # and now, time for some cleanup

    # filling in the midpoint for the cyan mask (as that's actually white)
    cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel33)

    # removing the midpoint in the cyan mask from the white mask
    white_mask = cv2.subtract(white_mask, cyan_mask)

    # filling in the midpoint for the orange mask (as that's actually yellow)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel33)

    # and removing the orange midpoint from some yellow
    yellow_mask = cv2.subtract(yellow_mask, orange_mask)

    # now I'm making the dictionary with the midpoints of each object
    posDict: Dict[str, Tuple[float,float]] = {
        "cyan":   getObjectMidpoint(cyan_mask),
        "red":    getObjectMidpoint(red_mask),
        "white":  getObjectMidpoint(white_mask),
        "blue":   getObjectMidpoint(blue_mask),
        "green":  getObjectMidpoint(green_mask),
        "yellow": getObjectMidpoint(yellow_mask),
        "orange": getObjectMidpoint(orange_mask)
    }

    return posDict


def getStereoPositions(left_in: np.ndarray, right_in: np.ndarray) -> \
    Dict[str,Tuple[Tuple[float, float],Tuple[float, float]]]:
    """
    Gets the positions of objects in the left and right *coloured* images.
    This **will** assume that the dimensions of the left and right images
    are identical, and that the two images are Y-aligned already.
    :param left_in: the left input image *in colour*
    :param right_in: the right input image *in colour*
    :return: a dictionary with the x and y positions of each of the objects
    in each image.

    1st item in the tuple: (xl,y') from left image.
    2nd item in the tuple: (xr,y') from right image.

    If an object is not present in **both** images, it will **not** be present
    in the returned dictionary.
    :raises: ValueError if the two images provided have different dimensions.
    """
    # gets the shape of the images
    yx: Tuple[int, int] = left_in.shape
    if yx != right_in.shape:
        # complains if the dimensions aren't identical.
        raise ValueError("Please provide images with identical shapes.")

    # gets midpoints for all the objects in each image.

    lDict: Dict[str, Tuple[float, float]] = getObjectMidpoints(left_in)
    """dictionary with midpoints for every object in the left image"""

    rDict: Dict[str, Tuple[float, float]] = getObjectMidpoints(right_in)
    """dictionary with midpoints for every object in the right image"""

    # half of the x and y dimensions of the images
    halfY: float = yx[0]/2
    halfX: float = yx[1]/2

    posDict: Dict[str, Tuple[Tuple[float, float], Tuple[float,float]]] = {}
    """a dictionary for all the calculated (X',Y') positions for each image"""

    # obtains the keys from lDict but as a list so it can be foreach'd
    leftKeys: List[str] = [*lDict.keys()]

    # now looks through each of those keys
    for k in leftKeys:
        if lDict[k] != (-1,-1):
            if rDict[k] != (-1,-1):
                # if both dictionaries have an actual value for the key

                # just getting a copy of those raw values real quick
                rawL: Tuple[float, float] = lDict[k]
                rawR: Tuple[float, float] = rDict[k]

                # work out the X' and the Y' stuff for left and right
                # and put it into the posDict.
                #   X` = x - halfX
                #   Y' = halfY - y
                posDict[k] = (
                    (rawL[0] - halfX, halfY - rawL[1]),
                    (rawR[0] - halfX, halfY - rawR[1])
                )
    # and now return the posDict
    return posDict




class FramePosData:
    """
    This class basically holds positional information for the objects
    in each frame.
    """

    placeholderOutString: str = "{:5}  {:8}  {:8.2e}  {}"
    """
    This is a placeholder string to be used when formatting the frame-by-frame
    printout of object positions.
    """

    focalLength: float = 12
    "Focal length of camera is 12m"
    baseline: float = 3500
    "baseline between cameras is 3.5km -> 3500m"
    pixelSize: float = float(1e-5)
    "pixel spacing: 10 microns -> 1e-5 metres"

    def __init__(self, frameNumber: int, left: np.ndarray, right: np.ndarray):
        """
        Constructor for the framePosData stuff
        :param frameNumber: what frame this is
        :param left: the left image in BGR colour
        :param right: the right image in BGR colour
        """
        self.frameNum: int = frameNumber

        """
        These are the left and right image X'Y' coords of every object
        in both the left and the right images.
        """
        imgPositions: \
            Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = \
            getStereoPositions(cv2.cvtColor(left, cv2.COLOR_BGR2HSV),
                               cv2.cvtColor(right, cv2.COLOR_BGR2HSV))

        """
        This will hold the depths of the objects
        """
        self.distances: Dict[str, float] = {}

        """
        this will hold the x' and y' co-ords of the objects
        """
        self.posXYZ: Dict[str, Tuple[float, float, float]] = {}


        objKeys: List[str] = [*imgPositions.keys()]
        # unpacking the keys/object names so we can iterate through them.
        for k in objKeys:
            # we obtain the info about current object from imagePositions
            currentPos: Tuple[Tuple[float, float], Tuple[float,float]] = \
                imgPositions[k]

            # x disparity = xl - xr
            xDisparity: float = currentPos[0][0] - currentPos[1][0]

            dist: float = ((FramePosData.focalLength * FramePosData.baseline) /
                           (xDisparity * FramePosData.pixelSize))
            """
            Z = (f * b) / (xl - xr)
            which is how we work out what dist is.
            """


            # putting the raw dist value into the distances dictionary
            self.distances[k] = dist

            rawX: float = ((-currentPos[1][0] * FramePosData.pixelSize) /
                           FramePosData.focalLength) * dist
            """
            (-xr/f) = (X/Z), therefore (-xr/f) * Z = X
            So I used that equation to find out what the actual X position
            of the object is in 3D space.
            """

            # just obtaining the midpoint of the Ys real quick
            yMid: float = (currentPos[0][1] + currentPos[1][1])/2

            rawY: float = -((yMid * FramePosData.pixelSize) /
                            FramePosData.focalLength) * dist
            """
            (xl / f) = (B-X)/Z 
            so, substituting the x for y
            (y / f) = (B-Y)/Z = -Y/Z
            and if we rearrange that so -Y is the result
            (y/f) * Z = -Y
            and negating that to get positive Y as the result
            -(y/f) * Z = Y
            """

            # we put the raw XYZ into posXYZ
            self.posXYZ[k] = (rawX, rawY, dist)

    def printFrameData(self) -> None:
        """
        This will be called when printing the data for the individual frames.
        Prints the depths for all the objects encountered.
        :return: nothing.
        """
        keys: List[str] = [*self.distances.keys()]

        for k in keys:
            print(FramePosData.placeholderOutString.format(
                self.frameNum,
                k,
                self.distances[k],
                self.posXYZ[k]
            ))




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
        handleShowingStuff()

        showLeft = cv2.bitwise_and(i_left, i_left, mask=m[0])
        showRight = cv2.bitwise_and(i_right, i_right, mask=m[1])
        cv2.putText(showLeft, label, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow("left", showLeft)
        cv2.imshow("right", showRight)
        handleShowingStuff()
        i += 1



if len(sys.argv) < 3:
    print("Usage:", sys.argv[0],
          "<frame count> ",
          "<left-hand frame filename template> ",
          "<right-hand frame filename template>",
          file=sys.stderr)
    sys.exit(1)

print("deez nutz lmao gottem")


frames: List[FramePosData] = []


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
        handleShowingStuff()
        showAllMasksForTesting(im_left, im_right, frame)

    frames.append(FramePosData(frame, im_left, im_right))

    print("next pls")

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

print("frame  identity  distance")
for f in frames:
    f.printFrameData()

# TODO
# work out trajectory of each object.
# I have XYZ pos of each object for each frame that they're there for,
# I just need to use them


print("sugma")



