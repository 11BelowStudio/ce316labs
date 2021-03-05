import sys
import cv2
import numpy as np
import scipy as sp
from scipy import ndimage as si
import typing
from typing import Tuple, List, Dict, Union
from math import sqrt, isclose, cos

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

"""
--DEBUGGING FUNCTIONS FOR SEEING HOW THINGS WORK--

You ever wanted to see what goes on under the hood with this program?
No?

Either way, here are some functions that can be used for debugging.
"""

debugging: bool = False
"""
Set this to 'True' if you want to enable these debug functions.
Or just keep it as 'False' if you want to just run the thing.
"""


def debug(leftFilename: str, rightFilename: str,
          leftIm: np.ndarray, rightIm: np.ndarray, frameNum: int = 0) -> None:
    """
    The wrapper function for the debugging functions, called by the main program
    if 'debug' is set to True.

    :param leftFilename: filename of the left image
    :param rightFilename: filename of the right image
    :param leftIm: the left image, BGR format
    :param rightIm: the right image, BGR format
    :param frameNum: What frame this is (used for labelling the mask
     previews). If not defined, defaults to 0.
    :return: nothing.
    """

    # prints the filenames, to make sure they're correct
    print(leftFilename)
    print(rightFilename)

    # shows the left and right images
    cv2.imshow("left", leftIm)
    cv2.imshow("right", rightIm)
    handleShowingStuff()
    showMasksForDebugging(leftIm, rightIm, frameNum)
    # and then proceeds to show all the masks produced for each object in
    # the left and the right images.


def getObjectMasks(hsvIn: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray,
                                               np.ndarray]:
    """
    Generates the object masks for a given HSV image.

    Yes, this uses some tuples that are defined as global variables in the
    'actually important code' section after this section of debugging code,
    because those tuples are part of the 'actually important code'.
    This has been done so, if this debugging code section is removed (along with
    the one branch in the main program that may potentially call this code),

    Either way, this function won't be run until after those declarations are
    reached, so no harm no foul.

    This code is explained better in the 'getObjectMidpoints' function,
    which has code that's basically identical to this.

    You're probably going to ask 'why have you duplicated the code?'

    Short answer: I don't want the production code to have a dependency on
    this debug code, and, if this debug code was to be omitted, I'd have a
    redundant dependency in the production code.

    :param hsvIn: the hsv image that contains the objects
    :return: masks for each of the 7 coloured objects that are in it.
    (cyan, red, white, blue, green, yellow, orange)
    """
    cyan_mask: np.ndarray = cv2.inRange(hsvIn, min_cyan, max_cyan)
    red_mask: np.ndarray = cv2.inRange(hsvIn, min_red, max_red)
    white_mask: np.ndarray = cv2.inRange(hsvIn, min_white, max_white)
    blue_mask: np.ndarray = cv2.inRange(hsvIn, min_blue, max_blue)
    green_mask: np.ndarray = cv2.inRange(hsvIn, min_green, max_green)
    yellow_mask: np.ndarray = cv2.inRange(hsvIn, min_yellow, max_yellow)
    orange_mask: np.ndarray = cv2.inRange(hsvIn, min_orange, max_orange)

    cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel33)

    white_mask = cv2.subtract(white_mask, cyan_mask)

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
     Tuples are in the form (left image mask, right image mask).
     The list itself is in the order
     cyan->red->white->blue->green->yellow->orange.
    """
    leftMasks = getObjectMasks(cv2.cvtColor(left_in, cv2.COLOR_BGR2HSV))
    rightMasks = getObjectMasks(cv2.cvtColor(right_in, cv2.COLOR_BGR2HSV))

    theMasks: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(0, 7):
        theMasks.append((leftMasks[i], rightMasks[i]))

    return theMasks


def showMasksForDebugging(i_left: np.ndarray, i_right: np.ndarray, f: int) -> \
        None:
    """
    This is here mostly for debugging purposes, showing the masks of each image.
    The 'left' image will have some text on it with the colour name identifier
    of the object, as well as what number frame this is.
    Apologies in advance if the annotations overlap with the position of
    one of the objects in the set of images you are using to mark this.

    :param i_left: left image (BGR format)
    :param i_right: right image (BGR format)
    :param f: frame number
    :return: nothing.
    """
    # order in which the masks of objects are returned
    objectOrder: List[str] = \
        ["cyan", "red", "white", "blue", "green", "yellow", "orange"]

    i = 0
    debugMasks = getStereoMasks(i_left, i_right)
    for m in debugMasks:
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


def handleShowingStuff() -> None:
    """
    This is here to handle actually showing stuff when debugging the program.
    Prints something to console to let the user know that they need to press
    something to continue, handles doing the cv2.waitKey(0) call (thereby
    allowing any pending cv2.imshow()'d images to be shown), and,
    if q or escape are pressed, the program will close.

    :return: nothing is returned.
    """
    print("press something to continue (press q or escape to quit)")
    key = cv2.waitKey(0)
    # quit if escape (27) or q (113) are pressed
    if key == 27 or key == 113:
        cv2.destroyAllWindows()
        print("quitting!")
        sys.exit(0)


"""
-- THE ACTUALLY IMPORTANT CODE THAT ACTUALLY DOES STUFF --

Yep. Everything from here is actually of some use.
"""

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
"""
A 3*3 numpy array of 1s, to be used when closing up holes in some of the masks
"""


def getObjectMidpoint(objectMask: np.ndarray) -> Tuple[float, float]:
    """
    Returns the midpoint of the region of 1s in the given binary image array

    If there is no region of 1s, we return (-1,-1). If there's a 1 on the
    boundary of the image, once again, we return (-1,-1), because we'll know
    that the midpoint we find will probably be inaccurate.

    This estimates the midpoint by finding the upper/lower X and Y bounds of
    that region of 1s in the image. Yes, it's a pretty naiive, brute force-y
    method. However, I tried several object/blob detection algorithms within
    OpenCV, however, none of them really worked as intended (not detecting the
    objects in the earlier images, not really being able to work out which
    keypoints correspond to each object, and refusing to detect the single
    object when I go through the effort of masking out everything else in the
    image), so I went 'fuck it, guess I'm doing it myself'

    :param objectMask: binary image, with 1s in the area where the object
     with the midpoint being looked for is, and 0s everywhere else.
    :return: A tuple holding the the midpoint of the object.
     If the object isn't present (mask all 0s), a value of (-1,-1) is returned.
     Additionally, if the object is at the edge of the image (a minimum is 0,
     or a maximum is at the maximum possible x/y), that heavily implies that
     the object is partially out-of-frame. Therefore, as that means the true
     bounds are likely to be out-of-frame, this midpoint detector will not find
     the true midpoint of the object, so it will give up and return -1s for that
     as well.
    """

    if objectMask.any():
        if objectMask[0].any() or objectMask[-1].any():
            # if there's anything in the topmost or bottommost row, that means
            # there's something on the image boundary, meaning that the midpoint
            # found will be inaccurate, so we're not going to bother finding it.
            return -1, -1
    else:
        # if nothing in the objectMask is a 1, we return -1s.
        return -1, -1

    # obtaining the shape of the actual mask image
    yx: Tuple[int, int] = objectMask.shape
    ny: int = yx[0]
    nx: int = yx[1]

    # and declaring some variables to hold the info we find out about
    # the shape of the object.
    minX: int = -1
    maxX: int = -1
    minY: int = -1
    maxY: int = -1
    notFoundFirst: bool = True  # set this to false when we find first pixel

    # now we just casually loop through the image pixels,
    # and find out about what sort of shape the object has
    for y in range(1, ny-1):
        # we already established that the topmost/bottommost rows are empty.
        if not objectMask[y].any():
            # we skip this row if there's no 1s in it. you are welcome.
            continue
        for x in range(0, nx):
            if objectMask[y][x] != 0: # if it's not 0, we've found it.
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

    # if it's at the x bounds of the image, the result definitely won't be
    # accurate, so we'll just return -1, -1.
    # (we already checked the y bounds earlier on)
    if (minX == 0) or (maxX == nx-1):
        return -1, -1

    # working out widths and heights
    w: int = (maxX - minX)
    h: int = (maxY - minY)

    # using that and the lower bounds for x and y to find the midpoints
    xMid: float = minX + (w / 2.0)
    yMid: float = minY + (h / 2.0)

    # and returning a tuple with those midpoints
    return xMid, yMid


def getObjectMidpoints(hsvIn: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """
    Gets the midpoints for the objects that may be in the given HSV image.
    We generate masks that contain only the region of the HSV object that is
    occupied by the pixels that make up a particular object, do a bit of cleanup
    for the objects that have somewhat overlapping pixel values, and then
    put those masks into getObjectMidpoint to produce a dictionary holding the
    midpoints of each of the objects in the image.

    If an object is not present in the image, its entry in the dictionary will
    have the default value of (-1,-1) instead of an actual midpoint.

    You might be wondering 'why am I making all the masks at once and then
    finding the midpoints from them all at once instead of just making a mask,
    getting the midpoint, and moving on to the next mask?'

    Simple answer: Doing it this way allows me to do the cleanup stuff that
    needs to be done for cyan/white/orange/yellow more effectively,
    and it means I can declare the results dictionary as a literal and also
    immediately return it. yay efficiency.

    :param hsvIn: the hsv image that contains the objects
    :return: dict with midpoints for each of the 7 coloured objects
     that might be present in the given image. If not present, midpoint will
     be (-1,-1). Keys are (cyan, red, white, blue, green, yellow, orange)
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

    # finally making + returning the dict with the midpoints of each object
    return {
        "cyan": getObjectMidpoint(cyan_mask),
        "red": getObjectMidpoint(red_mask),
        "white": getObjectMidpoint(white_mask),
        "blue": getObjectMidpoint(blue_mask),
        "green": getObjectMidpoint(green_mask),
        "yellow": getObjectMidpoint(yellow_mask),
        "orange": getObjectMidpoint(orange_mask)
    }


def getStereoPositions(left_in: np.ndarray, right_in: np.ndarray) -> \
        Dict[str,
             Tuple[Tuple[float, float],
                   Tuple[float, float]]
        ]:
    """
    Gets the positions of objects in the left and right *coloured* images.
    This **will** assume that the dimensions of the left and right images
    are identical, and that the two images are Y-aligned already.

    :param left_in: the left input image *in colour*
    :param right_in: the right input image *in colour*
    :return: a dictionary with the names of the objects in both images as keys,
     and a tuple, containing the  x' and y' positions of that particular object
     in both images as the value

        1st tuple in the value tuple: (x',y') from left image.
        2nd tuple in the value tuple: (x',y') from right image.

        If an object is not present in **both** images, it will **not** be
        present in the returned dictionary.
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
    halfY: float = yx[0] / 2
    halfX: float = yx[1] / 2

    posDict: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    """a dictionary for all the calculated (X',Y') positions for each image"""

    # obtains the keys from lDict but as a list so it can be foreach'd
    leftKeys: List[str] = [*lDict.keys()]
    """
    unpacking the keys/object names as a list so we can iterate through them.
    Why do I need to do this? Because Dict.keys() returns a KeysView object,
    which isn't iterable, and is generally awkward to work with. However,
    putting a KeysView kv into [*kv] basically unpacks it into a list.
    Which we can easily iterate through. So that's what happens here.
    """

    # now looks through each of those keys
    for k in leftKeys:
        if lDict[k] != (-1, -1):
            if rDict[k] != (-1, -1):
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



class Vector3D:
    """
    A class to represent a vector in 3D space
    """
    def __init__(self, x: float, y: float, z: float):
        """
        Constructs a Vector3D with the given x, y, and z coordinates
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        """
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def magnitude(self) -> float:
        """
        dist between 2 3D points:
        sqrt( ((x2-x1)^2) + ((y2-y1)^2) + ((z2 - z1)^2))
        and we know that x1,y1,z1 = 0 already (because vector comes from origin
        0) and x2, y2, and z2 are the x, y, and z of this vector.

        :return: the magnitude of this vector
        """
        return sqrt((self.x ** 2) + (self.y ** 2) + (self.z ** 2))

    # noinspection PyUnresolvedReferences
    def normalized(self):
        """
        Normalizes this Vector3D

        :return: This Vector3D, but with a magnitude of 1 instead.
        if this already had a magnitude of 0, it'll return itself as-is
        """
        mag: float = self.magnitude()
        if mag > 0:
            self.x = self.x / mag
            self.y = self.y / mag
            self.z = self.z / mag

        return self

    def subtract(self, other):
        """
        Subtracts the other Vector3D from this Vector3D, returning this
        modified Vector3D.

        :param other: the other Vector3D to subtract from this
        :return: this Vector3D minus 'other'
        """
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self

    # noinspection PyUnresolvedReferences
    def dot(self, other) -> float:
        """
        Returns the dot product of this vector3D and the other vector3D
        :param other: the other Vector3D this is being dot producted against
        :return: the dot product of this and the other vector3D
        """
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def isZero(self) -> bool:
        """
        Check if this vector is (0,0,0)
        :return: Returns true if x y and z are exactly equal to 0
        """
        return (self.x == 0) and (self.y == 0) and (self.z == 0)

    def __str__(self) -> str:
        """
        Outputs this as a string; as a tuple in the form (x, y, z)
        :return: a string of the tuple (self.x, self.y, self.z)

        """
        return str((self.x, self.y, self.z))


def normalizeVector(v: Vector3D) -> Vector3D:
    """
    Obtains a version of this Vector3 with the x, y, and z normalized

    :return: a version of this with a magnitude of 1. However,
    if this already had a magnitude of 0, it'll return a vector3 with
    a value of (0,0,0)
    """
    return Vector3D(v.x, v.y, v.z).normalized()


def subVector(lhs: Vector3D, rhs: Vector3D) -> Vector3D:
    """
    Subtracts one Vector3D from another Vector3D, without modifying the original
    :param lhs: left hand side
    :param rhs: right hand side vector
    :return: Vector3D that's equal to lhs - rhs
    """
    return Vector3D(lhs.x, lhs.y, lhs.z).subtract(rhs)


def normalizeVectorBetweenPoints(fromVec: Vector3D, to: Vector3D) -> Vector3D:
    """
    Get lhs-rhs but normalized instead (leaving lhs and rhs untouched)

    >>> normalizeVectorBetweenPoints(Vector3D(0,0,0),Vector3D(1,1,1))
    Vector3D(0.5773502691896258, 0.5773502691896258, 0.5773502691896258)

    >>> normalizeVectorBetweenPoints(Vector3D(0,0,0),Vector3D(2,2,2))
    Vector3D(0.5773502691896258, 0.5773502691896258, 0.5773502691896258)

    >>> normalizeVectorBetweenPoints(Vector3D(0,0,0),Vector3D(1,1.5,2))
    Vector3D(0.3713906763541037, 0.5570860145311556, 0.7427813527082074)

    >>> normalizeVectorBetweenPoints(Vector3D(1,1,1),Vector3D(1,1,1))
    Vector3D(0, 0, 0)

    :param fromVec: going from this vector
    :param to: to this other vector
    :return: to - fromVec but normalized
    """

    return subVector(fromVec, to).normalized()



placeholderOutString: str = "{:5}  {:8}  {:8.2e}  {}"
"""
This is a placeholder string to be used when formatting the frame-by-frame
printout of object data. Double space between each thing of data.
1st value: will be the frame number. 5 width, right-aligned.
2nd value: object identifier. 8 width, also left-aligned
3rd value: object distance (Z pos, in metres). 8 width, in the form 1.23e+45

4th value: just the raw (X,Y,Z) position of the object in 3D space (in metres),
for sake of curiousity.
"""
focalLength: float = 12
"Focal length of camera is 12m"
baseline: float = 3500
"baseline between cameras is 3.5km -> 3500m"
pixelSize: float = float(1e-5)
"pixel spacing: 10 microns -> 1e-5 metres"


def calculateAndPrintPositionsOfObjects(leftIm: np.ndarray,
                                        rightIm: np.ndarray,
                                        frameNum: int = 0) -> \
        Dict[str, Vector3D]:
    """
    Given a left image (BGR), a right image (BGR), and a frame number (optional)
    , this method will print the details about the identifiers and the depths
    (Z axis positions) of the objects in the image (formatted as per the
    assignment brief, using the global placeholderOutString), and will return a
    dictionary with vector3 positions of the objects in the images.

    Objects in only one image will be omitted. Distances will be in metres.

    leftIm and rightIm must be in BGR colour, and have identical dimensions.

    This will use the focalLength, baseline, and pixelSize global variables to
    calculate the positions of the objects.

    placeholderOutString, focalLength, baseline, and pixelSize are present
    just above this function.

    :param leftIm: The left stereo image to look at (BGR colour)
    :param rightIm: The right stereo image to look at (BGR colour)
    :param frameNum: The frame number (optional). Will only be used to prefix
     the printout. If not supplied, 0 will be used.
    :return: A dictionary with the identifiers of the objects identified, along
     with their vector3 positions, in metres, relative to the midpoint between
     the cameras.
    """

    imgPositions: \
        Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = \
        getStereoPositions(cv2.cvtColor(leftIm, cv2.COLOR_BGR2HSV),
                           cv2.cvtColor(rightIm, cv2.COLOR_BGR2HSV))
    """
    These are the left and right image X'Y' coords of every object
    in both the left and the right images.
    """

    posXYZ: Dict[str, Vector3D] = {}
    """
    this will hold the X, Y, Z coords of the objects in 3D space,
    using the midpoint between the cameras as the origin,
    measured in metres.
    """

    objKeys: List[str] = [*imgPositions.keys()]
    """
    unpacking the keys/object names as a list so we can iterate through them.
    Why do I need to do this? Because Dict.keys() returns a KeysView object,
    which isn't iterable, and is generally awkward to work with. However,
    putting a KeysView kv into [*kv] basically unpacks it into a list.
    Which we can easily iterate through. So that's what happens here.
    """
    for k in objKeys:

        # we obtain the info about current object's 2d pos from imagePositions
        currentPos: Tuple[Tuple[float, float], Tuple[float, float]] = \
            imgPositions[k]

        # x disparity = xl - xr
        xDisparity: float = currentPos[0][0] - currentPos[1][0]
        """ x disparity = xL - xR """

        #print(currentPos[0][0])
        #print(currentPos[1][0])
        #print(xDisparity)

        rawZ: float = (focalLength * baseline) / (xDisparity * pixelSize)
        """
        Z = (f * b) / (xl - xr)
        which is how we work out what dist is.
        """

        rawX: float = ((-currentPos[1][0] * pixelSize) / focalLength) * rawZ
        """
        (-xr/f) = (X/Z), therefore (-xr/f) * Z = X
        So I used that equation to find out what the actual X position
        of the object is in 3D space.
        """

        # just obtaining the midpoint of the Ys real quick
        yMid: float = (currentPos[0][1] + currentPos[1][1]) / 2

        rawY: float = -((yMid * pixelSize) / focalLength) * rawZ
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
        posXYZ[k] = Vector3D(rawX, rawY, rawZ)

        # Now, we just print the required info, as per the specification.
        print(placeholderOutString.format(
            frameNum,  # what frame number this is
            k,  # identifier of this object
            rawZ,  # we print the Z depth
            posXYZ[k]  # the XYZ pos. Not needed, but printed for transparency.
        ))

    # we finish by returning the posXYZ dictionary.
    return posXYZ



def isThisAStraightLine(line: List[Vector3D]) -> bool:
    """
    Returns whether or not a sequence of 3D points is a straight line,
    using the getNormDifferenceBetweenPoints function.

    If there's 2 or fewer points, it certainly ain't bent, so it will return
    true. Otherwise, it'll return true if all the points have an identical
    normalized vector between them, false otherwise.

    Due to the inherent uncertainty with how the points are calculated, I am
    giving some leeway in the calculations.

    :param line: the sequence of 3D points
    :return: true if they're a straight enough line, false otherwise.
    """

    if len(line) < 3:
        return True

    thisIndex: int = 1
    """ A cursor to the index of the line used for this iteration """

    startEndDiff: Vector3D = \
        normalizeVectorBetweenPoints(line[0], line[-1])

    # TODO
    # make list of all the normalized dot values (omitting vectors of zero)
    # then analyse if it's sus?

    print(startEndDiff)


    susCount: int = 0

    maxSus: int = int(len(line)/8)

    print("imposter is " + str(maxSus) + " sus!")

    while True:
        thisIndex += 1 #we move to the next index of the list
        if thisIndex == len(line):
            # we're basically emulating a do/while loop here
            # with a while condition of thisIndex < len(line)
            # and if the program hasn't failed yet, this succeeded.
            return True

        thisDiff: Vector3D = \
            normalizeVectorBetweenPoints(line[thisIndex-1], line[thisIndex])

        print(thisDiff)

        if thisDiff.isZero():
            continue

        thisDot: float = startEndDiff.dot(thisDiff)

        print(thisDot)

        # if the normalized difference between the previous index and this index
        # is not identical to the original normalized difference
        # if getNormVectorBetweenPoints(line[thisIndex-1], line[thisIndex]) != \
        #    normDiff:
        if not (isclose(thisDot, 1.0)):
            susCount += 1
            if susCount == maxSus:
                print("Failed at frame " + str(thisIndex))
                return False
            else:
                print("Frame " + str(thisIndex) + " is sus " + str(susCount))

        #lastDiff = thisDiff





def checkIfImageWasOpened(filename: str, img: Union[np.ndarray, None]) -> None:
    """
    This will check if the image with the given filename could be opened

    :param filename: the name of the file
    :param img: the numpy.ndarray (or lack thereof) that opencv could open
     using that given filename
    :return: nothing. But, if the file couldn't be opened (causing img to be
     None instead of a np.ndarray), the program complains and promptly closes.
    """
    if isinstance(img, type(None)):
        """
        If the image is actually NoneType, the program complains and closes.
        
        And you may ask yourself 'why am I doing this in such a convoluted way?'
        Simple answer: There is no simpler way to do it.
        Opencv doesn't throw an exception if the image couldn't be read, it just
        returns None.
        So, if it does return None, I could just detect it with 'if im == None',
        right? WRONG!
        Thing is, if it doesn't return None, it returns a numpy.ndarray. And if
        you attempt to compare one of those against None, guess what? You get
        a ValueError and a snarky comment saying 'oh no the truth value of this
        is ambiguous pls use .any() or .all() instead'
        But if I try to use those methods to check if the image exists, and the
        image doesn't exist, guess what? You get an AttributeError.
        
        Now, do I want to bother with throwing and catching exceptions manually?
        No. cba to deal with that overhead.
        
        Would I have preferred it if OpenCV could have just thrown an exception
        or just returned an empty array instead of returning a mcfucking None?
        
        Yes.
        
        But, alas, we live in a society. Rant over.
        """
        print("ERROR: Could not open the file called " + filename)
        sys.exit(1)

"""
-- THE MAIN PROGRAM --

Everything from here is the stuff that runs when you start running this.
"""

if len(sys.argv) < 4:
    # If you don't give 3 command line arguments, the program will complain
    print("Usage:", sys.argv[0],
          "<frame count> ",
          "<left-hand frame filename template> ",
          "<right-hand frame filename template>",
          file=sys.stderr)
    # and promptly quit
    sys.exit(1)

# reads the 1st actual command line argument as the count of frames to look at
nframes: int = int(sys.argv[1])


objectPositions: Dict[str, List[Vector3D]] = {
    "cyan": [],
    "red": [],
    "white": [],
    "blue": [],
    "green": [],
    "yellow": [],
    "orange": []
}
"""
This is a dictionary which will hold the positions of the objects for every
frame.
"""
print("frame  identity  distance")  # header for the required frame data info.

for frame in range(0, nframes):
    # we work out the filenames for the left and right images for this frame,
    # and then we open those images using opencv.
    # (and also check to see if the images could actually be opened.)
    fn_left = sys.argv[2] % frame
    im_left: np.ndarray = cv2.imread(fn_left)
    checkIfImageWasOpened(fn_left, im_left)

    fn_right = sys.argv[3] % frame
    im_right: np.ndarray = cv2.imread(fn_right)
    checkIfImageWasOpened(fn_right, im_right)

    if debugging:
        """
        You remember those testing functions from earlier, right?
        Well, this is where they get used. If you enabled 'testing' ofc.
        """
        debug(fn_left, fn_right, im_left, im_right, frame)
        # END OF TESTING CODE

    posXYZ: Dict[str, Vector3D] = \
        calculateAndPrintPositionsOfObjects(im_left, im_right, frame)
    """
    We obtain the identifiers and XYZ positions of all the objects that are
    present within both of the stereo frames.
    """

    #foundObjects: List[str] = [*posXYZ.keys()]
    # obtaining the keys for the objects which we have XYZ positions for
    for o in [*posXYZ.keys()]:
        objectPositions[o].append(posXYZ[o])
        # and we append them to the list of all positions for that object.



    # we put it on the list with all the others
    #frames.append(fData)

    # and we also print the data we need to print
    #fData.printFrameData()

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

# TODO
# work out trajectory of each object.
# I have XYZ pos of each object for each frame that they're there for,
# I just need to use them

# TODO
# get normalized distances between each of the points for each object
# if all identical: straight line
# if not all identical: alien

ufoList: List[str] = []
""" A list to hold all the UFOs """

for k in [*objectPositions.keys()]:
    print(k)
    if not isThisAStraightLine(objectPositions[k]):
        ufoList.append(k)

print(ufoList)


print("TODO remove this print statement")
