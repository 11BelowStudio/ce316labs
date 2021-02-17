import sys, cv2
import numpy as np

if len (sys.argv) < 3:
    print("Usage:", sys.argv[0],
          "<frame count> ",
          "<left-hand frame filename template> ",
          "<right-hand frame filename template>",
          file=sys.stderr)
    sys.exit(1)

print("deez nutz lmao gottem")

nframes = int (sys.argv[1])
for frame in range (0, nframes):
    fn_left = sys.argv[2] % frame
    im_left = cv2.imread(fn_left)
    fn_right = sys.argv[3] % frame
    im_right = cv2.imread(fn_right)
    print(fn_left)
    print(fn_right)
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