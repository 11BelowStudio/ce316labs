import cv2, math


def mean(im):
    "return mean of image"
    ny, nx, nc = im.shape
    sum = 0
    for y in range(0, ny):
        for x in range(0,nx):
            for c in range(0,nc):
                sum += im[y,x,c]
    return sum/(ny*nx*nc)


def sd_slow(im):
    "The average squared deviation from the mean"
    ny, nx, nc = im.shape
    sum = 0
    theMean = mean(im)
    for y in range(0,ny):
        for x in range(0, nx):
            for c in range(0, nc):
                v = im[y, x, c] - theMean
                sum += (v * v)
    return math.sqrt(sum/ny/nx/nc)


#TODO: look this up from lecture slides
def sd(im):
    return 0


sx = cv2.imread("sx.jpg")
print("The mean:")
print("My method:")
print(mean(sx))
print("cv2 method:")
print(sum(cv2.mean(sx))/3)


print("standard deviation")
print("my method (slow):")
print(sd_slow(sx))