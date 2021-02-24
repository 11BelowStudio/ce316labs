#!/usr/bin/env python
'''Generate a series of calibration frames using POV-ray.'''
from __future__ import division
import sys, os, math

def do_scene (x, y, z, fn):
    '''Generate a frame with the camera at x,y,z into fn and render it.'''
    f = open (fn, 'w')
    #print >>f, '#include "calibration_target.pov"'
    #print >>f, 'camera {'
    #print >>f, '  location <%.2f, %.2f, %.2f>' % (x, y, z)
    #print >>f, '  look_at <%.2f, 300, 280>' % x
    #print >>f, '}'
    f.write('#include "calibration_target.pov"')
    f.write('camera {')
    f.write('  location <%.2f, %.2f, %.2f>' % (x, y, z))
    f.write('  look_at <%.2f, 300, 280>' % x)
    f.write('}')
    #print (f, '#include "calibration_target.pov"')
    #print (f, 'camera {')
    #print >> f, '  location <%.2f, %.2f, %.2f>' % (x, y, z)
    #print >> f, '  look_at <%.2f, 300, 280>' % x
    #print >> f, '}'
    f.close ()
    os.system ('povray +I%s +FN +W640 +H480 +AA +A0.3 -D &> /dev/null' % fn)

# Main program: calculate the camera positions and generate the frames.
n = 30
for i in range (0, n):
    x =  75 + 100 * math.cos (i * math.pi / n)
    y =  50 + 100 * math.cos (i * math.pi / n)
    z = 650 + 100 * math.sin (i * math.pi / n)
    print (y, z)
    fn = 'calib-%3.3d.pov' % i
    do_scene (x, y, z, fn)
