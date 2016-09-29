import os
import re
import numpy as np

def goToFileDir(it):
    print "Current directory:"
    print(os.getcwd())
    os.chdir(os.path.dirname(os.path.realpath(it)))
    print "Change to:"
    print(os.getcwd())

def drawGaussian(canvas, center_x, center_y, std_x, std_y=None):
    """
    :param canvas: the matrix on which you will add the returned drawing.
    :param center_x: the center of the gaussian on X
    :param center_y: the center of the gaussian on Y
    :param std_x: the standard deviation of X
    :param std_y: the standard deviation of Y, use std_x if it is not specify or if it is set to None.
    :return: return a new matrix of the size of the provided canvas-matrix, but flattened. It is ready to be added to your canvas.
    """
    x = np.arange(0, canvas.shape[0], 1)
    y = np.arange(0, canvas.shape[1], 1)
    X,Y = np.meshgrid(x, y)
    if std_y == None:
        std_y = std_x
    pattern = np.exp(-((X-center_x)**2)/(2.0*std_x**2.0)-((Y-center_y)**2.0)/(2.0*std_y**2.0))
    return pattern.flatten()

def getGaussian(t, amplitude, centre, std): ## centre = centre gauss en ms, compter 3sigma pour largeur, sim_time = duree de la simulation en ms
    return amplitude*np.exp((-(t-centre)**2)/(std**2))

def tryint(s):
    try:
        return float(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    #print [ tryint(c) for c in re.split('([0-9]+)', s) ]
    # h = [ tryint(c) for c in re.split('([\d]+.?[\d]+)', s) ]
    h = [ tryint(c) for c in re.findall('(\d+\.?\d+)', s) ]
    #print h
    return h

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    return sorted(l, key=alphanum_key)