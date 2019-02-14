#+
# AUTHOR:
#    Mark Hannel
#
# NAME:
#    edge_convolve
#
# PURPOSE:
#    Convolves an array with a kernel and returns the result.  The array
#    is manipulated so that the edges
#
# CATEGORY:
#    Math routine
#
# CALLING SEQUENCE:
#    result = edge_convolve(a,k)
#
# INPUTS:
#    a: [nx, ny] image data
#    k: [kx, ky] kernel (dimensions must be less than that of a
#
# OUTPUTS:
#    result: [nx, ny]
#
# EXAMPLE:
#    >>> result = edge_convolve(a,k)
#

import numpy as nmp
from scipy import ndimage
from builtins import range

def extend_array(a,n,m=0):
    """
    Creates a larger version of the array a by repeating the edges 
    """
    if type(a) != nmp.ndarray:
        print('a must be an numpy array')
        return -1

    new = nmp.empty([a.shape[0]+2*n,a.shape[1]+2*m])
    new[n:-n,m:-m] = a

    #Grab the corner values
    top_left  = a[ 0, 0]*nmp.ones(n)
    top_right = a[ 0,-1]*nmp.ones(n)
    bot_left  = a[ -1,0]*nmp.ones(n)
    bot_right = a[-1,-1]*nmp.ones(n)

    #Grab sides
    top_side   = a[ 0, :]
    bot_side   = a[-1, :]
    left_side  = a[ :, 0]
    right_side = a[ :,-1]
    
    for i in range(0,n):
        new[ i,m:-m] = top_side      #Repeat the top side of a
        new[-i-1,m:-m] = bot_side    #Repeat the bottom side of a
        
    for j in range(0,m):
        new[n:-n, j]   = left_side   #Repeat the left side of a
        new[n:-n,-j-1] = right_side  #Repeat the right side of a
        new[ j,0:n]    = top_left    #Fill the top left corner
        new[-j-1,0:n]  = bot_left    #Fill the bottom left corner
        new[ j,-n:]    = top_right   #Fill the top right corner
        new[-j-1,-n:]  = bot_right   #Fill the bottom right corner

    return new

def edge_convolve(a,k):
    """
    Convolves an array with a kernel and returns the result.  The array
    is manipulated so that the edges

    Inputs:
    a: [nx, ny] image data
    k: [kx, ky] kernel (dimensions must be less than that of a

    Outputs:
    result: [nx, ny]

    Example:
    >>> result = edge_convolve(a,k)
    """

    if type(a) != nmp.ndarray:
        try: 
            a = nmp.array(a)
        except ValueError:
            print('a must be array-like')
            return -1

    if type(k) != nmp.ndarray:
        try: 
            k = nmp.array(k)
        except ValueError:
            print('k must be array-like')
            return -1
    
    if a.ndim != 2:
        print('a must be a 2D array of floats or ints')
        return -1

    if k.ndim != 2:
        print('k must be a 1D or 2D array of floats or ints')
        return -1
    
    a_x = a.shape[0]
    a_y = a.shape[1]

    k_x = k.shape[0]
    k_y = k.shape[1]

    if a_x < k_x or a_y < k_y:
        print('k must have smaller dimensions than a')
        return -1

    #Extend the edges of a
    #x_repeat = a.shape[0]/2
    #y_repeat = a.shape[1]/2
    x_repeat = 1
    y_repeat = 1

    
    a = extend_array(a,x_repeat,y_repeat)

    return ndimage.convolve(a,k)[x_repeat:-x_repeat,y_repeat:-y_repeat]
