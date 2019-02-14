#+
# NAME:
#    savgol2d()
#
# PURPOSE:
#    Generate two-dimensional Savitzky-Golay smoothing and derivative kernels
#
# CALLING SEQUENCE:
#    filter = savgol2d(dim, order)
#
# INPUTS:
#    dim: width of the filter [pixels]
#    order: The degree of the polynomial
#
# KEYWORD PARAMETERS:
#    dx: order of the derivative to compute in the x direction
#        Default: 0 (no derivative)
#    dy: order of derivative to compute in the y direction
#        Default: 0 (no derivative)
#
# OUTPUTS:
#    filter: [dim, dim] Two-dimensional Savitzky-Golay filter
#
# EXAMPLE:
# IDL> dadx = convol(a, savgol2d(11, 6, dx = 1))
#
# MODIFICATION HISTORY:
#  Algorithm based on SAVGOL2D:
#  Written and documented
#  Fri Apr 24 13:43:30 2009, Erik Rosolowsky <erosolo@A302357>
#
#  02/06/2013 Written by David G. Grier, New York University
#  09/2013 Translated to Python by Mark D. Hannel, New York University
#  Copyright (c) 2013 David G. Grier
#-

import numpy as nmp
from itertools import count

def savgol2d(dim, order, dx = 0, dy = 0):
   """
   Generates two-dimensional Savitzky-Golay smoothing and derivative kernels

   Inputs:
   dim: width of the filter [pixels]
   order: The degree of the polynomial

   Parameters:
   dx: order of the derivative to compute in the x direction
        Default: 0 (no derivative)
   dy: order of derivative to compute in the y direction
        Default: 0 (no derivative)

   Example:
   filter = savgol2d(11,6,dx=1)
   """
   umsg = 'USAGE: filter = dgsavgol2d(dim, order)'

#   if n_params() != 2 :    ### Python checks for correct number of 
#                           ### necessary arguments
#      print umsg
#      return -1

   if type(dim) != int:
      print(umsg)
      print('DIM should be the integer width of the filter')
      return -1
 
   if type(order) != int:
      print(umsg)
      print('ORDER should be the integer order of the interpolaying polynomial')
      return -1

   if order > dim :
      print(umsg)
      print('ORDER should be less than DIM')
      return -1
 
   if dx < 0 or dy < 0 or type(dx) != int or type(dy) != int:
      print(umsg)
      print('DX and DY should be non-negative integers')
      return -1

   if dx + dy >= order:
      print(umsg)
      print('DX + DY should not be greater than ORDER')
      return -1

   npts = dim**2

   temparr = nmp.arange(0,dim,dtype=float)-dim/2.

   x = nmp.tile(temparr,dim)
   y = nmp.repeat(temparr,dim)

   temp = int( (order+1)*(order+2)/2 )
   Q = nmp.arange(temp*npts,dtype = float).reshape(npts,temp)
   n = 0
   for nu in range(0, order+1):
      ynu =  y**nu
      for mu in range(0, order-nu+1):
         Q[:, n] = x**mu * ynu
         n += 1

   a = nmp.linalg.inv(nmp.dot(Q.transpose(),Q))
   a = nmp.dot(Q,a).transpose()

   Filter = nmp.zeros(npts)
   b = nmp.zeros(npts)
   b[0] = 1.
   ndx = dx +(order + 1) * dy

   for i in range(0, npts):
      Filter[i] = nmp.dot(a,b)[ndx]
      b = nmp.roll(b, 1)

   return Filter.reshape(dim, dim)
