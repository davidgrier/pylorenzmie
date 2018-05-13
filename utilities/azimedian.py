#+
# NAME:
#    azimedian
#
# PURPOSE:
#    Compute the azimuthal median of a two-dimensional data set
#    over angles about its center.
#
# CATEGORY:
#    Image Processing
#
# CALLING SEQUENCE:
#    result = azimedian(data)
#
# INPUTS:
#    data: two dimensional array of any type except string or complex
#
# KEYWORD PARAMETERS:
#    center: coordinates of center: [xc, yc].  Default is to use data's
#        geometric center.
#
#    rad: maximum radius of average [pixels]
#        Default: half the minimum dimension of the image.
#
#    weight: relative weighting of each pixel in data.
#        Default: uniform weighting.
#
#    deinterlace: If set to an even number, average only over even 
#        numbered lines.  Similarly if set to an odd number.
#        This is useful for analyzing interlaced video images.
#
# OUTPUTS:
#    result: data averaged over angles as a function of radius from
#        the center point, measured in pixels.  Result is single
#        precision.
#
# KEYWORD OUTPUTS:
#    rho: the radial position of each pixel in DATA relative to the
#        center at (xc, yc).
#
#    values: Azimuthal median at each pixel.
#
#    deviates: difference between DATA and azimuthal median at each
#        pixel.
#
# PROCEDURE:
#    data[y, x] sits at radius rrho = sqrt((x-xc)**2 + (y-yc)**2) 
#        from the center, (xc, yc).  Let R be the integer part
#        of rho, and dR the fractional part.  Then this point is
#        averaged into result(R) with a weight 1-dR and into
#        result(R+1) with a weight dR.
#
# RESTRICTIONS:
#    data must be two-dimensional and must not be string type
#
# MODIFICATION HISTORY:
# 08/18/13 Written in IDL by David G. Grier, New York University
# 11/11/13 Translated to Python by Mark D. Hannel, New York University
#
# Copyright (c) 2013 David G. Grier
#-

import numpy as np
from builtins import range

default = np.array([])

def azimedian(a, center = default, rad = None, deinterlace = 0, weight = 0,
                 rho = default, deviates=0, values=0):
   """
    Compute the azimuthal median of a two-dimensional data set
    over angles about its center.

    Inputs:
    data: two dimensional array of any type except string or complex

    Parameters:
    center: coordinates of center: [xc, yc].  Default is to use data's
        geometric center.

    rad: maximum radius of average [pixels]
        Default: half the minimum dimension of the image.
        Must be an int.

    weight: relative weighting of each pixel in data.
        Default: uniform weighting.

    deinterlace: If set to an even number, average only over even 
        numbered lines.  Similarly if set to an odd number.
        This is useful for analyzing interlaced video images.

    Example:
    >>> med = azimedian(data)
    """


   umsg = 'USAGE: result = azimedian(data)'

   if type(a) != np.ndarray:
      print(umsg)
      print('DATA must be a numpy array')
      raise TypeError


   sz = a.ndim
   if sz != 2 :
      print(umsg)
      print('DATA must be a two-dimensional array')
      raise TypeError

   nx,ny = a.shape			# width, height 

   #Set center. Default is the center of the image
   if len(center) == 2 :
      xc = float(center[0])
      yc = float(center[1])
   else:
      xc = 0.5 * (nx-1)   # indices start at 0
      yc = 0.5 * (ny-1)   # ... in y also


   #Set the maximum radius.  Default is the largest for an enscribed circle 
   rmax = rad if type(rad) == int else min(nx/2., ny/2.)

   #Account for complex data
   if type(a[0,0]) == complex: # complex data
      med = np.zeros(int(rmax+1), complex)
   else:                             # accumlate other types into double
      med = np.zeros(int(rmax+1))

   #Add contrast to the image if desired
   if weight != 0: 
      a *= weight

   #Distance from center to each pixel
   x = (np.arange(nx) - xc)**2
   y = (np.arange(ny) - yc)**2
   x,y = np.meshgrid(x,y)
   r = np.sqrt(x+y)

   #Deinterlace
   if deinterlace != 0:
      n0 = deinterlace % 2
      a = a[n0::2, :]
      rho = r[n0::2, :]
   else:
      rho = r

   #Make an array n which has the points lying between
   #r and r+1.  Evaluate the median of the image over these pixels
   for i in range(int(rmax+1)):
      n = (r>=i)*(r<(i+1))
      med[i] = np.median(a[n])
         


   #if arg_present(values) : 
   #   values = med[round(r)]
      
   #if arg_present(deviates) : 
   #   deviates = a - med[round(r)]

   return med
