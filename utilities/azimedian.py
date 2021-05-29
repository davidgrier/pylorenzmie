import numpy as np


def azimedian(data, center):
   '''Azimuthal median of data about center
    
   Parameters
   ----------
   data : array_like
       image data
   center : tuple
       (x_p, y_p) center of azimuthal median

   Returns
   -------
   avg : array_like
       One-dimensional azimuthal median of data about center
   '''
   x_p, y_p = center
   ny, nx = data.shape
   x = np.arange(nx) - x_p
   y = np.arange(ny) - y_p
   
   d = data.ravel()
   r = np.hypot.outer(y, x).astype(int).ravel()

   maxrange = np.max(r)
   med = [np.median(d[np.where(r == n)]) for n in np.arange(np.max(r))]

   return np.array(med)
