"""
Author: Mark Hannel
"""

import numpy as np
from pylorenzmie.detection.savgol2d import savgol2d
from pylorenzmie.detection.edge_convolve import edge_convolve
import random
from builtins import range

def voting(dadx, dady, dx, noise = 0.01, deinterlace=False,
               sample=1):
   """
   voting performs a voting algorithm based on low-noise gradients.
   """

   # Get the shape of the image.
   nx, ny = dadx.shape
   
   if noise is None:
      noise = mad(a)

   # Calculate the magnitude of the gradients.
   grada = np.sqrt(dadx**2 + dady**2)             # magnitude of the gradient
   dgrada = noise * np.sqrt(2. * sum(sum(dx**2))) # error in gradient 
                                                   # magnitude due to noise

   # Calculate where the gradient is significant.
   w = np.where(grada > 2.*dgrada)      # only consider votes with small 
                                        # angular uncertainty

   npts = len(w[0])
   b = np.zeros([nx, ny], dtype=int)      # accumulator array for the result

   if npts <= 0 : 
      return b

   yp, xp = np.mod(w,nx)         # coordinates of pixels with strong gradients

   if deinterlace: 
      yp = 2*yp + n0

   grada = grada[w[0],w[1]]      # gradient direction at each pixel
  
   dgrada = dgrada/grada
   costheta = dadx[w[0],w[1]] / grada
   sintheta = dady[w[0],w[1]] / grada

   rng = list(map(round,2./np.tan(dgrada/2.)))
   rng = np.array([ i if i < nx or i == -np.inf else nx for i in rng])
   mrange = int(max(rng))
   r = np.arange(2*mrange+1,dtype=float) - mrange

   nx -= 1
   ny -= 1

   if sample != 1:
      randInts = [random.randint(0,npts-1) for i in range(int(sample*npts))]
      xp = xp[randInts]
      yp = yp[randInts]
      rng = rng[randInts]
      costheta = costheta[randInts]
      sintheta = sintheta[randInts]

   npts = int(sample*npts)

   for i in range(npts):
      start = int(mrange-rng[i])
      end = int(mrange+rng[i])
      rr = r[start:end]
      x = (xp[i] + rr * costheta[i])
      x = [l if l > 0 else 0 for l in x]
      x = [m if m < nx else nx for m in x]

      y = (yp[i] + rr * sintheta[i])
      y = [l if l > 0 else 0 for l in y]
      y = [m if m < ny else ny for m in y]
      
      b[list(map(int,y)),list(map(int,x))] += 1

   # borders are over-counted because of > and <
   b[0, :] = 0
   b[:, 0] = 0
   b[:, -1] = 0
   b[-1, :] = 0

   return b

def orientTrans(dadx, dady, deinterlace=False):
   """
   The orientational transform
   """

   ny, nx = dadx.shape

   # orientational order parameter
   # psi = |\nabla a|**2 \exp(i 2 \theta)
   psi = dadx + 1.0j*dady ### FIX: May need to swap dadx, dady.
                          ### May also be faster not to use addition
   psi *= psi

   # Fourier transform of the orientational alignment kernel:
   # K(k) = e**(-2 i \theta) / k
   x_row = np.arange(nx)/float(nx) - 0.5
   y_col = np.arange(ny)/float(ny) - 0.5

   kx, ky = np.meshgrid(x_row,y_col)

   if deinterlace : 
      ky /= 2.

   k   = np.sqrt(kx**2 + ky**2) + 0.001
   ker = (kx - 1.0j*ky)**2 / k**3

   # convolve orientational order parameter with
   # orientational alignment kernel using
   # Fourier convolution theorem
   psi = np.fft.ifft2(psi)
   psi = np.fft.ifftshift(psi)
   psi *= ker
   psi = np.fft.fftshift(psi)
   psi = np.fft.fft2(psi)

   # intensity of convolution identifies rotationally
   # symmetric centers

   return np.real(psi*np.conj(psi))

def circletransform(a_, theory='orientTrans', noise=None, mrange=0, 
                    deinterlace=False, sample=1):
   """
   Performs a transform similar to a Hough transform for detecting circular 
   features in an image.

   Inputs:
   a: [nx, ny] greyscale image data

   Keyword Parameters:
   theory: Based on the gradients of the image, utilize a theory to expose
        circularly symmetric features.
   noise: estimate for additive pixel noise.
        Default: noise estimated by MAD().
   deinterlace: if set to an odd number, only perform transform on odd field 
        of an interlaced image. If set to an even number, transform even field.
        Default: Not set or set to zero: transform entire frame.
  
   Example:
   b = circletransform(a)
   """

   umsg = 'USAGE: b = circletransform(a)'

   if type(a_) != np.ndarray:
      print(umsg)
      print("a_ must be a numpy array")
      return -1

   if a_.ndim != 2:
      print(umsg)
      print('a_ must be a two-dimensional numeric array')
      return -1

   sz = a_.shape
   nx = sz[0]
   ny = sz[1]

   if type(mrange) != int or type(mrange) != float: 
      mrange = 100

   dodeinterlace = deinterlace if deinterlace > 0 else 0
   if dodeinterlace :
      n0 = deinterlace % 2
      a = a_[n0::2, :]
   else:
      a = a_

   dx = savgol2d(7, 3, dx = 1)
   
   dadx = -1 * edge_convolve(a, dx)       #FIXME:  Figure out why edge_convolve 
                                          #returns negative answer.
   dady = -1 * edge_convolve(a, np.transpose(dx))
   
   if dodeinterlace : 
      dady /= 2.    

   if theory == 'voting':
      return voting(dadx, dady, dx, noise=0.1, 
                        deinterlace=False, sample=0.8)

   if theory == 'orientTrans':
      return orientTrans(dadx, dady)

def test_circ():
   # Generate hologram.
   from pylorenzmie.theory import spheredhm as sph
   image = sph.spheredhm([100,0,100], 0.5, 1.5, 1.339, [641,481])

   # Circle transform the image.
   circ = circletransform(image, theory='orientTrans')

   import matplotlib.pyplot as plt
   plt.imshow(circ)
   plt.gray()
   plt.show()

if __name__ == '__main__':
   test_circ()
