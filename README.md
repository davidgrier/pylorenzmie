# pylorenzmie

<b>Python routines for tracking and characterizing colloidal particles
with in-line holographic video microscopy (HVM)</b>

## Explanation of the module
`lorenzmie` provides a set of routines for producing and analyzing holographic images of transparent dielectric spheres (pretty niche, huh?). An example image is given below:
<p align = "center">
<img src = "https://s15.postimg.org/5w1lokrob/hologram.png" />
</p>

The methods used to produce this image can be explained by the physical processes they are supposed to simulate:

1. Illumination in the form of polarized, coherent plane waves (laser light) strikes a micron-sized sphere (eg- a micron sized glass bead).
2. The illuminated sphere produces a scattered wave according to the Lorenz-Mie theory.
3. The plane waves and scattered wave interfere with each other at the camera plane, thus producing the above image.

What's special about these images? The image produced is very sensitive to the physical parameters of the sphere. For example, a slightly larger sphere might have made very different rings. It is this senstivity that makes these images so special.

For instance, suppose you have a sphere with _unknown_ physical properties. You can illuminate the sphere with a **known** laser source, record the scattering pattern, and **indirectly measure the physical properties of the sphere**. In particular, the radius, refractive index, and 3D position of the sphere can be measured. And because the electric fields at all points in the volume above the image can be approximately determined, we say that the image is holographic (ie - the 2D image contains all the information of a 3D volume). `lorenzmie` provides the means for fitting experimental holographic snapshots.

The methods provided here have been adopted, utilized and incrementally improved by David Grier's physics lab in the Center for Soft Matter research at NYU. For many of the exciting applications, see the references below.

## Author List:
Mark Hannel - Physics PhD Student at NYU University

## Where to get it.
[Mark Hannel's Github Repository](https://github.com/markhannel/lorenzmie)

## Coming Soon.
1.`svr` - Calculating holograms is computationally expensive. Instead, support vector machines can offer a less precise, yet computational cheap method for analyzing holographic images.

2.`debyewolf` - Current implementations of HVM assume the fields produced at the focal plane of an object are the same fields that arrive in the camera plane. `debyewolf` will utilize the Debye-Wolf integral (along with the abbe-sine condition and the conservation of energy) to account for the changes in amplitude, phase and direction imparted on the field present at the camera.

## Licensing.
[GPLv3](https://www.gnu.org/licenses/gpl-3.0.html)

## References:

1. S.-H. Lee, Y. Roichman, G.-R. Yi, S.-H. Kim, S.-M. Yang,
A. van Blaaderen, P. van Oostrum and D. G. Grier,
"Characterizing and tracking single colloidal particles with video
holographic microscopy," 
_Optics Express_ **15**, 18275-18282 (2007).

2. F. C. Cheong, B. Sun, R. Dreyfus, J. Amato-Grill, K. Xiao,
L. Dixon and D. G. Grier, "Flow visualization and flow cytometry with
holographic video microscopy," _Optics Express_ **17**
13071-13079 (2009).

3. F. C. Cheong, B. J. Krishnatreya and D. G. Grier,
"Strategies for three-dimensional particle tracking with
holographic video microscopy,"
_Optics Express_ **18**, 13563-13573 (2010).

4. H. Moyses, B. J. Krishnatreya and D. G. Grier,
_Optics Express_ **21** 5968-5973 (2013).

5. C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light
by Small Particles (New York, Wiley 1983).

6. W. Yang, "Improved recurstive algorithm for light scattering
by a multilayered sphere," _Applied Optics_ **42**, 1710--1720 (2003).

7. O. Pena and U. Pal, "Scattering of electromagnetic radiation
by a multilayered sphere," _Computer Physics Communications_
**180**, 2348-2354 (2009).

8. P. Messmer, P. J. Mullowney and B. E. Granger, 
"GPULib: GPU computing in high-level languages," 
_Computer Science and Engineering_ **10**, 70-73 (2008)
