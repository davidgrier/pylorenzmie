# pylorenzmie

Python routines for tracking and characterizing colloidal particles
with in-line holographic video microscopy (HVM)

## Explanation of the module
`pylorenzmie` provides a set of python classes for interacting with and
analyzing holographic microscopy data. The hologram of a colloidal
particle encodes comprehensive information about the particle's size,
composition, and location in three dimensions. This package extracts
that information by fitting a recorded hologram to a generative model
based on the Lorenz-Mie theory of light scattering.

<img src="docs/tutorials/crop.png" alt="Typical Hologram" width="200"/>

## Authors
David G. Grier (New York University), Lauren Altman, Fook Chiong Cheong, 
Mark D. Hannel, Sanghyuk Lee, Michael O'Brien, Kaitlynn Snyder, Jackie Sustiel

## Licensing.
[GPLv3](https://www.gnu.org/licenses/gpl-3.0.html)

## References:
### Lorenz-Mie analysis of colloidal particles
1. S.-H. Lee, Y. Roichman, G.-R. Yi, S.-H. Kim, S.-M. Yang,
   A. van Blaaderen, P. van Oostrum and D. G. Grier,
   "Characterizing and tracking single colloidal particles with video
   holographic microscopy," 
   _Optics Express_ **15**, 18275-18282 (2007).

### Lorenz-Mie theory of light scattering
1. C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light
   by Small Particles (Wiley 1983).
1. M. I. Mishchenko, L. D. Travis and A. A. Lacis, Scattering
   Absorption and Emission of Light by Small Particles (Cambridge
   University Press, 2002).
1. G. Gouesbet and G. Gréhan, Generalized Lorenz-Mie Theories
   (Springer, 2011).

### Computational methods
1. W. Yang, "Improved recurstive algorithm for light scattering
   by a multilayered sphere," _Applied Optics_ **42**, 1710--1720 (2003).
1. O. Pena and U. Pal, "Scattering of electromagnetic radiation
   by a multilayered sphere," _Computer Physics Communications_
   **180**, 2348-2354 (2009).
1. W. J. Wiscombe, "Improved Mie scattering algorithms,"
   _Applied Optics_ **19**, 1505-1509 (1980).
1. A. A. R. Neves and D. Pisignano, "Effect of finite terms on the
   truncation error of Mie series," _Optics Letters_ **37**,
   2481-2420 (2012).
