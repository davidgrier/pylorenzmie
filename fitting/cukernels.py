import cupy as cp


curesiduals = cp.ElementwiseKernel(
    'float64 holo, float64 data, float64 noise',
    'float64 residuals',
    'residuals = (holo - data) / noise',
    'curesiduals')

cuchisqr = cp.ReductionKernel(
    'float64 holo, float64 data, float64 noise',
    'float64 chisqr',
    '((holo - data) / noise) * ((holo - data) / noise)',
    'a + b',
    'chisqr = a',
    '0',
    'cuchisqr')

cuabsolute = cp.ReductionKernel(
    'float64 holo, float64 data, float64 noise',
    'float64 s',
    'abs((holo - data) / noise)',
    'a + b',
    's = a',
    '0',
    'cuabsolute')


curesidualsf = cp.ElementwiseKernel(
    'float32 holo, float32 data, float32 noise',
    'float32 residuals',
    'residuals = (holo - data) / noise',
    'curesiduals')

cuchisqrf = cp.ReductionKernel(
    'float32 holo, float32 data, float32 noise',
    'float32 chisqr',
    '((holo - data) / noise) * ((holo - data) / noise)',
    'a + b',
    'chisqr = a',
    '0',
    'cuchisqr')

cuabsolutef = cp.ReductionKernel(
    'float32 holo, float32 data, float32 noise',
    'float32 s',
    'abs((holo - data) / noise)',
    'a + b',
    's = a',
    '0',
    'cuabsolute')
