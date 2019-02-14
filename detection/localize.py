'''Module for localizing particle trajectories with tensorflow tracking.'''

from pylorenzmie.detection.h5video import TagArray
from pylorenzmie.detection.circletransform import circletransform
import trackpy as tp
import numpy as np


def localize(image, frame_no=None,
             locate_params={'diameter': 31,
                            'minmass': 30.},
             nfringes=25,
             maxrange=400.,
             crop_threshold=None):
    '''
    Localize features in image using circletransform
    and trackpy.locate
    on every pixel of an image and return features.
    
    Args:
        image: normalized image with median near 1.
    '''
    circ = circletransform(image, theory='orientTrans')
    circ = circ / np.amax(circ)
    circ = TagArray(circ, frame_no=frame_no)
    feats = tp.locate(circ,
                      engine='numba',
                      **locate_params)
    feats['w'] = 400.
    feats['h'] = 400.
    features = np.array(feats[['x', 'y', 'w', 'h']])
    for idx, feature in enumerate(features):
        s = feature_extent(image, (feature[0], feature[1]),
                           nfringes=nfringes,
                           maxrange=maxrange)
        if crop_threshold is not None and s > crop_threshold:
            s = crop_threshold
        features[idx][2] = s
        features[idx][3] = s
    return features, circ


def feature_extent(norm, center, nfringes=20, maxrange=400.):
    ravg, rstd = aziavg(norm, center)
    b = ravg - 1.
    ndx = np.where(np.diff(np.sign(b)))[0] + 1.
    if len(ndx) <= nfringes:
        return maxrange
    else:
        return float(ndx[nfringes])


def aziavg(data, center):
    x_p, y_p = center
    y, x = np.indices((data.shape))
    d = data.ravel()
    r = np.hypot(x - x_p, y - y_p).astype(np.int).ravel()
    nr = np.bincount(r)
    ravg = np.bincount(r, d) / nr
    avg = ravg[r]
    rstd = np.sqrt(np.bincount(r, (d - avg)**2) / nr)
    return ravg, rstd


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from pylorenzmie.theory.LMHologram import LMHologram
    from pylorenzmie.theory.Instrument import coordinates
    shape = [201, 251]
    h = LMHologram(coordinates=coordinates(shape))
    h.particle.r_p = [125, 75, 100]
    h.particle.a_p = 0.9
    h.particle.n_p = 1.45
    h.instrument.wavelength = 0.447
    data = h.hologram().reshape(shape)
    features, circ = localize(data, nfringes=30)
    fig, ax = plt.subplots()
    ax.imshow(circ, cmap='gray')
    for feature in features:
        x, y, w, h = feature
        rect = Rectangle(xy=(x - w/2, y - h/2), width=w, height=h,
                         fill=False, linewidth=3, edgecolor='r')
        ax.add_patch(rect)
    plt.show()
    
