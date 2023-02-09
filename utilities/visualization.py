from matplotlib import pyplot as plt
import numpy as np


def report(result):
    def value(val, err, dec=2):
        fmt = '{' + ':.{}f'.format(dec) + '}'
        return (fmt + ' +- ' + fmt).format(val, err)
    res = ['x_p = ' + value(result.x_p, result.dx_p) + ' [pixels]',
           'y_p = ' + value(result.y_p, result.dy_p) + ' [pixels]',
           'z_p = ' + value(result.z_p, result.dz_p) + ' [pixels]',
           'a_p = ' + value(result.a_p, result.da_p, 3) + ' [um]',
           'n_p = ' + value(result.n_p, result.dn_p, 4)]
    print('npixels = {}'.format(result.npix))
    print(*res, sep='\n')
    print('chisq = {:.2f}'.format(result.redchi))


def present(feature):
    fig, axes = plt.subplots(ncols=3,
                             figsize=(10, 4),
                             constrained_layout=True)

    vmin = np.min(feature.data) * 0.9
    vmax = np.max(feature.data) * 1.1
    style = dict(vmin=vmin, vmax=vmax, cmap='gray')

    images = [feature.data,
              feature.hologram(),
              feature.residuals()+1]
    labels = ['Data', 'Fit', 'Residuals']

    for ax, image, label in zip(axes, images, labels):
        ax.imshow(image, **style)
        ax.axis('off')
        ax.set_title(label)
    plt.show()
