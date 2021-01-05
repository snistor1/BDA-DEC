import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from scipy import linalg
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from DEC.standard_dec import load_custom_data

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = f'{DIR_PATH}/../data'
RESULT_DIR = f'{DIR_PATH}/result'


color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def get_data(input_file):
    return load_custom_data(os.path.join(DATA_DIR, input_file))


def plot_results(X, Y_, means, covariances, title):
    splot: Axes = plt.subplot(1, 1, 1)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        if not np.any(Y_ == i):
            continue
        plt.scatter(X[0][Y_ == i], X[1][Y_ == i], .8, color=color)

        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(X[0].min(), X[0].max())
    plt.ylim(X[1].min(), X[1].max())
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.savefig(os.path.join(RESULT_DIR, 'clusters_EM.png'))


def gaussian_mixture(X, n_clusters):
    gm = GaussianMixture(n_clusters).fit(X)
    return gm


def main():
    X, Y = get_data('2d-10c.dat')
    gm: GaussianMixture = gaussian_mixture(X, Y)
    predictions = gm.predict(X)
    plot_results(X, predictions, gm.means_, gm.covariances_, 'Gaussian Mixture')

    print("Homogeneity score - EM:", metrics.homogeneity_score(Y, predictions))
    print("Completeness score - EM:", metrics.completeness_score(Y, predictions))
    print("V score - EM:", metrics.completeness_score(Y, predictions))
    print("ARI - EM:", adjusted_rand_score(Y, predictions))

    plt.show()


if __name__ == '__main__':
    main()
