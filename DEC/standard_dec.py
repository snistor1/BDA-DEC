import numpy as np

from copy import deepcopy
from sklearn.datasets import load_iris, make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from defines import *


def initialize(X, shape, domain, eps=10e-6, beta=BETA,
               dtype=np.float64):
    """
    Initializes the population for the CDE clustering algorithm.

    Args:
        shape (iterable of int):
            The shape of the population. It must contain 2 integers, the first
            one being the size of the population and the second one being the
            number of dimensions of the data points.
        domain (iterable of int):
            The domain in which the initial values are generated. The centroids
            are generated uniformly within this interval, while the variances
            are initialized uniformly in the half-opend interval
            (0, (b - a) / beta]. All components are bound by the same interval.
        eps (float):
            Floating-point value that represents the smallest tolerable error.
            It's used to create a half-open interval for generating the
            variance components.
        beta (float):
            Scaling factor for the size of the variance. Should be between 0
            and 2.
        dtype (np.dtype):
            Type of the entries in the population.

    Returns: np.ndarray
        The initialized CDE population, of shape (2, shape[0], shape[1]). The
        centroid matrix is placed at index 0 and the variance matrix at index
        1.
    """
    a, b = domain
    assert a < b, "domain invalid; a must be smaller than b"
    rng = np.random.default_rng()
    # centroids = a + (b - a) * rng.random(size=shape, dtype=dtype)
    centroids = X[rng.integers(X.shape[0], size=shape[0])]
    variances = eps + ((b - a) / beta) * rng.random(size=shape, dtype=dtype)
    return np.vstack((centroids[np.newaxis, ...],
                      variances[np.newaxis, ...]))


def get_crossover_indices(n, ranges=None):
    """
    Generates the distinct indices i, r1, r2 and r3 that are used in the
    crossover procedure.

    Args:
        n (int):
            The size of the population.
        ranges (np.ndarray):
            A 2D matrix used to generate the crossover indices, in which row
            i contains all numbers from 0 to n-1 except i. These represent the
            sample spaces of np.random.choice for each individual in the
            population.

    Returns: np.ndarray
        The n x 4 matrix that contains i on column 0 and r{1...3} on columns
        {1...3}.
    """
    assert n > 3, 'The population must have at least 4 individuals'
    rng = np.random.default_rng()
    rs = np.empty((n, 3), dtype=np.int64)
    if ranges is not None:
        computed_ranges = ranges
    else:
        computed_ranges = np.array([np.delete(np.arange(n), i)
                                    for i in range(n)])
    for i in range(n):
        rs[i, :] = rng.choice(computed_ranges[i], size=3, replace=False)
    return np.hstack((np.arange(n).reshape((-1, 1)), rs))


def search(pop, F=F_VAL, ranges=None):
    """
    The search operator shifts individuals in order to look for local
    optimum values. It mutates and performs crossover all-in-one.

    Args:
        pop (np.ndarray):
            3D array that represents our population of gaussians.
        F (float):
            Scaling factor for the influence of the differential in the
            value of the mutated individual.
        ranges (np.ndarray):
            A 2D matrix used to generate the crossover indices. See
            get_crossover_indices.

    Returns: np.ndarray
        The new population obtained by altering the original individuals.
    """
    pop_size = pop.shape[1]
    rng = np.random.default_rng()
    rs = get_crossover_indices(pop_size, ranges=ranges)
    pop_i, pop_r1, pop_r2, pop_r3 = (pop[:, rs[:, 0]],
                                     pop[:, rs[:, 1]],
                                     pop[:, rs[:, 2]],
                                     pop[:, rs[:, 3]])
    # See eq. (6) in distributed CDE paper and eq. (12) in the normal
    # CDE paper.
    new_pop = np.where(rng.random(size=pop.shape,
                                  dtype=np.float32) < CROSSOVER_P,
                       pop_r3 + F * (pop_r1 - pop_r2),
                       pop_i)
    # Make variances positive.
    new_pop[1] = np.abs(new_pop[1])
    return new_pop


def compute_distances(X, Y):
    """
    Computes the Mahalanobis distances between X and Y, for the special case
    where covariance between components is 0.

    Args:
        X (np.ndarray):
            3D array that represents our population of gaussians. It is
            assumed that X[0] is the 2D matrix containing the coordinates
            of the centroids and X[1] represents the 2D matrix of variances.
        Y (np.ndarray):
            2D or 3D array that can represent either a data matrix or a
            DE population. If it represents a population, only the centroids
            are taken into consideration.

    Returns: np.ndarray
        A matrix that contains all distances for each row of X to all rows
        of Y, computed with the variances found in X.
    """
    assert X.ndim == 3 and X.shape[0] == 2, \
        'X must have shape (2,_,_)'
    assert Y.ndim == 2 or (Y.ndim == 3 and Y.shape[0] == 2), \
        'Y must have shape (_,_) or (2,_,_)'
    m = X.shape[1]
    if Y.ndim == 2:
        n = Y.shape[0]
        points = Y
    else:
        n = Y.shape[1]
        points = Y[0]
    centers = X[0]
    sigmas = X[1]
    dist_matrix = np.empty((m, n), dtype=X.dtype)
    for i in range(m):
        # Broadcasting
        diff = (centers[i] - points) / sigmas[i]
        # This computes the sum of the pairwise products of the rows. In other
        # words, it computes sum([x[i] * y[i] for i in range(x.shape[0])]).
        dist_matrix[i, :] = np.einsum('ij,ij->i', diff, diff)
    return dist_matrix


def classify_data(repr_set, data):
    cluster_result = np.zeros(shape=(data.shape[0],))
    distances = compute_distances(repr_set[:2, :, :], data)
    distances = np.swapaxes(distances, 0, 1)
    min_idx, min_values = distances.argmin(axis=1), distances.min(axis=1)
    cluster_mask = min_values <= DELTA_3
    clusters = min_idx[cluster_mask]
    cluster_result[cluster_mask] = repr_set[2][clusters, 0]

    # Clusters refining below
    if distances.shape[1] > 1:  # if we have at least 2 representatives for each input, we can try to refine them
        cluster_min_size = data.shape[0] / (4 * len(np.unique(repr_set[2][:, 0])))
        unique, counts = np.unique(cluster_result, return_counts=True)
        outlier_mask = unique == 0
        unique, counts = unique[~outlier_mask], counts[~outlier_mask]
        small_clusters = unique[counts < cluster_min_size]
        sorted_distances = np.sort(distances, axis=1)
        for i in range(len(cluster_result)):
            if cluster_result[i] in small_clusters and sorted_distances[i][1] <= DELTA_3:
                cluster_result[i] = repr_set[2][1, 0]
    return cluster_result


def compute_repr_distance(x, y, sigma):
    diff = (x - y) / sigma
    return np.dot(diff, diff)


def collect_repr(pop, data):
    centers = pop[0]
    sigmas = pop[1]
    repr_set = np.empty(shape=(3, pop.shape[1], pop.shape[2]))
    repr_unique = {(tuple(centers[0]), tuple(sigmas[0]), 1)}
    repr_set[0][0] = centers[0]
    repr_set[1][0] = sigmas[0]
    repr_set[2][0] = 1
    nr_repr = 1
    next_label = 2
    for i in range(1, len(centers)):
        distances = [compute_repr_distance(centers[i], repr_set[0][j], repr_set[1][j]) for j in range(nr_repr)]
        closest = np.argmin(distances, axis=0)
        if type(closest) is np.ndarray:
            closest = closest[0]
        x_distance = compute_repr_distance(centers[i], repr_set[0][closest], sigmas[i])
        if distances[closest] <= DELTA_1 or x_distance <= DELTA_1:
            new_repr = ((centers[i] + repr_set[0][closest]) / 2,
                        np.maximum(sigmas[i], repr_set[1][closest]),
                        next_label)
            if (tuple(new_repr[0]), tuple(new_repr[1]), new_repr[2]) not in repr_unique:
                repr_set[0][nr_repr] = new_repr[0]
                repr_set[1][nr_repr] = new_repr[1]
                repr_set[2][nr_repr] = new_repr[2]
                repr_unique.add((tuple(new_repr[0]), tuple(new_repr[1]), new_repr[2]))
                next_label += 1
                nr_repr += 1
        elif distances[closest] > DELTA_2 and x_distance > DELTA_2:
            if (tuple(centers[i]), tuple(sigmas[i]), next_label) not in repr_unique:
                repr_set[0][nr_repr] = centers[i]
                repr_set[1][nr_repr] = sigmas[i]
                repr_set[2][nr_repr] = next_label
                repr_unique.add((tuple(centers[i]), tuple(sigmas[i]), next_label))
                next_label += 1
                nr_repr += 1
        else:
            if (tuple(centers[i]), tuple(sigmas[i]), repr_set[2][closest, 0]) not in repr_unique:
                repr_set[0][nr_repr] = centers[i]
                repr_set[1][nr_repr] = sigmas[i]
                repr_set[2][nr_repr] = repr_set[2][closest]
                repr_unique.add((tuple(centers[i]), tuple(sigmas[i]), repr_set[2][closest, 0]))
                nr_repr += 1
    repr_set = repr_set[:, :nr_repr, :]

    # Refinement process below
    fitness = get_fitness(repr_set[:2, :, :], data)
    for _ in range(nr_repr):
        change = False
        for i in range(nr_repr - 1):
            for j in range(1, nr_repr):
                if tuple(repr_set[2][i]) != tuple(repr_set[2][j]) and \
                        (compute_repr_distance(repr_set[0][i], repr_set[0][j], repr_set[1][i]) <= DELTA_2 or
                         compute_repr_distance(repr_set[0][i], repr_set[0][j], repr_set[1][j]) <= DELTA_2):
                    change = True
                    if fitness[i] > fitness[j]:
                        repr_set[2][j] = deepcopy(repr_set[2][i])
                    else:
                        repr_set[2][i] = deepcopy(repr_set[2][j])
        if not change:
            break

    clustered_result = classify_data(repr_set, data)
    return clustered_result


def get_fitness(population, data):
    """
    Function that computes fitness values (a.k.a g(c, sigma)) for the
    population relative to the provided data.

    Args:
        population (np.ndarray):
            3D array that represents our population of gaussians.
        data (np.ndarray):
            2D array of data points.

    Returns: np.ndarray
        The fitness values of each individual in the population.
    """
    pop_size = population.shape[1]
    data_size = data.shape[0]
    sigmas = population[1]
    # We need the squared distances in order to compute the fitness score.
    distances = compute_distances(population, data)
    # You should check the correctness of the following formulas. Many
    # mistakes can be made here!
    norm_factors = (1 / np.sqrt(np.prod(sigmas,
                                        axis=1))) / pop_size
    # np.dot(x, np.ones) is equivalent to np.sum(x, axis=1), but faster.
    return norm_factors * np.dot(np.exp(-0.5 * distances),
                                 np.ones(data_size))


def differential_clustering(X, n_iter, crowding=True):
    """
    CDE-based clustering algorithm with gaussian mixtures.

    Args:
        X (np.ndarray):
            2D array of normalized data points. These are the data points
            that we need to cluster.
        n_iter (int):
            Number of iterations for the clustering algorithm.
        crowding (bool):
            This flag distinguishes between normal DE and crowding DE (CDE).
            Default value is True (False branch hasn't been implemented).
    """
    # This population is only for testing purposes.
    pop = initialize(X, (POP_SIZE, X.shape[1]), [0.0, 0.5])
    # Precompute ranges that are needed for selecting indices distinct from
    # i, for all i in {0...m-1}. It's an implementation detail, but it reduces
    # execution time to initialize it only once here.
    precomputed_ranges = np.array([np.delete(np.arange(pop.shape[1]), i)
                                   for i in range(pop.shape[1])])
    if __debug__:
        assert X.shape[1] == 2, "Can only plot datasets with exactly 2" \
            " dimensions"
        plt.ion()
        fig, ax = plt.subplots()
        data_scatter = ax.scatter(X[:, 0], X[:, 1], c='green')
        clusters = []
        for i in range(pop.shape[1]):
            ell = Ellipse((pop[0, i, 0], pop[0, i, 1]), pop[1, i, 0], pop[1, i, 1],
                          color='red', alpha=0.2)
            clusters.append(ell)
            ax.add_patch(ell)
        # cluster_scatter = ax.scatter(pop[0, :, 0], pop[0, :, 1], c='red')
        plt.draw()
    for i in range(n_iter):
        # Construct new (c_i, sigma_i).
        print(f'Iteration: {i + 1}')
        new_pop = search(pop, ranges=precomputed_ranges)
        if crowding:
            # Find the element (c, sigma) most similar to (c_i, sigma_i).
            dist_between_pops = compute_distances(new_pop, pop)
            most_similar_idxs = np.argmin(dist_between_pops, axis=1)
            # Since some clusters can be the closest to more than 1 point and
            # we don't want to recompute fitness values needlessly, we will
            # only compute fitness for the unique values. We keep the inverse
            # mapping in order to transform our unique values back to the
            # original indices which included the repeated values.
            unique_idxs, inv_map = np.unique(most_similar_idxs,
                                             return_inverse=True)
            # These are the individuals that we want to replace.
            replacement_candidates = pop[:, unique_idxs, :]
            old_fitness, new_fitness = (get_fitness(replacement_candidates, X),
                                        get_fitness(new_pop, X))
            # Remap unique values to original array.
            old_fitness = old_fitness[inv_map]
            # Replace individuals where g(c_i, sigma_i) > g(c, sigma).
            replace_mask = new_fitness > old_fitness
            pop[:, replace_mask, :] = new_pop[:, replace_mask, :]
        else:
            raise NotImplementedError("TODO: Implement without crowding")
        if __debug__:
            for i in range(len(clusters)):
                clusters[i].set_center((pop[0, i, 0], pop[0, i, 1]))
                clusters[i].width = pop[1, i, 0]
                clusters[i].height = pop[1, i, 1]
            # cluster_scatter.set_offsets(np.c_[pop[0, :, 0], pop[0, :, 1]])
            fig.canvas.draw_idle()
            plt.pause(0.05)
    if __debug__:
        plt.waitforbuttonpress()
    # clustered_result = collect_repr(pop, X)
    # print(clustered_result)
    # TODO: make some plots here or something, anything


def main():
    scaler = StandardScaler()
    # X, y = load_iris(return_X_y=True)
    # scaler.fit(X[:, :2])
    # differential_clustering(scaler.transform(X[:, :2]), N_ITER)
    X, y = make_blobs(200, 2)
    scaler.fit(X)
    differential_clustering(scaler.transform(X), N_ITER)


if __name__ == '__main__':
    main()
