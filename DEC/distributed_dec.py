from functools import partial
from math import sqrt
import os
import sys
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, DenseVector, VectorUDT
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import SparkSession, Window, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, StructType, StructField, NullType
# from sklearn.preprocessing import StandardScaler

N_ITER = 5
POP_SIZE = 40
BETA = 3.0
F_VAL = 0.3
CROSSOVER_P = 0.2
DELTA_1 = 0.01
DELTA_2 = 0.05
DELTA_3 = 0.9

N_PARTITIONS = 4
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = f'{DIR_PATH}/../data'
RESULT_DIR = f'{DIR_PATH}/result'
spark = SparkSession.builder \
                    .appName("Differential Evolution Clustering") \
                    .getOrCreate()


def initialize(X, n_pop, eps=10e-6, beta=BETA, dtype=np.float64, smart_init=False):
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
    a, b = np.min(X, axis=0), np.max(X, axis=0)
    assert (a < b).all(), "domain invalid; a must be smaller than b"
    rng = np.random.default_rng()
    if not smart_init:
        centroids = X[rng.integers(X.shape[0], size=n_pop)]
    else:
        centroids = np.zeros(shape=(n_pop, X[0].shape[0]))
        # the first centroid will be a randomly selected data point
        centroids[0] = X[rng.integers(X.shape[0], size=1)]
        for k in range(1, n_pop):
            max_dist, max_p = 0, 0
            for i in range(X.shape[0]):
                point = X[i, :]
                min_dist, min_p = 999, 0
                # compute distance of 'point' from each of the previously selected centroid
                # and store the minimum distance
                for j in range(k):
                    dist = np.linalg.norm(point - centroids[j])
                    if dist < min_dist:
                        min_dist, min_p = dist, i
                if min_dist > max_dist:
                    max_dist, max_p = min_dist, min_p
            # select data point with maximum distance as our next centroid
            centroids[k] = X[max_p, :]

    # centroids = a + (b - a) * rng.random(size=shape, dtype=dtype)
    variances = eps + rng.random(size=(n_pop, X.shape[1])) * ((b - a) / beta)
    # variances = eps + ((b - a) / beta) * rng.random(size=shape, dtype=dtype)
    return np.vstack((centroids[np.newaxis, ...],
                      variances[np.newaxis, ...]))


@F.udf(returnType=DoubleType())
def mahalanobis(c1 : DenseVector, c2 : DenseVector, sigma1 : DenseVector):
    c1_arr, c2_arr, sigma1_arr = c1.toArray(), c2.toArray(), sigma1.toArray()
    diff = ((c1_arr - c2_arr) / sigma1_arr)
    return float(np.dot(diff, diff))


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


def search(pop_df, F=F_VAL):
    """
    The search operator shifts individuals in order to look for local
    optimum values. It mutates and performs crossover all-in-one.
    """
    pop = np.array(pop_df.select("centroids", "variances").collect())
    local_ids = np.array(pop_df.select("id").collect())
    pop_size = pop.shape[0]
    rng = np.random.default_rng()
    rs = get_crossover_indices(pop_size)
    pop_i, pop_r1, pop_r2, pop_r3 = (pop[rs[:, 0]],
                                     pop[rs[:, 1]],
                                     pop[rs[:, 2]],
                                     pop[rs[:, 3]])
    # See eq. (6) in distributed CDE paper and eq. (12) in the normal
    # CDE paper.
    new_pop = np.where(rng.random(size=pop.shape,
                                  dtype=np.float32) < CROSSOVER_P,
                       pop_r3 + F * (pop_r1 - pop_r2),
                       pop_i)
    # Make variances positive.
    centroids, variances = np.swapaxes(new_pop, 0, 1)
    variances = np.abs(variances)
    pop_pandas = pd.DataFrame(np.hstack((centroids, variances)))
    pop_pandas['id'] = local_ids
    pop_pandas = pop_pandas[['id'] + [col for col in pop_pandas.columns
                                      if col != 'id']]
    new_pop_df = spark.createDataFrame(pop_pandas)
    midpoint = (len(new_pop_df.columns) - 1) // 2
    new_pop_columns = new_pop_df.columns[1:]
    assembler = VectorAssembler(inputCols=new_pop_columns[:midpoint],
                                outputCol="centroids")
    new_pop_df = assembler.transform(new_pop_df)
    assembler = VectorAssembler(inputCols=new_pop_columns[midpoint:],
                                outputCol="variances")
    new_pop_df = assembler.transform(new_pop_df) \
        .select("id", "centroids", "variances") \
        .repartition(N_PARTITIONS, "id")
    return new_pop_df


def compute_distances(X: DataFrame, Y: DataFrame) -> DataFrame:
    m, n = len(X.columns), len(Y.columns)
    assert m == 3 and n in [1, 3]
    if n == 1:
        distances_df = X.withColumnRenamed("centroids", "c1") \
                        .withColumnRenamed("variances", "s1") \
                        .crossJoin(Y).withColumnRenamed("points", "c2")
    else:
        distances_df = X.withColumnRenamed("centroids", "c1") \
                        .withColumnRenamed("variances", "s1") \
                        .withColumnRenamed("id", "id1") \
                        .crossJoin(Y.select(F.col("id").alias("id2"),
                                            F.col("centroids").alias("c2"),
                                            F.col("variances").alias("s2")))
    distances_df = distances_df.withColumn("distance",
                                           mahalanobis(F.col("c1"),
                                                       F.col("c2"),
                                                       F.col("s1")))
    return distances_df


@udf(returnType=DoubleType())
def norm(sigma: DenseVector):
    return float((1 / np.sqrt(np.prod(sigma.toArray()))) / POP_SIZE)


def get_fitness(population: DataFrame, data: DataFrame) -> DataFrame:
    # We need the squared distances in order to compute the fitness score.
    distances_df = compute_distances(population, data)
    fitness_df = distances_df.withColumn("norm_factors",
                                         norm(F.col("s1"))) \
                             .selectExpr("id",
                                         "norm_factors * EXP(-0.5 * distance) AS fitness") \
                             .groupBy("id") \
                             .agg(F.sum("fitness").alias("fitness")) \
                             .withColumnRenamed("id", "id_fitness")
    return fitness_df


def compute_repr_distance(x, y, sigma):
    diff = (x - y) / sigma
    return np.dot(diff, diff)


def collect_repr(pop, data=None):
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
        distances = np.array([compute_repr_distance(centers[i], repr_set[0][j], repr_set[1][j]) for j in range(nr_repr)])
        distances = np.divide(distances, np.sum(distances))
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

    if data is not None:
        # Refinement process below
        fitness = get_fitness(repr_set[:2, :, :], data)
        for _ in range(nr_repr):
            change = False
            for i in range(nr_repr - 1):
                for j in range(i + 1, nr_repr):
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
    return repr_set


def compute_local_distances(X, Y):
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
    distances = compute_local_distances(repr_set[:2, :, :], data)
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
    values = np.unique(cluster_result)
    for i in range(len(values)):
        cluster_result[cluster_result == values[i]] = i + 1
    return cluster_result


def differential_clustering_distributed(X):
    N_DIMS = np.array(X.take(1)).size
    # Generate population.
    sample_X = np.array(X.sample(False, 0.05).collect()).reshape(-1, 2)
    centroids, variances = initialize(sample_X, POP_SIZE)
    arr = map(lambda x, y: (Vectors.dense(x), Vectors.dense(y)),
              centroids, variances)
    population_df = spark.createDataFrame(arr, ["centroids", "variances"]) \
        .withColumn("id", F.monotonically_increasing_id()) \
        .select("id", "centroids", "variances")
    population_df = population_df \
        .repartition(N_PARTITIONS, "id")
    # population_df.show()
    population_df = population_df.cache()
    for i in range(N_ITER):
        print(f"Iteration: {i + 1}")
        # Search for better candidates.
        new_population_df = search(population_df)
        fitness_df = get_fitness(population_df, X)
        pop_distance_df = compute_distances(population_df, new_population_df)
        w = Window.partitionBy("id1")
        min_distances_df = pop_distance_df.withColumn("min_distance", F.min("distance").over(w)) \
                                          .where(F.col("distance") == F.col("min_distance")) \
                                          .drop("min_distance")
        min_distances_df = min_distances_df.cache()
        w = Window.partitionBy("id1")
        min_distances_df = min_distances_df \
            .withColumn("first_id2", F.first("id2").over(w)) \
            .where(F.col("id2") == F.col("first_id2")) \
            .drop("first_id2")
        min_distances_df = min_distances_df \
            .join(fitness_df.withColumnRenamed("fitness", "fitness1"),
                  on=(min_distances_df.id1 == fitness_df.id_fitness)) \
            .drop("id_fitness") \
            .join(fitness_df.withColumnRenamed("fitness", "fitness2"),
                  on=(min_distances_df.id2 == fitness_df.id_fitness)) \
            .drop("id_fitness")
        population_df = min_distances_df \
            .withColumn("final_c",
                        F.when(F.col("fitness1") > F.col("fitness2"), F.col("c1")).otherwise(F.col("c2"))) \
            .withColumn("final_s",
                        F.when(F.col("fitness1") > F.col("fitness2"), F.col("s1")).otherwise(F.col("s2"))) \
            .select("id1", "final_c", "final_s") \
            .withColumnRenamed("id1", "id") \
            .withColumnRenamed("final_c", "centroids") \
            .withColumnRenamed("final_s", "variances")
        population_df = population_df.repartition(N_PARTITIONS, "id")

    local_centroids = np.array(population_df.select("centroids").collect()) \
        .reshape(-1, N_DIMS)
    local_variances = np.array(population_df.select("variances").collect()) \
        .reshape(-1, N_DIMS)
    local_pop = np.array([local_centroids, local_variances])
    print("Collecting global representatives...")
    reprs = collect_repr(local_pop)
    X_collect = np.array(X.select("points").collect()) \
        .reshape(-1, N_DIMS)
    y_pred = classify_data(reprs, X_collect)
    return y_pred
    # ...or...
    # Convert representatives to DataFrame
    # <...>
    # y_pred = classify_data(reprs, X)


def load_dataset(filename: str, delimiter=' '):
    # spark = SparkSession.builder \
    #                     .appName("Differential Evolution Clustering") \
    #                     .getOrCreate()
    data = spark.read \
                .format("csv") \
                .option("inferSchema", "true") \
                .option("header", "false") \
                .option("delimiter", delimiter) \
                .load(os.path.join(DATA_DIR, filename))
    label_column = data.schema[len(data.columns) - 1].name
    X = data.drop(label_column).repartition(N_PARTITIONS).cache()
    y = np.array(data.select(label_column).collect()) \
        .reshape(-1)
    # Preprocess data.
    unlist = F.udf(lambda x: float(list(x)[0]), DoubleType())
    for col_name in X.columns:
        # Convert column to vector type.
        assembler = VectorAssembler(inputCols=[col_name],
                                    outputCol=col_name + "_Vect")
        # Apply MinMaxScaler.
        scaler = MinMaxScaler(inputCol=col_name + "_Vect",
                              outputCol=col_name + "_Scaled")
        # Create pipeline.
        pipeline = Pipeline(stages=[assembler, scaler])
        # Fit pipeline on dataframe.
        X = pipeline.fit(X).transform(X) \
                           .withColumn(col_name + "_Scaled",
                                       unlist(col_name + "_Scaled")) \
                           .drop(col_name + "_Vect", col_name) \
                           .withColumnRenamed(col_name + "_Scaled", col_name)
    assembler = VectorAssembler(inputCols=X.columns,
                                outputCol="points")
    X = assembler.transform(X).select("points")
    return X, y


def main():
    X, y = load_dataset("2d-10c.dat", delimiter=' ')
    y_pred = differential_clustering_distributed(X)
    print(y_pred)


if __name__ == '__main__':
    main()
