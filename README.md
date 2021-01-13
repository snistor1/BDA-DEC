# Clustering with Differential Evolution in PySpark
This is an implementation of a clustering algorithm with crowding mechanism that utilizes differential evolution as a meta-heuristic for finding the optimal partitioning of the data. Two versions have been created: a sequential version that executes on a single computer and a distributed version that splits the dataset and the population among many computational devices and executes in parallel.

DEC\standard_dec.py - The sequential version of the Clustering with Differential Evolution (CDE) algorithm.
DEC\distributed_dec.py - The distributed version of the CDE algorithm.

## Requirements
The implemented scripts are compatible only with the Python-3 version and the following packages are necessary:
* matplotlib==3.3.3
* numpy==1.19.4
* pandas==1.2.0
* pyspark==3.0.1
* scikit_learn==0.24.0
* scipy==1.3.3
* findspark (needed on google cloud)

## Single run usage
    python DEC\standard_dec.py
    python DEC\distributed_dec.py
    python EM\expectation_maximization
The algorithm hyperparameters and certain settings (number of desired partitions, path to the data and output folders) can be changed directly in these scripts. 
The filename of the dataset must be specified in the script.

## Experiments running
For a comparative analysis between DEC algorithms and EM, there are 2 notebooks:
* `Comparative analysis large dataset.ipynb`
* `Comparative analysis small dataset.ipynb`
Each of them plots the resulted clusters, prints the metrics achieved by each algorithm and creates a time comparison plot.

## Individual contribution of each team member
* Ghiga Claudiu:
    1. Implementation of Distributed DEC components
    2. Memory and speed improvements to both DEC versions
    3. Documentation for certain script methods
    4. Support for real-time visualisation of DEC results while running the algorithm
* Luncasu Bogdan: 
    1. Comparative analysis with EM on a small dataset and a big dataset for each implementation of DEC
    2. Various quality tests to ensure that the distributed version and the non-distributed one works as expected
    3. Experiments in order to find good hyperparameters for DEC algorithms
    4. Notebook support configuration on google cloud
    5. Project structure adjustments for making it easier to run a comparative analysis
* Nistor Serban:
    1. Implementation of Standard DEC components
    2. Overall project structuring and refactorizations
    3. Research on improvements and K-Means++ initialization method for Standard DEC
    4. A method to compute the best DELTA_x parameters on a given dataset
    5. Google cloud project deployment