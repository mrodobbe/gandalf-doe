from GPyOpt.experiment_design import initial_design
import numpy as np
from gandalf_doe.domain import Domain
from gandalf_doe.clustering.initialization_cluster import *
from numpy.typing import NDArray
from typing import Union, Any


class Pool:

    def __init__(self, domain: Domain, pool: Union[None, NDArray[Any]] = None, pool_size: int = None):
        self.space = domain.design_space
        self.lower_bound = domain.min_values
        self.upper_bound = domain.max_values

        if pool is None and pool_size is not None:
            self.dimension = pool_size
            self.initialize_pool()
        elif pool is not None:
            self.pool = pool
            self.dimension = self.pool.shape[0]
        else:
            raise KeyError("No pool nor a pool size is defined.")

    def initialize_pool(self, pool_size: int = None):
        if pool_size is not None:
            self.dimension = pool_size

        self.pool = initial_design('random', self.space, self.dimension)

    def normalize(self):
        self.pool = (self.pool - self.lower_bound) / (self.upper_bound - self.lower_bound)

    def scale(self, length_scale):
        pass

    def descale(self, length_scale):
        pass

    def denormalize(self):
        self.pool = self.pool * (self.upper_bound - self.lower_bound) + self.lower_bound

    def add_experiments(self, x):
        self.pool = np.vstack((self.pool, x))


def initialize_cluster(domain: Domain, pool: Pool, n_init, objective=None, normalize=True, scale=False):

    cluster_init = KMeans(n_clusters=n_init).fit(pool.pool)
    x_init = np.empty((n_init, pool.pool.shape[1]))

    """
    x_init_ctu = cluster_init.cluster_centers_
    x_init = discrete_constraints_8d(x_init_ctu)
    """

    for k in range(n_init):
        var = cluster_init.transform(pool.pool)[:, k]
        x_init[k] = pool.pool[np.argmin(var)]

    if normalize:
        x_init = denormalizer(x_init, domain.min_values, domain.max_values)
    elif scale:
        x_init = descale(x_init, None)

    if objective is not None:
        y_init = np.empty((x_init.shape[0], 1))
        for j in range(x_init.shape[0]):
            y_init[j] = objective(np.array([x_init[j]]))
        return x_init, y_init

    else:
        return x_init, None


def draw_cluster_from_pool(pool: Pool, n: int):
    query_cluster = 0

    while query_cluster == 0:
        cluster = KMeans(n_clusters=n + 1).fit(pool.pool)
        print(cluster)
        new_cluster = select_clusters(cluster, n)
        query_cluster = largest_empty_cluster(new_cluster)

    x_cluster = elements_cluster(cluster, query_cluster, pool.pool)

    return x_cluster, query_cluster


def draw_multiple_clusters_from_pool(pool: Pool, n_cluster: int, n: int):
    query_clusters = [0, 0, 0]

    cluster = KMeans(n_clusters=n_cluster + 1).fit(pool.pool)
    print("cluster", cluster)
    new_cluster = select_clusters(cluster, n_cluster)
    query_clusters = get_multiple_clusters(new_cluster, n)

    x_clusters = []
    q_clusters = []
    for x in query_clusters:
        x_cluster = elements_cluster(cluster, x, pool.pool)
        if len(x_cluster) > 0:
            x_clusters.append(x_cluster)
            q_clusters.append(x)
    x_clusters = np.asarray(x_clusters)

    return x_clusters, query_clusters
