from sklearn.cluster import KMeans
from GPyOpt.experiment_design import initial_design
from gandalf_doe.clustering.utils import normalizer, denormalizer, rescale, descale
import numpy as np


def initialization_cluster(space, N, n_init, lower_bound, upper_bound, objective=None, normalize=True, scale=False):
    pool = initial_design('random', space, N)
    print(f"Initial shape: {pool.shape}")
    if normalize:
        pool = normalizer(pool, lower_bound, upper_bound)
    elif scale:
        pool = rescale(pool, None)
    cluster_init = KMeans(n_clusters=n_init).fit(pool)

    """
    x_init_ctu = cluster_init.cluster_centers_
    x_init = discrete_constraints_8d(x_init_ctu)
    """
    x_init = np.empty((n_init, pool.shape[1]))
    for k in range(n_init):
        var = cluster_init.transform(pool)[:, k]
        x_init[k] = pool[np.argmin(var)]
    if normalize:
        x_init = denormalizer(x_init, lower_bound, upper_bound)
    elif scale:
        x_init = descale(x_init, None)
    if objective is not None:
        y_init = np.empty((x_init.shape[0], 1))
        for j in range(x_init.shape[0]):
            y_init[j] = objective(np.array([x_init[j]]))
        return x_init, y_init
    else:
        return x_init, None


def select_clusters(cluster, m):
    old_clusters = cluster.labels_[-m:]
    new_clusters = []

    for j in range(m+1):
        if j+1 not in old_clusters:
            new_clusters.append((j+1, np.count_nonzero(cluster.labels_ == j+1)))

    return new_clusters


def select_clusters_pool(cluster, train_data):
    old_clusters = cluster.labels_
    new_clusters = []
    remove_clusters = []

    for j in range(len(old_clusters)):
        if j in train_data:
            remove_clusters.append(old_clusters[j])

    for i in range(len(train_data)+1):
        if i not in remove_clusters:
            new_clusters.append((i, np.count_nonzero(cluster.labels_ == i)))
    return new_clusters


def get_multiple_clusters(new_clusters, n: int):
    counts = []
    query_clusters = []
    for j in new_clusters:
        query_clusters.append(j[0])
        counts.append(j[1])

    counts = np.asarray(counts)
    query_clusters = np.asarray(query_clusters)
    top_clusters = np.argsort(counts)[-n:]
    return query_clusters[top_clusters]


def largest_empty_cluster(new_clusters):
    count = 0
    query_cluster = 0
    for j in new_clusters:
        if j[1] > count:
            query_cluster = j[0]
            count = j[1]
    return query_cluster


def elements_cluster(cluster, label, pool):
    X_cluster = []
    for j in range(cluster.labels_.shape[0]):
        if cluster.labels_[j] == label:
            X_cluster.append(pool[j])
    return np.array(X_cluster)


def initialization_cluster_pool(pool, n_init, objective, normalize=False, scaler=None):
    if normalize:
        pool = scaler.transform(pool)
    cluster_init = KMeans(n_clusters=n_init).fit(pool)
    """
    Xinit_ctu = cluster_init.cluster_centers_
    Xinit = discrete_constraints_8d(Xinit_ctu)
    """
    Xinit = np.empty((n_init, pool.shape[1]))
    Yinit = np.empty((n_init, objective.shape[1]))
    L = []
    for k in range(n_init):
        var = cluster_init.transform(pool)[:, k]
        Xinit[k] = pool[np.argmin(var)]
        Yinit[k] = objective[np.argmin(var)]
        L.append(np.argmin(var))
    if normalize:
        Xinit = scaler.inverse_transform(Xinit)
        pool = scaler.inverse_transform(pool)
    return Xinit, Yinit, L
