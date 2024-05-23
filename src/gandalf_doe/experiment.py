from gandalf_doe.clustering.selection import *
from gandalf_doe.clustering.pool import *
from gandalf_doe.domain import Domain
from scipy.spatial.distance import euclidean
import pandas as pd
import numpy as np
from typing import Union, List
from numpy.typing import NDArray


class Experiment:

    def __init__(self, domain: Domain, pool_size: int = 100000, n_init: int = 10,
                 objective=None, normalize: bool = True, scale: bool = False,
                 length_scale: Union[None, NDArray] = None, mode: str = "EMOC",
                 pool: Union[None, NDArray[Any]] = None, clustering: bool = True,
                 al_steps: int = 1e9, exploration: float = 0.05, threshold: int = 30, epsilon: float = 1e-3,
                 scaling: int = 100):

        self.domain = domain
        self.pool_size = pool_size
        self.n_init = n_init
        self.objective = objective
        self.normalize = normalize
        self.variables = self.domain.variables
        self.names = [variable.name for variable in self.variables]
        self.scale = scale
        self.mode = mode

        if length_scale is None:
            self.length_scale = np.ones((1, len(self.domain.variables)))
        else:
            self.length_scale = length_scale

        if pool is None:
            self.pool: Pool = Pool(self.domain, pool_size=self.pool_size)
        else:
            self.pool = Pool(self.domain, pool=pool)

        self.run = 0
        self.clustering = clustering
        self.al_steps = al_steps
        self.exploration = exploration
        self.threshold = threshold
        self.epsilon = epsilon
        self.maximum_found = None
        self.f = scaling

    def initialize_experiments(self, fixed_pool: bool = False):
        if not fixed_pool:
            self.pool.initialize_pool()

        if self.normalize:
            self.pool.normalize()
        elif self.scale:
            self.pool.scale(self.length_scale)

        x, y = initialize_cluster(self.domain, self.pool, self.n_init,
                                  objective=self.objective, normalize=self.normalize, scale=self.scale)

        experiment_dict = dict()
        for i in range(len(self.names)):
            name = self.names[i]
            experiment_dict[name] = x[:, i]

        if self.objective is not None:
            experiment_dict["output"] = y
        else:
            experiment_dict["output"] = ["" for _ in range(self.n_init)]

        if self.normalize:
            self.pool.denormalize()
        elif self.scale:
            self.pool.descale(self.length_scale)

        df_exp = pd.DataFrame(experiment_dict)

        return df_exp

    def suggest_experiments(self, previous: pd.DataFrame = None, fixed_pool: bool = False, n_conditions: int = 0):
        if self.maximum_found is not None:
            print(f"Optimization finished. A maximum of {self.maximum_found: .2f} is found.")
            return previous

        df_x = previous[self.names]
        x = df_x.to_numpy(dtype=np.float64)
        y = previous["output"].to_numpy(dtype=np.float64).reshape(-1, 1)
        n = x.shape[0]
        new_experiments = n - self.run
        self.run = n

        if not fixed_pool:
            self.pool.initialize_pool(pool_size=self.f * n)

        if self.run < self.al_steps:
            self.pool.add_experiments(x[-new_experiments:])

        if self.normalize:
            self.pool.normalize()
        elif self.scale:
            self.pool.scale(self.length_scale)

        cluster, query_cluster = draw_cluster_from_pool(self.pool, n=x.shape[0])
        cluster_pool = Pool(self.domain, pool=cluster)

        if not self.clustering or self.run > self.al_steps:
            cluster_pool = Pool(self.domain, pool=self.pool.pool)

        if self.normalize:
            cluster_pool.denormalize()
        elif self.scale:
            cluster_pool.descale(self.length_scale)

        s = Selection(cluster_pool.pool, self.domain)
        draw = np.random.rand(1)
        al_selection = True

        if self.run > self.al_steps and draw > self.exploration:
            al_selection = False
            x_new, expected_improve = s.select_improve(x, y)
            if expected_improve < self.epsilon and self.run < self.threshold:
                print("No improve expected! Converting to active learning.")
                al_selection = True
                s = self.select_next(x)

            elif self.run >= self.threshold and expected_improve < self.epsilon:
                print(f"Optimization finished. A maximum of {max(y)[0]: .2f} is found.")
                self.maximum_found = max(y)[0]
            else:
                print(f"New maximum expected with improve of {expected_improve:.3f}!")
        elif draw <= self.exploration:
            s = self.select_next(x)

        if al_selection:
            if self.mode == 'uncertainty':
                x_new = s.select_uncertainty(x, y)
            elif self.mode == 'EMOC':
                try:
                    x_new, self.length_scale = s.select_emoc(x, y)
                except:
                    raise
            elif self.mode == 'center':
                x_new = s.select_center(cluster, query_cluster)
            elif self.mode == 'random':
                x_new = s.select_random()
            else:
                raise KeyError("Only the following modes are supported: uncertainty, EMOC, center, random.")

        if self.objective is not None:
            y_new = self.objective(np.array([x_new]))
        else:
            y_new = [""]

        if self.normalize:
            self.pool.denormalize()
        elif self.scale:
            self.pool.descale(self.length_scale)

        if fixed_pool:
            self.pool.pool = np.delete(self.pool.pool,
                                       np.where(np.all(self.pool.pool == x_new, axis=1))[0][0], 0)

        measured = previous.copy()

        new_index = previous.index[-1] + 1
        previous.loc[new_index] = list(x_new) + list(y_new)

        if n_conditions > 0:

            print(f"The selected cluster has {len(s.pool)} points")

            same_conditions = np.where((s.pool[:, 0] != x_new[0]) &
                                       (s.pool[:, 1] != x_new[1]) &
                                       (s.pool[:, 2] != x_new[2]) &
                                       (s.pool[:, 3] == x_new[3]) &
                                       (s.pool[:, 4] == x_new[4]) &
                                       (s.pool[:, 5] == x_new[5]) &
                                       (s.pool[:, 6] == x_new[6]))[0]

            if len(same_conditions) == 0:
                print("No similar conditions in the selected pool")
                pass

            elif len(same_conditions) < n_conditions + 1:
                print(f"There are {len(same_conditions)} similar conditions in the selected pool")
                for value in same_conditions:
                    cat = s.pool[value]
                    new_index = new_index + 1
                    previous.loc[new_index] = list(cat) + list(y_new)

            else:
                print(f"There are {len(same_conditions)} similar conditions in the selected pool")
                variances = []
                for value in same_conditions:
                    cat = s.pool[value]
                    variance = self.predict_outcome(cat, measured)["var"]
                    variances.append(variance)

                variances = np.array(variances).reshape(-1)
                top_3_ids = np.argsort(variances)[-n_conditions:]

                for value in top_3_ids:
                    cat = s.pool[value]
                    new_index = new_index + 1
                    previous.loc[new_index] = list(cat.reshape(-1)) + list(y_new)

        return previous

    def select_next(self, x):
        cluster, query_cluster = draw_cluster_from_pool(self.pool, n=x.shape[0])
        cluster_pool = Pool(self.domain, pool=cluster)
        if self.normalize:
            cluster_pool.denormalize()
        elif self.scale:
            cluster_pool.descale(self.length_scale)

        s = Selection(cluster_pool.pool, self.domain)
        return s

    def predict_outcome(self, x_new: np.ndarray, previous: pd.DataFrame = None, normalize: bool = True,
                        kernel_type: str = "RBF", ard: bool = True, rprop_iters: int = 250,
                        model_iters: int = 5000, noise_var: float = 1.0, variance: float = 1.0,
                        length_scale: bool = True,
                        power: float = 2.0, c: float = 1.0) -> pd.DataFrame:
        variables = self.domain.variables
        names = [variable.name for variable in variables]
        df_x = previous[names]
        x = df_x.to_numpy(dtype=np.float64)
        if normalize:
            lower_bound = self.domain.min_values
            upper_bound = self.domain.max_values
            x = (x - lower_bound) / (upper_bound - lower_bound)
            x_new = (x_new - lower_bound) / (upper_bound - lower_bound)

        y = previous["output"].to_numpy(dtype=np.float64).reshape(-1, 1)
        gp = GP(x, y, kernel_type, ard, rprop_iters,
                model_iters, noise_var, variance,
                length_scale, c=c, power=power, normalize=normalize)
        gp.set_length_scale(self.domain)
        gp.model.randomize()
        gp.optimize_model()

        if len(x_new.shape) == 1:
            x_new = np.array([x_new])

        mean, variance = gp.predict(x_new)
        if normalize:
            x_new = x_new * (upper_bound - lower_bound) + lower_bound

        predictions_dict = dict()
        for i in range(len(names)):
            name = names[i]
            predictions_dict[name] = x_new[:, i]
        predictions_dict["mean"] = mean.reshape(-1)
        predictions_dict["var"] = variance.reshape(-1)

        df_pred = pd.DataFrame(predictions_dict)

        return df_pred

    def suggest_batch_of_experiments(self, previous: pd.DataFrame = None,
                                     fixed_pool: bool = False,
                                     n_clusters: int = 1, n_conditions: int = 1,
                                     constants: Union[None, List[int]] = None) -> pd.DataFrame:

        if constants is None:
            constants = []

        df_x = previous[self.names]
        x = df_x.to_numpy(dtype=np.float64)
        y = previous["output"].to_numpy(dtype=np.float64).reshape(-1, 1)
        n = x.shape[0]
        new_experiments = n - self.run

        measured = previous.copy()
        self.run = n

        if not fixed_pool:
            self.pool.initialize_pool(pool_size=self.f * n)

        self.pool.add_experiments(x[-new_experiments:])

        if self.normalize:
            self.pool.normalize()
        elif self.scale:
            self.pool.scale(self.length_scale)

        clusters, query_clusters = draw_multiple_clusters_from_pool(self.pool, n_cluster=x.shape[0], n=n_clusters)
        print(f"There are {len(clusters)} clusters selected.")

        for i_cluster in range(len(clusters)):
            cluster = clusters[i_cluster]
            print(f"Cluster {i_cluster+1} with {len(cluster)} points.")
            query_cluster = query_clusters[i_cluster]
            cluster_pool = Pool(self.domain, pool=cluster)

            if self.normalize:
                cluster_pool.denormalize()
            elif self.scale:
                cluster_pool.descale(self.length_scale)

            s = Selection(cluster_pool.pool, self.domain)
            draw = np.random.rand(1)
            al_selection = True

            if self.run > self.al_steps and draw > self.exploration:
                al_selection = False
                x_new, expected_improve = s.select_improve(x, y)
                if expected_improve < self.epsilon and self.run < self.threshold:
                    print("No improve expected!")

                elif self.run >= self.threshold:
                    print(f"Optimization finished. A maximum of {max(y)[0]: .2f} is found.")
                    self.maximum_found = max(y)[0]
                else:
                    print(f"New maximum expected with improve of {expected_improve:.3f}!")
            elif draw <= self.exploration:
                s = self.select_next(x)

            if al_selection:
                if self.mode == 'uncertainty':
                    x_new = s.select_uncertainty(x, y)
                elif self.mode == 'EMOC':
                    try:
                        x_new, self.length_scale = s.select_emoc(x, y)
                    except:
                        raise
                elif self.mode == 'center':
                    x_new = s.select_center(cluster, query_cluster)
                elif self.mode == 'random':
                    x_new = s.select_random()
                else:
                    raise KeyError("Only the following modes are supported: uncertainty, EMOC, center, random.")

            if self.objective is not None:
                y_new = self.objective(np.array([x_new]))
            else:
                y_new = [""]

            new_index = previous.index[-1] + 1
            previous.loc[new_index] = list(x_new) + list(y_new)

            if n_conditions > 0:

                print(f"The selected cluster has {len(s.pool)} points")
                all_indices = set(range(len(self.variables)))
                not_equals = list(all_indices - set(constants))

                conditions = []

                # Add `!=` conditions
                for idx in not_equals:
                    conditions.append(s.pool[:, idx] != x_new[idx])

                # Add `==` conditions
                for idx in constants:
                    conditions.append(s.pool[:, idx] == x_new[idx])

                combined_condition = np.logical_and.reduce(conditions)
                same_cluster = np.where(combined_condition)[0]

                if len(same_cluster) == 0:
                    print("No similar conditions in the selected pool")
                    conditions = []

                    # Add `!=` conditions
                    for idx in not_equals:
                        conditions.append(self.pool.pool[:, idx] != x_new[idx])

                    # Add `==` conditions
                    for idx in constants:
                        conditions.append(self.pool.pool[:, idx] == x_new[idx])

                    combined_condition = np.logical_and.reduce(conditions)
                    same_cluster = np.where(combined_condition)[0]

                    print(f"There are {len(same_cluster)} similar conditions in the general pool")
                    if len(same_cluster) == 0:
                        print("No similar conditions in the general pool")
                    elif len(same_cluster) < n_conditions + 1:
                        print(f"There are {len(same_cluster)} similar conditions in the general pool")
                        for value in same_cluster:
                            cat = self.pool.pool[value]
                            new_index = new_index + 1
                            previous.loc[new_index] = list(cat) + list(y_new)
                    else:
                        print(f"There are {len(same_cluster)} similar conditions in the general pool")
                        cats = self.pool.pool[same_cluster]
                        for j in range(n_conditions):
                            new_selection = Selection(cats, self.domain)
                            x_new, self.length_scale = new_selection.select_emoc(x, y)
                            new_index = new_index + 1
                            previous.loc[new_index] = list(x_new) + list(y_new)
                            same_condition = np.where((cats[:, 0] == x_new[0]) &
                                                      (cats[:, 1] == x_new[1]) &
                                                      (cats[:, 2] == x_new[2]) &
                                                      (cats[:, 3] == x_new[3]) &
                                                      (cats[:, 4] == x_new[4]) &
                                                      (cats[:, 5] == x_new[5]) &
                                                      (cats[:, 6] == x_new[6]))[0]
                            cats = np.delete(cats, same_condition, axis=0)

                elif len(same_cluster) < n_conditions + 1:
                    print(f"There are {len(same_cluster)} similar conditions in the selected pool")
                    for value in same_cluster:
                        cat = s.pool[value]
                        new_index = new_index + 1
                        previous.loc[new_index] = list(cat) + list(y_new)

                else:
                    print(f"There are {len(same_cluster)} similar conditions in the selected pool")
                    variances = []
                    distances = []
                    weights = []
                    for value in same_cluster:
                        cat = s.pool[value]
                        d = euclidean(cat, x_new)
                        distances.append(d)
                        variance = self.predict_outcome(cat, measured)["var"]
                        variances.append(variance)
                        weight = np.sqrt(variance ** 2 + d ** 2)
                        weights.append(weight)

                    weights = np.array(weights).reshape(-1)
                    top_3_ids = np.argsort(weights)[-n_conditions:]

                    for value in top_3_ids:
                        cat = s.pool[same_cluster[value]]
                        new_index = new_index + 1
                        previous.loc[new_index] = list(cat.reshape(-1)) + list(y_new)

        if self.normalize:
            self.pool.denormalize()

        elif self.scale:
            self.pool.descale(self.length_scale)

        if fixed_pool:
            self.pool.pool = np.delete(self.pool.pool,
                                       np.where(np.all(self.pool.pool == x_new, axis=1))[0][0], 0)

        return previous
