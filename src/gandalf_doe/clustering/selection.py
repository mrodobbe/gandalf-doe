import GPy
from gandalf_doe.optimizer.rprop_gpy import RProp
import numpy as np
from gandalf_doe.acquisition.EMOC import calc_emoc
from gandalf_doe.domain import Domain
from sklearn.cluster import KMeans
import random


def normalize(values, lower_bound, upper_bound):
    new_values = (values - lower_bound) / (upper_bound - lower_bound)
    return new_values


def denormalize(values, lower_bound, upper_bound):
    new_values = values * (upper_bound - lower_bound) + lower_bound
    return new_values


class GP:

    def __init__(self, x: np.ndarray, y: np.ndarray, kernel_type: str = "RBF", ard: bool = True, 
                 rprop_iters: int = 250, model_iters: int = 5000, noise_var: float = 1.0, variance: float = 1.0, lengthscale: bool = True, 
                 power: float = 2.0, c: float = 1.0, normalize: bool = True):
        self.x = x
        self.y = y
        self.ard = ard
        self.rprop_iters = rprop_iters
        self.model_iters = model_iters
        self.noise_var = noise_var
        self.kernel_type = kernel_type
        self.variance = variance
        self.lengthscale = lengthscale
        self.power = power
        self.c = c
        self.normalize = normalize
        if self.kernel_type == "RBF":
            self.kernel = GPy.kern.RBF(input_dim=self.x.shape[1], ARD=self.ard, variance=self.variance)
        elif self.kernel_type == "linear":
            self.kernel = GPy.kern.Linear(input_dim=self.x.shape[1], ARD=self.ard)
        elif self.kernel_type == "matern32":
            self.kernel = GPy.kern.Matern32(input_dim=self.x.shape[1], ARD=self.ard, variance=self.variance)
        elif self.kernel_type == "matern52":
            self.kernel = GPy.kern.Matern52(input_dim=self.x.shape[1], ARD=self.ard, variance=self.variance)
        elif self.kernel_type == "exponential":
            self.kernel = GPy.kern.Exponential(input_dim=self.x.shape[1], ARD=self.ard, variance=self.variance)
        elif self.kernel_type == "rq":
            self.kernel = GPy.kern.RatQuad(input_dim=self.x.shape[1], ARD=self.ard, variance=self.variance, power=self.power)
        elif self.kernel_type == "spline":
            self.kernel = GPy.kern.Spline(input_dim=self.x.shape[1], variance=self.variance, c=self.c)
        else:
            raise ValueError(f"Kernel {self.kernel_type} is not supported. Choose from RBF, linear, matern32, matern52.") 
            
        self.model = GPy.models.GPRegression(self.x, self.y, self.kernel, normalizer=True, noise_var=self.noise_var)

    def set_length_scale(self, domain: Domain):
        if self.ard and self.lengthscale:
            if not self.normalize:
                self.kernel.lengthscale = list(0.5 * (domain.min_values + domain.max_values))
            else:
                self.kernel.lengthscale = list(0.5 * np.ones(len(domain.min_values)))
        else:
            self.kernel.lengthscale = None

    def optimize_model(self):
        self.model.optimize(RProp(max_iters=self.rprop_iters), messages=False)
        self.model.optimize(messages=False, max_iters=self.model_iters)

    def predict(self, x_new):
        mean, variance = self.model.predict(x_new)
        return mean, variance


class Selection:

    def __init__(self, pool: np.ndarray, domain: Domain):
        self.pool = pool
        self.domain = domain

    def select_random(self):
        return random.choice(self.pool)

    def select_center(self, cluster, new_cluster):
        var = cluster.transform(self.pool)[:, new_cluster]
        return self.pool[np.argmin(var)]

    def select_uncertainty(self, x: np.ndarray, y: np.ndarray):
        gp = GP(x, y)
        gp.set_length_scale(self.domain)
        gp.optimize_model()
        var = gp.model.predict(self.pool)[1]

        return self.pool[np.argmax(var)]

    def select_emoc(self, x: np.ndarray, y: np.ndarray):
        n_x = normalize(x, self.domain.min_values, self.domain.max_values)
        gp = GP(n_x, y)
        gp.set_length_scale(self.domain)
        gp.model.randomize()
        gp.optimize_model()
        sigma_n = gp.model.Gaussian_noise[0]
        n_pool = normalize(self.pool, self.domain.min_values, self.domain.max_values)
        var = calc_emoc(n_pool, x, gp.kernel, gp.model, sigma_n)

        return self.pool[np.argmax(var)], gp.kernel.lengthscale

    def select_improve(self, x: np.ndarray, y: np.ndarray):
        gp = GP(x, y)
        gp.set_length_scale(self.domain)
        gp.optimize_model()
        mean = gp.model.predict(self.pool)[0]
        expected_improve = max(mean)[0] - max(y)[0]
        print(f"Predicted maximum = {max(mean)[0]:.2f}")

        return self.pool[np.argmax(mean)], expected_improve


def select_uncertainty(pool, X, Y, lower_bound, upper_bound):
    kernel = GPy.kern.RBF(input_dim=X.shape[1], ARD=True)
    model = GPy.models.GPRegression(X, Y, kernel, normalizer=True)
    kernel.lengthscale = list((lower_bound + upper_bound)/2)
    model.optimize(RProp(), messages=False)
    model.optimize(messages=False, max_iters=5000)
    var = model.predict(pool)[1]
    return pool[np.argmax(var)]


def select_emoc(pool, X, Y, lower_bound, upper_bound):
    kernel = GPy.kern.RBF(input_dim=X.shape[1], ARD=True)
    model = GPy.models.GPRegression(X, Y, kernel, normalizer=True)
    kernel.lengthscale = list((lower_bound + upper_bound)/2)
    model.optimize(RProp(max_iters=250), messages=False)
    model.optimize(messages=False, max_iters=5000)
    sigma_n = model.Gaussian_noise[0]
    var = calc_emoc(pool, X, kernel, model, sigma_n)
    return pool[np.argmax(var)], kernel.lengthscale


def select_center(pool, cluster, new_cluster):
    var = cluster.transform(pool)[:, new_cluster]
    return pool[np.argmin(var)]


def select_random(pool):
    return random.choice(pool)

