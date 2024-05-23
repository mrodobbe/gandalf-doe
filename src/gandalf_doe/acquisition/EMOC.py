import numpy as np
from GPyOpt.acquisitions.base import AcquisitionBase
from GPy.models import GPRegression
from scipy.special import hyp1f1
import scipy
import math


class EMOC(AcquisitionBase):
    """
        General template to create a new GPyOPt acquisition function

        :param model: GPyOpt class of model
        :param space: GPyOpt class of domain
        :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
        :param cost_with_gradients: function that provides the evaluation cost and its gradients

        """

    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    def __init__(self, model, space, optimizer, X, pseudo_kernel, cost_with_gradients=None, **kwargs):
        super(EMOC, self).__init__(model, space, optimizer)
        # self.optimizer = optimizer
        # self.model = model
        self.kernel = pseudo_kernel
        if model.noise_var is None:
            self.sigmaN = 0
        else:
            self.sigmaN = model.noise_var
        self.norm = 1
        # self.X = X
        self.X = np.copy(X)
        # print('update', self.X[-1])

    def gaussian_absolute_moment(self, mu_tilde, pred_var):
        f11 = hyp1f1(-0.5 * self.norm, 0.5, -0.5 * np.divide(mu_tilde ** 2, pred_var))
        prefactors = ((2 * pred_var ** 2) ** (self.norm / 2.0) * math.gamma((1 + self.norm) / 2.0)) / np.sqrt(
            np.pi)

        return np.multiply(prefactors, f11)

    def calc_emoc(self, x):

        emoc_scores = np.asmatrix(np.empty([x.shape[0], 1], dtype=np.float64))
        mu_tilde = np.asmatrix(np.zeros([x.shape[0], 1], dtype=np.float64))

        k_all = self.kernel.K(np.vstack([self.X, x]))
        k = k_all[0:self.X.shape[0], self.X.shape[0]:]
        self_kdiag = np.asmatrix(np.diag(k_all[self.X.shape[0]:, self.X.shape[0]:])).T
        sigma_f = self.model.predict(x)[1]
        moments = np.asmatrix(self.gaussian_absolute_moment(np.asarray(mu_tilde), np.asarray(sigma_f)))

        term1 = 1.0 / (sigma_f + self.sigmaN)

        term2 = np.asmatrix(np.ones((self.X.shape[0] + 1, x.shape[0])), dtype=np.float64) * (-1.0)
        term2[0:self.X.shape[0], :] = np.linalg.solve(
            self.kernel.K(self.X) + np.identity(self.X.shape[0], dtype=np.float64) * self.sigmaN, k)

        pre_calc_mult = np.dot(term2[:-1, :].T, k_all[0:self.X.shape[0], :])
        for idx in range(x.shape[0]):
            v_all = term1[idx, :] * (pre_calc_mult[idx, :] + np.dot(term2[-1, idx].T, k_all[self.X.shape[0] + idx, :]))
            emoc_scores[idx, :] = np.mean(np.power(np.abs(v_all), self.norm))
        sol = np.multiply(emoc_scores, moments)
        sol = np.asarray(sol).reshape(-1)
        sol = np.array([sol]).reshape(-1, 1)
        return sol

    def _compute_acq(self, x):
        # --- DEFINE YOUR ACQUISITION HERE (TO BE MAXIMIZED)
        #
        # Compute here the value of the new acquisition function. Remember that x is a 2D  numpy array
        # with a point in the domain in each row. f_acqu_x should be a column vector containing the
        # values of the acquisition at x.
        #
        # print(self.calcEMOC(x).shape)
        return self.calc_emoc(x)


def gaussian_absolute_moment(mu_tilde, pred_var, norm=1):
    f11 = hyp1f1(-0.5 * norm, 0.5, -0.5 * np.divide(mu_tilde ** 2, pred_var))
    prefactors = ((2 * pred_var ** 2) ** (norm / 2.0) * math.gamma((1 + norm) / 2.0)) / np.sqrt(np.pi)

    return np.multiply(prefactors, f11)


def calc_emoc(x, X, kernel, model: GPRegression, sigma_n, norm=1):

    emoc_scores = np.asmatrix(np.empty([x.shape[0], 1], dtype=np.float64))
    mu_tilde = np.asmatrix(np.zeros([x.shape[0], 1], dtype=np.float64))

    k_all = kernel.K(np.vstack([X, x]))
    k = k_all[0:X.shape[0], X.shape[0]:]
    self_kdiag = np.asmatrix(np.diag(k_all[X.shape[0]:, X.shape[0]:])).T
    sigma_f = model.predict(x)[1]
    moments = np.asmatrix(gaussian_absolute_moment(np.asarray(mu_tilde), np.asarray(sigma_f)))

    term1 = 1.0 / (sigma_f + sigma_n)

    term2 = np.asmatrix(np.ones((X.shape[0] + 1, x.shape[0])), dtype=np.float64) * (-1.0)
    term2[0:X.shape[0], :] = np.linalg.solve(
        kernel.K(X) + np.identity(X.shape[0], dtype=np.float64) * sigma_n, k)

    pre_calc_mult = np.dot(term2[:-1, :].T, k_all[0:X.shape[0], :])
    for idx in range(x.shape[0]):
        v_all = term1[idx, :] * (pre_calc_mult[idx, :] + np.dot(term2[-1, idx].T, k_all[X.shape[0] + idx, :]))
        emoc_scores[idx, :] = np.mean(np.power(np.abs(v_all), norm))
    sol = np.multiply(emoc_scores, moments)
    sol = np.asarray(sol).reshape(-1)
    sol = np.array([sol]).reshape(-1, 1)
    return sol
