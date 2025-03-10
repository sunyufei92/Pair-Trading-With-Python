from typing import Callable

import numpy as np
from scipy.optimize import brentq

from base import Copula


class Gumbel(Copula):
    def __init__(self, theta: float = None, threshold: float = 1e-10):
        super().__init__('Gumbel')
        # Lower than this amount will be rounded to threshold
        self.threshold = threshold
        self.theta = theta  # Gumbel copula parameter

    def sample(self, num: int = None, unif_vec: np.array = None) -> np.array:
        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec.")

        theta = self.theta  # Use the default input

        # Distribution of C(U1, U2). To be used for numerically solving the inverse
        def _Kc(w: float, theta: float):
            return w * (1 - np.log(w) / theta)

        # Generate pairs of indep uniform dist vectors
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))

        # Compute Gumbel copulas from the independent uniform pairs
        sample_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            sample_pairs[row] = self._generate_one_pair(pair[0],
                                                        pair[1],
                                                        theta=theta,
                                                        Kc=_Kc)

        return sample_pairs

    def _generate_one_pair(self, v1: float, v2: float, theta: float, Kc: Callable[[float, float], float]) -> tuple:
        # Numerically root finding for w1, where Kc(w1) = v2
        if v2 > self.threshold:
            w = brentq(lambda w1: Kc(w1, theta) - v2, self.threshold, 1)
        else:
            w = 1e10  # Below the threshold, gives a large number as root
        u1 = np.exp(v1 ** (1 / theta) * np.log(w))
        u2 = np.exp((1 - v1) ** (1 / theta) * np.log(w))

        return u1, u2

    def _get_param(self):
        descriptive_name = 'Bivariate Gumbel Copula'
        class_name = 'Gumbel'
        theta = self.theta
        info_dict = {'Descriptive Name': descriptive_name,
                     'Class Name': class_name,
                     'theta': theta}

        return info_dict

    def c(self, u: float, v: float) -> float:
        theta = self.theta
        # Prepare parameters
        u_part = (-np.log(u)) ** theta
        v_part = (-np.log(v)) ** theta
        expo = (u_part + v_part) ** (1 / theta)

        # Assembling for P.D.F.
        pdf = 1 / (u * v) \
              * (np.exp(-expo)
                 * u_part / (-np.log(u)) * v_part / (-np.log(v))
                 * (theta + expo - 1)
                 * (u_part + v_part) ** (1 / theta - 2))

        return pdf

    def C(self, u: float, v: float) -> float:
        theta = self.theta
        # Prepare parameters
        expo = ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta)

        # Assembling for P.D.F.
        cdf = np.exp(-expo)

        return cdf

    def condi_cdf(self, u: float, v: float) -> float:
        theta = self.theta
        expo = ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** ((1 - theta) / theta)
        result = self.C(u, v) * expo * (-np.log(v)) ** (theta - 1) / v

        return result

    @staticmethod
    def theta_hat(tau: float) -> float:
        return 1 / (1 - tau)
