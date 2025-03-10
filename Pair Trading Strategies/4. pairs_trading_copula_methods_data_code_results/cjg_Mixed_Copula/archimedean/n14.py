from typing import Callable

import numpy as np
from scipy.optimize import brentq

from base import Copula


class N14(Copula):
    def __init__(self, theta: float = None, threshold: float = 1e-10):
        super().__init__('N14')
        # Lower than this amount will be rounded to threshold
        self.threshold = threshold
        self.theta = theta  # Default input

    def sample(self, num: int = None, unif_vec: np.array = None) -> np.array:
        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec")

        theta = self.theta  # Use the default input

        def _Kc(w: float, theta: float):
            return -w * (-2 + w ** (1 / theta))

        # Generate pairs of indep uniform dist vectors. Use numpy to generate
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))

        # Compute Gumbel copulas from the unif pairs
        sample_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            sample_pairs[row] = self._generate_one_pair(pair[0], pair[1], theta=theta, Kc=_Kc)

        return sample_pairs

    def _generate_one_pair(self, v1: float, v2: float, theta: float, Kc: Callable[[float, float], float]) -> tuple:
        if v2 > self.threshold:
            w = brentq(lambda w1: Kc(w1, theta) - v2,
                       self.threshold, 1 - self.threshold)
        else:
            w = self.threshold  # Below the threshold, gives threshold as the root
        u1 = (1 + (v1 * (w ** (-1 / theta) - 1) ** theta) ** (1 / theta)) ** (-theta)
        u2 = (1 + ((1 - v1) * (w ** (-1 / theta) - 1) ** theta) ** (1 / theta)) ** (-theta)

        return u1, u2

    def _get_param(self) -> dict:
        descriptive_name = 'Bivariate Nelsen 14 Copula'
        class_name = 'N14'
        theta = self.theta
        info_dict = {'Descriptive Name': descriptive_name,
                     'Class Name': class_name,
                     'theta': theta}

        return info_dict

    def c(self, u: float, v: float) -> float:
        theta = self.theta
        u_ker = -1 + np.power(u, 1 / theta)
        v_ker = -1 + np.power(v, 1 / theta)
        u_part = (-1 + np.power(u, -1 / theta)) ** theta
        v_part = (-1 + np.power(v, -1 / theta)) ** theta
        cdf_ker = 1 + (u_part + v_part) ** (1 / theta)

        numerator = (u_part * v_part * (cdf_ker - 1)
                     * (-1 + theta + 2 * theta * (cdf_ker - 1)))

        denominator = ((u_part + v_part) ** 2 * cdf_ker ** (2 + theta)
                       * u * v * u_ker * v_ker * theta)

        pdf = numerator / denominator

        return pdf

    def C(self, u: float, v: float) -> float:
        theta = self.theta
        u_part = (-1 + np.power(u, -1 / theta)) ** theta
        v_part = (-1 + np.power(v, -1 / theta)) ** theta
        cdf = (1 + (u_part + v_part) ** (1 / theta)) ** (-1 * theta)

        return cdf

    def condi_cdf(self, u: float, v: float) -> float:
        theta = self.theta
        v_ker = -1 + np.power(v, -1 / theta)
        u_part = (-1 + np.power(u, -1 / theta)) ** theta
        v_part = (-1 + np.power(v, -1 / theta)) ** theta
        cdf_ker = 1 + (u_part + v_part) ** (1 / theta)

        numerator = v_part * (cdf_ker - 1)
        denominator = v ** (1 + 1 / theta) * v_ker * (u_part + v_part) * cdf_ker ** (1 + theta)

        result = numerator / denominator

        return result

    @staticmethod
    def theta_hat(tau: float) -> float:
        # N14 has a closed form solution
        result = (1 + tau) / (2 - 2 * tau)

        return result
