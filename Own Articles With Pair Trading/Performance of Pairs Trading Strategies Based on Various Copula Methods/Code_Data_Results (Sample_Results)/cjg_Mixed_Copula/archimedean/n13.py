from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from base import Copula


class N13(Copula):
    def __init__(self, theta: float = None, threshold: float = 1e-10):
        super().__init__('N13')
        # Lower than this amount will be rounded to threshold
        self.threshold = threshold
        self.theta = theta  # Default input

    def sample(self, num: int = None, unif_vec: np.array = None) -> np.array:
        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec.")

        theta = self.theta  # Use the default input

        def _Kc(w: float, theta: float):
            return w + 1 / theta * (
                    w - w * np.power((1 - np.log(w)), 1 - theta) - w * np.log(w))

        # Generate pairs of indep uniform dist vectors. Use numpy to generate
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))

        # Compute N13 copulas from the i.i.d. unif pairs
        sample_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            sample_pairs[row] = self._generate_one_pair(pair[0],
                                                        pair[1],
                                                        theta=theta,
                                                        Kc=_Kc)

        return sample_pairs

    def _generate_one_pair(self, v1: float, v2: float, theta: float, Kc: Callable[[float, float], float]) -> tuple:
        if v2 > self.threshold:
            w = brentq(lambda w1: Kc(w1, theta) - v2,
                       self.threshold, 1 - self.threshold)
        else:
            w = self.threshold  # Below the threshold, gives threshold as the root
        u1 = np.exp(
            1 - (v1 * ((1 - np.log(w)) ** theta - 1) + 1) ** (1 / theta))

        u2 = np.exp(
            1 - ((1 - v1) * ((1 - np.log(w)) ** theta - 1) + 1) ** (1 / theta))

        return u1, u2

    def _get_param(self) -> dict:
        descriptive_name = 'Bivariate Nelsen 13 Copula'
        class_name = 'N13'
        theta = self.theta
        info_dict = {'Descriptive Name': descriptive_name,
                     'Class Name': class_name,
                     'theta': theta}

        return info_dict

    def c(self, u: float, v: float) -> float:
        theta = self.theta
        u_part = (1 - np.log(u)) ** theta
        v_part = (1 - np.log(v)) ** theta
        Cuv = self.C(u, v)

        numerator = (Cuv * u_part * v_part
                     * (-1 + theta + (-1 + u_part + v_part) ** (1 / theta))
                     * (-1 + u_part + v_part) ** (1 / theta))

        denominator = u * v * (1 - np.log(u)) * (1 - np.log(v)) * (-1 + u_part + v_part) ** 2

        pdf = numerator / denominator

        return pdf

    def C(self, u: float, v: float) -> float:
        theta = self.theta
        u_part = (1 - np.log(u)) ** theta
        v_part = (1 - np.log(v)) ** theta
        cdf = np.exp(
            1 - (-1 + u_part + v_part) ** (1 / theta))

        return cdf

    def condi_cdf(self, u, v) -> float:
        theta = self.theta
        u_part = (1 - np.log(u)) ** theta
        v_part = (1 - np.log(v)) ** theta
        Cuv = self.C(u, v)

        numerator = Cuv * (-1 + u_part + v_part) ** (1 / theta) * v_part
        denominator = v * (-1 + u_part + v_part) * (1 - np.log(v))

        result = numerator / denominator

        return result

    @staticmethod
    def theta_hat(tau: float) -> float:
        # Calculate tau(theta) = 1 + 4*intg_0^1[phi(t)/d(phi(t)) dt]
        def kendall_tau(theta):
            # phi(t)/d(phi(t)), phi is the generator function for this copula
            pddp = lambda x: -((x - x * (1 - np.log(x)) ** (1 - theta) - x * np.log(x)) / theta)
            result = quad(pddp, 0, 1, full_output=1)[0]
            return 1 + 4 * result

        # Numerically find the root.
        result = brentq(lambda theta: kendall_tau(theta) - tau, 1e-7, 100)

        return result
