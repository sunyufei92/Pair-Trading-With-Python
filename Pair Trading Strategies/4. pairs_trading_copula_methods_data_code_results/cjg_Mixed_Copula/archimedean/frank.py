import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from base import Copula


class Frank(Copula):

    def __init__(self, theta: float = None, threshold: float = 1e-10):
        super().__init__('Frank')
        # Lower than this amount will be rounded to threshold
        self.threshold = threshold
        self.theta = theta  # Default input

    def sample(self, num: int = None, unif_vec: np.array = None) -> np.array:
        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec.")

        theta = self.theta  # Use the default input

        # Generate pairs of indep uniform dist vectors. Use numpy to generate
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))

        # Compute Frank copulas from the unif pairs
        sample_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            sample_pairs[row] = self._generate_one_pair(pair[0],
                                                        pair[1],
                                                        theta=theta)

        return sample_pairs

    @staticmethod
    def _generate_one_pair(u1: float, v2: float, theta: float) -> tuple:
        u2 = -1 / theta * np.log(1 + (v2 * (1 - np.exp(-theta))) /
                                 (v2 * (np.exp(-theta * u1) - 1)
                                  - np.exp(-theta * u1)))

        return u1, u2

    def _get_param(self) -> dict:
        descriptive_name = 'Bivariate Frank Copula'
        class_name = 'Frank'
        theta = self.theta
        info_dict = {'Descriptive Name': descriptive_name,
                     'Class Name': class_name,
                     'theta': theta}

        return info_dict

    def c(self, u: float, v: float) -> float:
        theta = self.theta
        et = np.exp(theta)
        eut = np.exp(u * theta)
        evt = np.exp(v * theta)
        pdf = (et * eut * evt * (et - 1) * theta /
               (et + eut * evt - et * eut - et * evt) ** 2)

        return pdf

    def C(self, u: float, v: float) -> float:
        theta = self.theta
        cdf = -1 / theta * np.log(
            1 + (np.exp(-1 * theta * u) - 1) * (np.exp(-1 * theta * v) - 1)
            / (np.exp(-1 * theta) - 1))

        return cdf

    def condi_cdf(self, u: float, v: float) -> float:
        theta = self.theta
        enut = np.exp(-u * theta)
        envt = np.exp(-v * theta)
        ent = np.exp(-1 * theta)
        result = (envt * (enut - 1)
                  / ((ent - 1) + (enut - 1) * (envt - 1)))

        return result

    @staticmethod
    def theta_hat(tau: float) -> float:
        r"""
        Calculate theta hat from Kendall's tau from sample data.

        :param tau: (float) Kendall's tau from sample data.
        :return: (float) The associated theta hat for this very copula.
        """

        def debye1(theta: float) -> float:
            result = quad(lambda x: x / theta / (np.exp(x) - 1), 0, theta)

            return result[0]

        def kendall_tau(theta: float) -> float:
            return 1 - 4 / theta + 4 * debye1(theta) / theta

        # Numerically find the root
        result = brentq(lambda theta: kendall_tau(theta) - tau, -100, 100)

        return result
