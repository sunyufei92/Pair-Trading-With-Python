import numpy as np
import scipy.stats as ss
from sklearn.covariance import EmpiricalCovariance

from base import Copula


class GaussianCopula(Copula):
    def __init__(self, cov: np.array = None):
        r"""
        Initiate a Gaussian copula object.

        :param cov: (np.array) Covariance matrix (NOT correlation matrix), measurement of covariance. The class will
            calculate correlation internally once the covariance matrix is given.
        """

        super().__init__('Gaussian')

        self.cov = None
        self.rho = None

        if cov is not None:
            self.cov = cov  # Covariance matrix
            # Correlation
            self.rho = cov[0][1] / (np.sqrt(cov[0][0]) * np.sqrt(cov[1][1]))

    def sample(self, num: int = None) -> np.array:
        cov = self.cov

        gaussian_pairs = self._generate_corr_gaussian(num, cov)
        sample_pairs = ss.norm.cdf(gaussian_pairs)

        return sample_pairs

    @staticmethod
    def _generate_corr_gaussian(num: int, cov: np.array) -> np.array:
        # Generate bivariate normal with mean 0 and intended covariance
        rand_generator = np.random.default_rng()
        result = rand_generator.multivariate_normal(mean=[0, 0], cov=cov, size=num)

        return result

    def _get_param(self) -> dict:
        descriptive_name = 'Bivariate Gaussian Copula'
        class_name = 'Gaussian'
        cov = self.cov
        rho = self.rho
        info_dict = {'Descriptive Name': descriptive_name,
                     'Class Name': class_name,
                     'cov': cov,
                     'rho': rho}

        return info_dict

    def fit(self, u: np.array, v: np.array) -> float:
        super().fit(u, v)
        # 1. Calculate covariance matrix using sklearn
        # Correct matrix dimension for fitting in sklearn
        unif_data = np.array([u, v]).reshape(2, -1).T
        value_data = ss.norm.ppf(unif_data)  # Change from quantile to value

        # Getting empirical covariance matrix
        cov_hat = EmpiricalCovariance().fit(value_data).covariance_
        self.cov = cov_hat
        self.rho = cov_hat[0][1] / (np.sqrt(cov_hat[0][0]) * np.sqrt(cov_hat[1][1]))

        return self.rho

    def c(self, u: float, v: float) -> float:
        rho = self.rho
        inv_u = ss.norm.ppf(u)
        inv_v = ss.norm.ppf(v)

        exp_ker = (rho * (-2 * inv_u * inv_v + inv_u ** 2 * rho + inv_v ** 2 * rho)
                   / (2 * (rho ** 2 - 1)))

        pdf = np.exp(exp_ker) / np.sqrt(1 - rho ** 2)

        return pdf

    def C(self, u: float, v: float) -> float:
        corr = [[1, self.rho], [self.rho, 1]]  # Correlation matrix
        inv_cdf_u = ss.norm.ppf(u)  # Inverse cdf of standard normal
        inv_cdf_v = ss.norm.ppf(v)
        mvn_dist = ss.multivariate_normal(mean=[0, 0], cov=corr)  # Joint cdf of multivariate normal
        cdf = mvn_dist.cdf((inv_cdf_u, inv_cdf_v))

        return cdf

    def condi_cdf(self, u, v) -> float:
        rho = self.rho
        inv_cdf_u = ss.norm.ppf(u)
        inv_cdf_v = ss.norm.ppf(v)
        sqrt_det_corr = np.sqrt(1 - rho * rho)
        result = ss.norm.cdf((inv_cdf_u - rho * inv_cdf_v)
                             / sqrt_det_corr)

        return result

    @staticmethod
    def theta_hat(tau: float) -> float:
        r"""
        Calculate theta hat from Kendall's tau from sample data.

        :param tau: (float) Kendall's tau from sample data.
        :return: (float) The associated theta hat for this very copula.
        """

        return np.sin(tau * np.pi / 2)
