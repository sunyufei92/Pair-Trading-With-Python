import numpy as np
import scipy.stats as ss
from scipy.integrate import dblquad
from scipy.optimize import minimize
from scipy.special import gamma as gm
from sklearn.covariance import EmpiricalCovariance

from base import Copula


class StudentCopula(Copula):
    def __init__(self, nu: float = None, cov: np.array = None):
        r"""
        Initiate a Student copula object.

        :param nu: (float) Degrees of freedom.
        :param cov: (np.array) Covariance matrix (NOT correlation matrix), measurement of covariance. The class will
            calculate correlation internally once the covariance matrix is given.
        """

        super().__init__('Student')
        self.nu = nu  # Degree of freedom
        self.theta = None
        self.cov = None
        self.rho = None

        if cov is not None:
            self.cov = cov  # Covariance matrix
            # Correlation from covariance matrix
            self.rho = cov[0][1] / (np.sqrt(cov[0][0]) * np.sqrt(cov[1][1]))

    def sample(self, num: int = None) -> np.array:
        cov = self.cov
        nu = self.nu

        student_pairs = self._generate_corr_student(num, cov, nu)
        t_dist = ss.t(df=nu)
        copula_pairs = t_dist.cdf(student_pairs)

        return copula_pairs

    @staticmethod
    def _generate_corr_student(num: int, cov: np.ndarray, nu: float) -> tuple:
        # Sample from bivariate Normal with cov=cov
        rand_generator = np.random.default_rng()
        normal = rand_generator.multivariate_normal(mean=[0, 0], cov=cov, size=num)

        # Sample from Chi-square with df=nu
        chisq = rand_generator.chisquare(df=nu, size=num)
        result = np.zeros((num, 2))
        for row_idx, row in enumerate(result):
            row[0] = normal[row_idx][0] / np.sqrt(chisq[row_idx] / nu)
            row[1] = normal[row_idx][1] / np.sqrt(chisq[row_idx] / nu)

        return result

    def _get_param(self) -> dict:
        descriptive_name = r'Bivariate Student-t Copula'
        class_name = r'Student'
        cov = self.cov
        rho = self.rho
        nu = self.nu
        info_dict = {'Descriptive Name': descriptive_name,
                     'Class Name': class_name,
                     'cov': cov,
                     'rho': rho,
                     'nu (degrees of freedom)': nu}

        return info_dict

    def c(self, u: float, v: float) -> float:
        nu = self.nu
        rho = self.rho
        corr = [[1, rho],
                [rho, 1]]
        t_dist = ss.t(df=nu)
        y1 = t_dist.ppf(u)
        y2 = t_dist.ppf(v)

        numerator = self._bv_t_dist(x=(y1, y2),
                                    mu=(0, 0),
                                    cov=corr,
                                    df=nu)
        denominator = t_dist.pdf(y1) * t_dist.pdf(y2)

        pdf = numerator / denominator

        return pdf

    def C(self, u: float, v: float) -> float:
        # Avoid errors when u, v too close to 0
        u = max(u, 1e-6)
        v = max(v, 1e-6)
        corr = [[1, self.rho], [self.rho, 1]]  # Correlation matrix

        # Get raw result from integration on pdf
        def t_pdf_local(x1, x2):
            return self._bv_t_dist(x=[x1, x2], mu=[0, 0], cov=corr, df=self.nu)

        inv_t = (ss.t(df=self.nu)).ppf
        raw_result = dblquad(t_pdf_local, -np.inf, inv_t(u), -np.inf, inv_t(v), epsabs=1e-4, epsrel=1e-4)[0]

        # Map result back to [0, 1]
        cdf = max(min(raw_result, 1), 0)

        return cdf

    def condi_cdf(self, u: float, v: float) -> float:
        rho = self.rho
        nu = self.nu
        t_dist = ss.t(nu)
        t_dist_nup1 = ss.t(nu + 1)
        inv_cdf_u = t_dist.ppf(u)
        inv_cdf_v = t_dist.ppf(v)
        numerator = (inv_cdf_u - rho * inv_cdf_v) * np.sqrt(nu + 1)
        denominator = np.sqrt((1 - rho ** 2) * (inv_cdf_v ** 2 + nu))

        result = t_dist_nup1.cdf(numerator / denominator)

        return result

    @staticmethod
    def theta_hat(tau: float) -> float:
        r"""
        Calculate theta hat from Kendall's tau from sample data.

        :param tau: (float) Kendall's tau from sample data.
        :return: (float) The associated theta hat for this very copula.
        """

        return np.sin(tau * np.pi / 2)

    def fit(self, u: np.array, v: np.array) -> float:
        super().fit(u, v)
        # 1. Calculate covariance matrix using sklearn
        # Correct matrix dimension for fitting in sklearn
        unif_data = np.array([u, v]).reshape(2, -1).T
        t_dist = ss.t(df=self.nu)
        value_data = t_dist.ppf(unif_data)  # Change from quantile to value

        # Getting empirical covariance matrix
        cov_hat = EmpiricalCovariance().fit(value_data).covariance_
        self.cov = cov_hat
        self.rho = cov_hat[0][1] / (np.sqrt(cov_hat[0][0]) * np.sqrt(cov_hat[1][1]))

        return self.rho

    @staticmethod
    def _bv_t_dist(x: np.array, mu: np.array, cov: np.array, df: float) -> float:
        x1 = x[0] - mu[0]
        x2 = x[1] - mu[1]
        c11 = cov[0][0]
        c12 = cov[0][1]
        c21 = cov[1][0]
        c22 = cov[1][1]
        det_cov = c11 * c22 - c12 * c21
        # Pseudo code: (x.transpose)(cov.inverse)(x)/ Det(cov)
        xT_covinv_x = (-2 * c12 * x1 * x2 + c11 * (x1 ** 2 + x2 ** 2)) / det_cov

        numerator = gm((2 + df) / 2)
        denominator = (gm(df / 2) * df * np.pi * np.sqrt(det_cov)
                       * np.power(1 + xT_covinv_x / df, (2 + df) / 2))

        result = numerator / denominator

        return result


def fit_nu_for_t_copula(u: np.array, v: np.array, nu_tol: float = None) -> float:
    # Define the objective function
    def neg_log_likelihood_for_t_copula(nu):
        # 1. Calculate covariance matrix using sklearn
        # Correct matrix dimension for fitting in sklearn
        unif_data = np.array([u, v]).reshape(2, -1).T
        t_dist = ss.t(df=nu)
        value_data = t_dist.ppf(unif_data)  # Change from quantile to value
        # Getting empirical covariance matrix
        cov_hat = EmpiricalCovariance().fit(value_data).covariance_
        cop = StudentCopula(nu=nu, cov=cov_hat)
        log_likelihood = cop.get_log_likelihood_sum(u, v)

        return -log_likelihood  # Minimizing the negative of likelihood

    # Optimizing to find best nu
    nu0 = np.array([3])
    # Constraint: nu between [1, 15]. Too large nu value will lead to calculation issues for gamma function
    cons = ({'type': 'ineq', 'fun': lambda nu: nu + 1},  # x - 1 > 0
            {'type': 'ineq', 'fun': lambda nu: -nu + 15})  # -x + 15 > 0 (i.e., x - 10 < 0)

    res = minimize(neg_log_likelihood_for_t_copula, nu0, method='COBYLA', constraints=cons,
                   options={'disp': False}, tol=nu_tol)

    return res['x'][0]
