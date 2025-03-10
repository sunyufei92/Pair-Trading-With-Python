from abc import ABC, abstractmethod
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss


class Copula(ABC):
    def __init__(self, copula_name: str):
        # Name of each types of copula
        self.archimedean_names = ('Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14')
        self.elliptic_names = ('Gaussian', 'Student')
        self.theta = None
        self.rho = None
        self.nu = None
        self.copula_name = copula_name

    def describe(self) -> pd.Series:
        description = pd.Series(self._get_param())

        return description

    @staticmethod
    def theta_hat(tau: float) -> float:
        r"""
        Calculate theta hat from Kendall's tau from sample data.

        :param tau: (float) Kendall's tau from sample data.
        :return: (float) The associated theta hat for this very copula.
        """

    def get_cop_density(self, u: float, v: float, eps: float = 1e-5) -> float:
        # Mapping u, v back to the valid computational interval
        u = min(max(eps, u), 1 - eps)
        v = min(max(eps, v), 1 - eps)

        # Wrapper around individual copula's c method
        return self.c(u, v)

    def get_cop_eval(self, u: float, v: float, eps: float = 1e-5) -> float:
        # Mapping u, v back to the valid computational interval
        u = min(max(eps, u), 1 - eps)
        v = min(max(eps, v), 1 - eps)

        # Wrapper around individual copula's C method
        return self.C(u, v)

    def get_condi_prob(self, u: float, v: float, eps: float = 1e-5) -> float:
        # Mapping u, v back to the valid computational interval
        u = min(max(eps, u), 1 - eps)
        v = min(max(eps, v), 1 - eps)

        # Wrapper around individual copula's condi_cdf method
        return self.condi_cdf(u, v)

    def get_log_likelihood_sum(self, u: np.array, v: np.array) -> float:
        # Likelihood quantity for each pair of data, stored in a list
        likelihood_list = [self.c(xi, yi) for (xi, yi) in zip(u, v)]

        # Sum of logarithm of likelihood data
        log_likelihood_sum = np.sum(np.log(likelihood_list))

        return log_likelihood_sum

    def c(self, u: float, v: float) -> float:
        """
        Placeholder for calculating copula density.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        """

    def C(self, u: float, v: float) -> float:
        """
        Placeholder for calculating copula evaluation.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        """

    def condi_cdf(self, u: float, v: float) -> float:
        """
        Placeholder for calculating copula conditional probability.

        :param u: (float) A real number in [0, 1].
        :param v: (float) A real number in [0, 1].
        """

    @abstractmethod
    def sample(self, num: int = None, unif_vec: np.array = None) -> np.array:
        """
        Place holder for sampling from copula.

        :param num: (int) Number of points to generate.
        :param unif_vec: (np.array) Shape=(num, 2) array, two independent uniformly distributed sets of data.
        """

    def fit(self, u: np.array, v: np.array) -> float:
        # Calculate Kendall's tau from data
        tau = ss.kendalltau(u, v)[0]

        # Translate Kendall's tau into theta
        theta_hat = self.theta_hat(tau)
        self.theta = theta_hat

        return theta_hat

    @abstractmethod
    def _get_param(self):
        """
        Placeholder for getting the parameter(s) of the specific copula.
        """

    @staticmethod
    def _3d_surface_plot(x: np.array, y: np.array, z: np.array, bounds: list, title: str, **kwargs) -> Figure:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xticks(np.linspace(bounds[0], bounds[1], 6))
        ax.set_yticks(np.linspace(bounds[0], bounds[1], 6))
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.plot_surface(x, y, z, **kwargs)
        plt.title(title)

        return fig

    @staticmethod
    def _2d_contour_plot(x: np.array, y: np.array, z: np.array, bounds: float, title: str,
                         levels: list, **kwargs) -> Figure:
        fig = plt.figure()
        contour_plot = plt.contour(x, y, z, levels, colors='k', linewidths=1., linestyles=None, **kwargs)
        plt.clabel(contour_plot, fontsize=8, inline=1)
        plt.xlim(bounds)
        plt.ylim(bounds)
        plt.title(title)

        return fig

    def plot_cdf(self, plot_type: str = '3d', grid_size: int = 50, levels: list = None, **kwargs) -> plt.axis:
        title = "Copula CDF"

        bounds = [0 + 1e-2, 1 - 1e-2]
        u_grid, v_grid = np.meshgrid(
            np.linspace(bounds[0], bounds[1], grid_size),
            np.linspace(bounds[0], bounds[1], grid_size))

        z = np.array(
            [self.C(u, v) for u, v in zip(np.ravel(u_grid), np.ravel(v_grid))])

        z = z.reshape(u_grid.shape)

        if plot_type == "3d":
            ax = self._3d_surface_plot(u_grid, v_grid, z, [0, 1], title, **kwargs)

        elif plot_type == "contour":
            # Calculate levels if not given by user
            if not levels:
                min_ = np.nanpercentile(z, 5)
                max_ = np.nanpercentile(z, 95)
                levels = np.round(np.linspace(min_, max_, num=5), 3)
            ax = self._2d_contour_plot(u_grid, v_grid, z, [0, 1], title, levels, **kwargs)

        else:
            raise ValueError('Only contour and 3d plot options are available.')

        return ax

    def plot_scatter(self, num_points: int = 100) -> Axes:
        samples = self.sample(num=num_points)
        ax = sns.kdeplot(x=samples[:, 0], y=samples[:, 1], shade=True)
        ax.set_title('Scatter/heat plot for generated copula samples.')

        return ax

    def plot_pdf(self, plot_type: str = '3d', grid_size: int = 50, levels: list = None, **kwargs) -> Figure:
        title = " Copula PDF"

        if plot_type == "3d":
            bounds = [0 + 1e-1 / 2, 1 - 1e-1 / 2]
        else:  # plot_type == "contour"
            bounds = [0 + 1e-2, 1 - 1e-2]

        u_grid, v_grid = np.meshgrid(
            np.linspace(bounds[0], bounds[1], grid_size),
            np.linspace(bounds[0], bounds[1], grid_size))

        z = np.array(
            [self.c(u, v) for u, v in zip(np.ravel(u_grid), np.ravel(v_grid))])

        z = z.reshape(u_grid.shape)

        if plot_type == "3d":
            ax = self._3d_surface_plot(u_grid, v_grid, z, [0, 1], title, **kwargs)

        elif plot_type == "contour":
            # Calculate levels if not given by user
            if not levels:
                min_ = np.nanpercentile(z, 5)
                max_ = np.nanpercentile(z, 95)
                levels = np.round(np.linspace(min_, max_, num=5), 3)
            ax = self._2d_contour_plot(u_grid, v_grid, z, [0, 1], title, levels, **kwargs)

        else:
            raise ValueError('Only contour and 3d plot options are available.')

        return ax
