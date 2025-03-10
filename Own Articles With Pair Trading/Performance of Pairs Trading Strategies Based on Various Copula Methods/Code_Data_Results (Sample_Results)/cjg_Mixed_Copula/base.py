from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from basic_base import Copula


class MixedCopula(Copula, ABC):
    def __init__(self, copula_name: str):
        super().__init__(copula_name)
        self.weights = None
        self.copulas = None

    def describe(self) -> pd.Series:
        description = pd.Series(self._get_param())

        return description

    @abstractmethod
    def _get_param(self):
        """
        Get the parameters of the mixed copula.
        """

    def get_cop_density(self, u: float, v: float, eps: float = 1e-5) -> float:
        # Mapping u, v back to the valid computational interval
        u = min(max(eps, u), 1 - eps)
        v = min(max(eps, v), 1 - eps)

        # linear combo w.r.t. weights for each copula in the mix
        pdf = np.sum([self.weights[i] * cop.c(u, v) for i, cop in enumerate(self.copulas)])

        return pdf

    def get_cop_eval(self, u: float, v: float, eps: float = 1e-4) -> float:
        # Mapping u, v back to the valid computational interval
        u = min(max(eps, u), 1 - eps)
        v = min(max(eps, v), 1 - eps)

        # linear combo w.r.t. weights for each copula in the mix
        cdf = np.sum([self.weights[i] * cop.C(u, v) for i, cop in enumerate(self.copulas)])

        return cdf

    def get_condi_prob(self, u: float, v: float, eps: float = 1e-5) -> float:
        # Mapping u, v back to the valid computational interval
        u = min(max(eps, u), 1 - eps)
        v = min(max(eps, v), 1 - eps)

        # linear combo w.r.t. weights for each copula in the mix
        result = np.sum([self.weights[i] * cop.condi_cdf(u, v) for i, cop in enumerate(self.copulas)])

        return result

    def sample(self, num: int) -> np.array:
        # Generate a list of identities in {0, 1, 2} with given probability to determine which copula each
        # observation comes from. e.g. cop_identities[100, 2] means the 100th observation comes from copula 2
        cop_identities = np.random.choice([0, 1, 2], num, p=self.weights)

        # Generate random pairs from the copula given by cop_identities
        sample_pairs = np.zeros(shape=(num, 2))
        for i, cop_id in enumerate(cop_identities):
            sample_pairs[i] = self.copulas[cop_id].sample(num=1).flatten()

        return sample_pairs

    @staticmethod
    def _away_from_0(x: float, lower_limit: float = -1e-5, upper_limit: float = 1e-5) -> float:
        small_pos_bool = (0 <= x < upper_limit)  # Whether it is a small positive number
        small_neg_bool = (lower_limit < x < 0)  # Whether it is a small negative number
        small_bool = small_pos_bool or small_neg_bool  # Whether it is a small number
        # If not small, then return the param
        # If small, then return the corresponding limit
        remapped_param = (x * int(not small_bool)
                          + upper_limit * int(small_pos_bool) + lower_limit * int(small_neg_bool))

        return remapped_param
