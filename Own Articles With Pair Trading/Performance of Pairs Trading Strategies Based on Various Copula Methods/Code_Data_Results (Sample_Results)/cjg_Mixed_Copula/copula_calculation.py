from typing import Callable, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF

from base import Copula
from elliptical import StudentCopula, fit_nu_for_t_copula


def find_marginal_cdf(x: np.array, empirical: bool = True, **kwargs) -> Callable[[float], float]:
    # Make sure it is an np.array
    x = np.array(x)
    x = x[~np.isnan(x)]  # Delete nan values

    prob_floor = kwargs.get('prob_floor', 0.00001)
    prob_cap = kwargs.get('prob_cap', 0.99999)

    if empirical:
        # Use empirical cumulative density function on data
        fitted_cdf = lambda data: max(min(ECDF(x)(data), prob_cap), prob_floor) if not np.isnan(data) else np.nan
        # Vectorize so it works on an np.array
        v_fitted_cdf = np.vectorize(fitted_cdf)
        return v_fitted_cdf

    return None


def construct_ecdf_lin(train_data: np.array, upper_bound: float = 1 - 1e-5, lower_bound: float = 1e-5) -> Callable:
    train_data_np = np.array(train_data)  # Convert to numpy array for the next step in case the input is not
    train_data_np = train_data_np[~np.isnan(train_data_np)]  # Remove nan value from the array

    step_ecdf = ECDF(train_data_np)  # train an ecdf on all training data
    # Sorted unique elements. They are the places where slope changes for the cumulative density
    slope_changes = np.unique(np.sort(train_data_np))
    # Calculate the ecdf at the points of slope change
    sample_ecdf_at_slope_changes = np.array([step_ecdf(unique_value) for unique_value in slope_changes])
    # Linearly interpolate. Allowing extrapolation to catch data out of range
    # x: unique elements in training data; y: the ecdf value for those training data
    interp_ecdf = interp1d(slope_changes, sample_ecdf_at_slope_changes, assume_sorted=True, fill_value='extrapolate')

    # Implement the upper and lower bound the ecdf
    def bounded_ecdf(x):
        if np.isnan(x):  # Map nan input to nan
            result = np.NaN
        else:  # Apply the upper and lower bound
            result = max(min(interp_ecdf(x), upper_bound), lower_bound)

        return result

    # Vectorize it to work with arrays
    return np.vectorize(bounded_ecdf)


def to_quantile(data: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    column_count = len(data.columns)  # Number of columns
    cdf_lst = [None] * column_count  # List to store all marginal cdf functions
    quantile_data_lst = [None] * column_count  # List to store all quantile data in pd.Series

    # Loop through all columns
    for i in range(column_count):
        cdf_lst[i] = construct_ecdf_lin(data.iloc[:, i])
        quantile_data_lst[i] = data.iloc[:, i].map(cdf_lst[i])

    quantile_data = pd.concat(quantile_data_lst, axis=1)  # Form the quantile DataFrame

    return quantile_data, cdf_lst


def sic(log_likelihood: float, n: int, k: int = 1) -> float:
    sic_value = np.log(n) * k - 2 * log_likelihood

    return sic_value


def aic(log_likelihood: float, n: int, k: int = 1) -> float:
    aic_value = (2 * n / (n - k - 1)) * k - 2 * log_likelihood

    return aic_value


def hqic(log_likelihood: float, n: int, k: int = 1) -> float:
    hqic_value = 2 * np.log(np.log(n)) * k - 2 * log_likelihood

    return hqic_value


def scad_penalty(x: float, gamma: float, a: float) -> float:
    # Bool variables for branchless construction
    is_linear = (np.abs(x) <= gamma)
    is_quadratic = np.logical_and(gamma < np.abs(x), np.abs(x) <= a * gamma)
    is_constant = (a * gamma) < np.abs(x)

    # Assembling parts
    linear_part = gamma * np.abs(x) * is_linear
    quadratic_part = (2 * a * gamma * np.abs(x) - x ** 2 - gamma ** 2) / (2 * (a - 1)) * is_quadratic
    constant_part = (gamma ** 2 * (a + 1)) / 2 * is_constant

    return linear_part + quadratic_part + constant_part


def scad_derivative(x: float, gamma: float, a: float) -> float:
    part_1 = gamma * (x <= gamma)
    part_2 = gamma * (a * gamma - x) * ((a * gamma - x) > 0) / ((a - 1) * gamma) * (x > gamma)

    return part_1 + part_2


def adjust_weights(weights: np.array, threshold: float) -> np.array:
    raw_weights = np.copy(weights)
    # Filter out components that have low weights. Low weights will be 0
    filtered_weights = raw_weights * (raw_weights > threshold)
    # Normalize the filtered weights. Make the total weight a partition of [0, 1]
    scaler = np.sum(filtered_weights)
    adjusted_weights = filtered_weights / scaler

    return adjusted_weights


def fit_copula_to_empirical_data(x: np.array, y: np.array, copula: Copula) -> tuple:

    num_of_instances = len(x)  # Number of instances

    # Finding an inverse cumulative density distribution (quantile) for each stock price series
    s1_cdf = construct_ecdf_lin(x)
    s2_cdf = construct_ecdf_lin(y)

    # Quantile data for each stock w.r.t. their cumulative log return
    u1_series = s1_cdf(x)
    u2_series = s2_cdf(y)

    # 将 u1_series 和 u2_series 转换为 Pandas Series
    u1_series = pd.Series(u1_series)
    u2_series = pd.Series(u2_series)
    data = pd.DataFrame({'x': u1_series, 'y': u2_series})

    # Get log-likelihood value and the copula with parameters fitted to training data
    if copula == StudentCopula:
        fitted_nu = fit_nu_for_t_copula(u1_series, u2_series, nu_tol=0.05)
        copula_obj = StudentCopula(nu=fitted_nu, cov=None)
        copula_obj.fit(u1_series, u2_series)
        log_likelihood = copula_obj.get_log_likelihood_sum(u1_series, u2_series)
    else:
        # 对于 FJGMixCop，传入 DataFrame 类型的数据
        copula_obj = copula()
        log_likelihood = copula_obj.fit(data)

    # Information criterion for evaluating model fitting performance
    sic_value = sic(log_likelihood, n=num_of_instances)
    aic_value = aic(log_likelihood, n=num_of_instances)
    hqic_value = hqic(log_likelihood, n=num_of_instances)

    result_dict = {'Copula Name': copula_obj.copula_name,
                   'SIC': sic_value,
                   'AIC': aic_value,
                   'HQIC': hqic_value}

    return result_dict, copula_obj, s1_cdf, s2_cdf
