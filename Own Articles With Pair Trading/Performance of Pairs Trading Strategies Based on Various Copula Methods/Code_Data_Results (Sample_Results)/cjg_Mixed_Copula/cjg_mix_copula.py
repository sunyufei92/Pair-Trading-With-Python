import numpy as np
import pandas as pd
from scipy.optimize import minimize

import copula_calculation as ccalc
from archimedean import Gumbel, Clayton, Joe
from base import MixedCopula


class CJGMixCop(MixedCopula):

    def __init__(self, cop_params: tuple = None, weights: tuple = None):

        super().__init__('CJGMixCop')
        self.copula_name = 'CJGMixCop'  # 添加 copula_name 属性
        self.cop_params = cop_params
        self.weights = weights
        self.clayton_cop, self.joe_cop, self.gumbel_cop = None, None, None
    
        # 如果提供了 cop_params，则初始化各个 Copula
        if cop_params is not None:
            self.clayton_cop = Clayton(theta=self.cop_params[0])
            self.joe_cop = Joe(theta=self.cop_params[1])
            self.gumbel_cop = Gumbel(theta=self.cop_params[2])
    
        self.copulas = [self.clayton_cop, self.joe_cop, self.gumbel_cop]

    def fit(self, data: pd.DataFrame, max_iter: int = 25, gamma_scad: float = 0.6, a_scad: float = 6,
            weight_margin: float = 1e-2) -> float:
        
        # 确保 data 是 Pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=['x', 'y'])
    
        # 通过将原始数据映射到边缘 cdf，生成 quantile_data DataFrame
        quantile_data = data.multiply(0)
        cdf1 = ccalc.construct_ecdf_lin(data.iloc[:, 0])
        cdf2 = ccalc.construct_ecdf_lin(data.iloc[:, 1])
        quantile_data.iloc[:, 0] = data.iloc[:, 0].map(cdf1)
        quantile_data.iloc[:, 1] = data.iloc[:, 1].map(cdf2)
        # 拟合分位数数据
        weights, cop_params = self._fit_quantile_em(quantile_data, max_iter, gamma_scad, a_scad)
        # 后处理权重。将过小的权重舍弃
        weights = ccalc.adjust_weights(weights, threshold=weight_margin)
    
        # 内部构建拟合结果的参数和权重
        self.weights = weights
        self.cop_params = cop_params
        # 使用更新的参数更新 Copula
        self.clayton_cop = Clayton(theta=self.cop_params[0])
        self.joe_cop = Joe(theta=self.cop_params[1])
        self.gumbel_cop = Gumbel(theta=self.cop_params[2])
        # 用于 MixedCopula 超类的列表
        self.copulas = [self.clayton_cop, self.joe_cop, self.gumbel_cop]
    
        # 计算对数似然值之和作为拟合结果
        u1 = quantile_data.iloc[:, 0].to_numpy()
        u2 = quantile_data.iloc[:, 1].to_numpy()
        sum_log_likelihood = self._ml_qfunc(u1, u2, cop_params, weights, gamma_scad, a_scad, if_penalty=False)
    
        return sum_log_likelihood

    def _fit_quantile_em(self, quantile_data: pd.DataFrame, max_iter: int, gamma_scad: float,
                         a_scad: float) -> (np.array, np.array):
        """
        通过期望最大化（EM）方法从分位数数据中拟合 cop_params 和 weights。
    
        :param quantile_data: (pd.DataFrame) 用于拟合的分位数数据。
        :param max_iter: (int) 可选。EM 方法的最大迭代次数。默认值为 25。
        :param gamma_scad: (float) SCAD 惩罚项的调整参数。
        :param a_scad: (float) SCAD 惩罚项的调整参数。
        :return: (tuple) 拟合的权重 (3, ) np.array 和拟合的 cop_params (3, ) np.array。
        """
    
        # 初始猜测的权重和 Copula 参数
        init_weights = [0.33, 0.33, 1 - 0.33 - 0.33]
        init_cop_params = [2, 2, 2]  # 初始猜测的 Clayton、Joe 和 Gumbel 的 theta 参数
        # 使用初始猜测进行第一次计算
        weights = self._expectation_step(quantile_data, gamma_scad, a_scad, init_cop_params, init_weights)
        cop_params = self._maximization_step(quantile_data, gamma_scad, a_scad, init_cop_params, weights)
        # 我们要优化的完整参数，包括权重和 cop_params
        old_full_params = np.concatenate([init_weights, init_cop_params], axis=None)
        new_full_params = np.concatenate([weights, cop_params], axis=None)
    
        # 初始化循环条件
        l1_diff = np.linalg.norm(old_full_params - new_full_params, ord=1)
        i = 1
        # 当达到最大迭代次数或 l1 差异足够小，终止循环
        while i < max_iter and l1_diff > 1e-2:
            # 更新旧参数
            old_full_params = np.concatenate([weights, cop_params], axis=None)
            # 1. 期望步
            weights = self._expectation_step(quantile_data, gamma_scad, a_scad, cop_params, weights)
            # 2. 最大化步
            # 如果 Joe Copula 的权重较小，则使用替代的最大化步骤以提高性能
            if weights[1] < 1e-2:
                weights = ccalc.adjust_weights(weights, threshold=1e-2)
                cop_params = self._maximization_step_no_joe(quantile_data, gamma_scad, a_scad, cop_params, weights)
            else:
                cop_params = self._maximization_step(quantile_data, gamma_scad, a_scad, cop_params, weights)
            # 更新新参数
            new_full_params = np.concatenate([weights, cop_params], axis=None)
            # 更新 l1 差异和计数器
            l1_diff = np.linalg.norm(old_full_params - new_full_params, ord=1)
            i += 1
    
        return weights, cop_params

    @staticmethod
    def _expectation_step(quantile_data: pd.DataFrame, gamma_scad: float, a_scad: float,
                          cop_params: list, weights: list) -> np.array:
        """
        EM 方法中用于拟合混合 Copula 的期望步骤。
    
        :param quantile_data: (pd.DataFrame) 用于拟合的分位数数据。
        :param gamma_scad: (float) SCAD 惩罚项的调整参数。
        :param a_scad: (float) SCAD 惩罚项的调整参数。
        :param cop_params: (list) 形状为 (3, )，用于依赖性的 Copula 参数。这是其初始猜测值。
        :param weights: (list) 形状为 (3, )，混合 Copula 的权重。
        :return: (np.array) 形状为 (3, )，更新后的权重。
        """
    
        num = len(quantile_data)  # 数据点数量
    
        # 使用 numpy 数组实现底层密度计算
        u1 = quantile_data.iloc[:, 0].to_numpy()
        u2 = quantile_data.iloc[:, 1].to_numpy()
    
        # 迭代步骤以找到权重
        # 初始化迭代计算的参数
        diff = 1  # 更新的权重与上一步相比的差异
        tol_weight = 1e-2  # 当权重差异 <= tol_weight，结束循环
        iteration = 0
        # 使用给定的 cop_params 初始化 Copula（cop_params 在此方法中不变）
        local_copulas = [Clayton(theta=cop_params[0]),
                         Joe(theta=cop_params[1]),
                         Gumbel(theta=cop_params[2])]
    
        # 当差异 <= 容差或迭代次数超过 10，退出循环
        while diff > tol_weight and iteration < 10:  # 少量迭代以避免过拟合
            new_weights = np.array([np.nan] * 3)
            iteration += 1
            for i in range(3):  # 对于混合 Copula 的每个组件
                # 计算 (Cai et al. 2014) 公式中的各个部分
                sum_ml_lst = u1 * 0
                for t in range(num):  # 对于每个数据点
                    denominator = np.sum([weights[j] * local_copulas[j].get_cop_density(u=u1[t], v=u2[t])
                                          for j in range(3)])
                    numerator = weights[i] * local_copulas[i].get_cop_density(u=u1[t], v=u2[t])
                    sum_ml_lst[t] = numerator / denominator
    
                sum_ml = np.sum(sum_ml_lst)
                numerator = weights[i] * ccalc.scad_derivative(weights[i], gamma_scad, a_scad) - sum_ml / num
                denominator = np.sum([weight * ccalc.scad_derivative(weight, gamma_scad, a_scad)
                                      for weight in weights]) - 1
                new_weights[i] = numerator / denominator
    
            # 以 l1 范数定义差异
            diff = np.sum(np.abs(weights - new_weights))
            weights = np.copy(new_weights)  # 仅取值
    
        return weights

    def _maximization_step(self, quantile_data: pd.DataFrame, gamma_scad: float, a_scad: float, cop_params: list,
                           weights: list) -> np.array:
        """
        EM 方法中用于拟合混合 Copula 的最大化步骤。
    
        :param quantile_data: (pd.DataFrame) 用于拟合的分位数数据。
        :param gamma_scad: (float) SCAD 惩罚项的调整参数。
        :param a_scad: (float) SCAD 惩罚项的调整参数。
        :param cop_params: (list) 形状为 (3, ), Copula 参数的初始猜测。
        :param weights: (list) 形状为 (3, ), 混合 Copula 的权重。
        :return: (np.array) 形状为 (3, ), 更新后的 Copula 参数。
        """
    
        # 使用 numpy 数组实现底层密度计算
        u1 = quantile_data.iloc[:, 0].to_numpy()
        u2 = quantile_data.iloc[:, 1].to_numpy()
        eps = 1e-3  # 最小化容差
        cop_params = np.array(cop_params)  # 转换为 numpy 数组
    
        # 定义目标函数 -Q
        def q_func(my_cop_params):
            # 计算关于 cop_params 的数值梯度
            result = self._ml_qfunc(u1, u2, my_cop_params, weights, gamma_scad, a_scad, multiplier=-1)
    
            return result  # - (最大似然 - 惩罚)
    
        # 使用 L-BFGS-B 方法找到使 -Q 最小的 cop_params
        init_cop_params = cop_params  # Copula 参数的初始猜测
        # 参数边界：根据 Copula 的定义调整边界
        bnds = ((eps, None), (1 + eps, None), (eps, None))
    
        res = minimize(q_func, x0=init_cop_params, method='L-BFGS-B', bounds=bnds,
                       options={'disp': False, 'maxiter': 20}, tol=0.1)
    
        return res.x  # 返回更新后的 Copula 参数

    @staticmethod
    def _ml_qfunc(u1: np.array, u2: np.array, cop_params: list, weights: list,
                  gamma_scad: float, a_scad: float, if_penalty: bool = True, multiplier: float = 1) -> float:
        """
        EM 方法中要最小化的目标函数。通常在文献中表示为 Q。
    
        它是对数似然减去 SCAD 惩罚项。
    
        :param u1: (np.array) 1D 向量数据。需要在 [0, 1] 内均匀分布。
        :param u2: (np.array) 1D 向量数据。需要在 [0, 1] 内均匀分布。
        :param cop_params: (list) 形状为 (3, ), Copula 参数。
        :param weights: (list) 形状为 (3, ), 混合 Copula 的权重。
        :param gamma_scad: (float) SCAD 惩罚项的调整参数。
        :param a_scad: (float) SCAD 惩罚项的调整参数。
        :param if_penalty: (bool) 可选。是否添加 SCAD 惩罚项。默认为 True。
        :param multiplier: (float) 可选。将计算结果乘以一个数。当优化算法搜索最小值而非最大值时，通常使用 -1。默认为 1。
        :return: (float) 目标函数的值。
        """
    
        num = len(u1)
        # 为了可读性，重新分配 Copula 参数和权重
        theta_c, theta_j, theta_g = cop_params
        weight_c, weight_j, _ = weights
    
        # 创建用于密度计算的本地 Copula
        clayton_cop = Clayton(theta=theta_c)
        joe_cop = Joe(theta=theta_j)
        gumbel_cop = Gumbel(theta=theta_g)
    
        # 计算混合 Copula 在数据上的对数似然
        likelihood_list_clayton = np.array([clayton_cop.get_cop_density(u1_i, u2_i) for (u1_i, u2_i) in zip(u1, u2)])
        likelihood_list_joe = np.array([joe_cop.get_cop_density(u1_i, u2_i) for (u1_i, u2_i) in zip(u1, u2)])
        likelihood_list_gumbel = np.array([gumbel_cop.get_cop_density(u1_i, u2_i) for (u1_i, u2_i) in zip(u1, u2)])
        likelihood_list_mix = (weight_c * likelihood_list_clayton + weight_j * likelihood_list_joe
                               + (1 - weight_c - weight_j) * likelihood_list_gumbel)
        log_likelihood_sum = np.sum(np.log(likelihood_list_mix))
    
        # 计算惩罚项
        penalty = num * np.sum([ccalc.scad_penalty(weights[k], gamma=gamma_scad, a=a_scad) for k in range(3)])
    
        return (log_likelihood_sum - penalty * int(if_penalty)) * multiplier

    def _maximization_step_no_joe(self, quantile_data: pd.DataFrame, gamma_scad: float, a_scad: float,
                                  cop_params: list, weights: list) -> np.array:
        """
        当 Joe Copula 权重较小时，EM 方法中的最大化步骤。
        """
        # 使用 numpy 数组实现底层密度计算
        u1 = quantile_data.iloc[:, 0].to_numpy()
        u2 = quantile_data.iloc[:, 1].to_numpy()
        cop_params = np.array(cop_params)  # 形状为 (3, )
    
        # 定义 eps
        eps = 1e-3
    
        # 定义目标函数 -Q。
        def q_func_no_joe(my_cop_params):  # 形状为 (2, ); weights 形状为 (3, )
            # 计算目标函数
            result = self._ml_qfunc_no_joe(u1, u2, my_cop_params, weights, gamma_scad, a_scad, multiplier=-1)
            return result  # - (max_likelihood - penalty)
    
        # 初始猜测仅包含 Clayton 和 Frank Copula 的参数
        init_cop_params = np.array([cop_params[0], cop_params[2]])  # 形状为 (2, )
    
        # 参数边界：theta_c >= eps, theta_f 无限制
        bnds = ((eps, None), (eps, None))
    
        res = minimize(q_func_no_joe, x0=init_cop_params, method='L-BFGS-B', bounds=bnds,
                       options={'disp': False, 'maxiter': 20}, tol=0.1)
    
        # 更新参数，不改变 Joe Copula 的参数
        params_without_updating_joe = np.array([res.x[0], cop_params[1], res.x[1]])
        return params_without_updating_joe  # 返回更新后的 Copula 参数
    
    @staticmethod
    def _ml_qfunc_no_joe(u1: np.array, u2: np.array, cop_params: list, weights: list,
                         gamma_scad: float, a_scad: float, if_penalty: bool = True, multiplier: float = 1) -> float:
        """
        在没有更新 Joe Copula 参数时，EM 方法中要最小化的目标函数。
    
        :param u1: (np.array) 1D 向量数据。需要在 [0, 1] 内均匀分布。
        :param u2: (np.array) 1D 向量数据。需要在 [0, 1] 内均匀分布。
        :param cop_params: (list) 形状为 (2, ), Copula 参数。
        :param weights: (list) 形状为 (3, ), 混合 Copula 的权重。
        :param gamma_scad: (float) SCAD 惩罚项的调整参数。
        :param a_scad: (float) SCAD 惩罚项的调整参数。
        :param if_penalty: (bool) 可选。是否添加 SCAD 惩罚项。默认为 True。
        :param multiplier: (float) 可选。将计算结果乘以一个数。当优化算法搜索最小值而非最大值时，通常使用 -1。默认为 1。
        :return: (float) 目标函数的值。
        """
    
        num = len(u1)
        # 为了可读性，重新分配 Copula 参数和权重
        theta_c, theta_g = cop_params
        weight_c, _, _ = weights  # 不使用 Joe Copula 的权重
    
        # 创建用于密度计算的本地 Copula
        clayton_cop = Clayton(theta=theta_c)
        gumbel_cop = Gumbel(theta=theta_g)
    
        # 计算混合 Copula 在数据上的对数似然
        likelihood_list_clayton = np.array([clayton_cop.get_cop_density(u1_i, u2_i) for (u1_i, u2_i) in zip(u1, u2)])
        likelihood_list_gumbel = np.array([gumbel_cop.get_cop_density(u1_i, u2_i) for (u1_i, u2_i) in zip(u1, u2)])
        likelihood_list_mix = weight_c * likelihood_list_clayton + (1 - weight_c) * likelihood_list_gumbel
        log_likelihood_sum = np.sum(np.log(likelihood_list_mix))
    
        # 计算惩罚项
        penalty = num * np.sum([ccalc.scad_penalty(weights[k], gamma=gamma_scad, a=a_scad) for k in range(3)])
    
        return (log_likelihood_sum - penalty * int(if_penalty)) * multiplier

    
    def _get_param(self) -> dict:
        """
        获取此混合 Copula 实例的名称和参数。
    
        :return: (dict) 此 Copula 的名称和参数。
        """
    
        descriptive_name = 'Bivariate Clayton-Joe-Gumbel Mixed Copula'
        class_name = 'CJGMixCop'
        cop_params = self.cop_params
        weights = self.weights
        info_dict = {'Descriptive Name': descriptive_name,
                     'Class Name': class_name,
                     'Clayton theta': cop_params[0], 'Joe theta': cop_params[1], 'Gumbel theta': cop_params[2],
                     'Clayton weight': weights[0], 'Joe weight': weights[1], 'Gumbel weight': weights[2]}
    
        return info_dict
