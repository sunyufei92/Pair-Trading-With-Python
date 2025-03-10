
# pylint: disable=broad-exception-raised)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DistanceStrategy:

    def __init__(self):
        """
        Initialize Distance strategy.
        """

        # Internal parameters
        self.min_normalize = None  # Minimum values for each price series used for normalization
        self.max_normalize = None  # Maximum values for each price series used for normalization
        self.pairs = None  # Created pairs after the form_pairs stage
        self.train_std = None  # Historical volatility for each chosen pair portfolio
        self.normalized_data = None  # Normalized test dataset
        self.portfolios = None  # Pair portfolios composed from test dataset
        self.train_portfolio = None  # Pair portfolios composed from train dataset
        self.trading_signals = None  # Final trading signals
        self.num_crossing = None  # Number of zero crossings from train dataset

    def form_pairs(self, train_data, method='standard', industry_dict=None, num_top=20, skip_top=0, selection_pool=10000,
                   list_names=None):

        # If np.array given as an input
        if isinstance(train_data, np.ndarray):
            train_data = pd.DataFrame(train_data, columns=list_names)

        # Normalizing input data
        normalized, self.min_normalize, self.max_normalize = self.normalize_prices(train_data)

        # Dropping observations with missing values (for distance calculation)
        normalized = normalized.dropna(axis=0)

        # If industry dictionary is given, pairs are matched within the same industry group
        all_pairs = self.find_pair(normalized, industry_dict)

        # Choosing needed pairs to construct a portfolio
        self.pairs = self.sort_pairs(all_pairs, selection_pool)

        # Calculating historical volatility of pair portfolios (diffs of normalized prices)
        self.train_std = self.find_volatility(normalized, self.pairs)

        # Creating portfolios for pairs chosen in the pairs formation stage with train dataset
        self.train_portfolio = self.find_portfolios(normalized, self.pairs)

        # Calculating the number of zero crossings from the dataset
        self.num_crossing = self.count_number_crossing()

        # In case of a selection method other than standard or industry is used, sorting paris
        # based on the method
        self.selection_method(method, num_top, skip_top)

        # Storing only the necessary values for pairs selected in the above
        self.num_crossing = {pair: self.num_crossing[pair] for pair in self.pairs}
        self.train_std = {pair: self.train_std[pair] for pair in self.pairs}
        self.train_portfolio = self.train_portfolio[self.train_portfolio.columns
                                                        .intersection([str(pair) for pair in self.pairs])]

    def selection_method(self, method, num_top, skip_top):

        if method not in ['standard', 'zero_crossing', 'variance']:
            # Raise an error if the given method is inappropriate.
            raise Exception("Please give an appropriate method for sorting pairs between ‘standard’, "
                            "‘zero_crossing’, or 'variance'")

        if method == 'standard':

            self.pairs = self.pairs[skip_top:(skip_top + num_top)]

        elif method == 'zero_crossing':

            # Sorting pairs from the dictionary by the number of zero crossings in a descending order
            sorted_pairs = sorted(self.num_crossing.items(), key=lambda x: x[1], reverse=True)

            # Picking top pairs
            pairs_selected = sorted_pairs[skip_top:(skip_top + num_top)]

            # Removing the number of crossings, so we have only tuples with elements
            pairs_selected = [x[0] for x in pairs_selected]

            self.pairs = pairs_selected

        else:

            # Sorting pairs from the dictionary by the size of variance in a descending order
            sorted_pairs = sorted(self.train_std.items(), key=lambda x: x[1], reverse=True)

            # Picking top pairs
            pairs_selected = sorted_pairs[skip_top:(skip_top + num_top)]

            # Removing the variance, so we have only tuples with elements
            pairs_selected = [x[0] for x in pairs_selected]

            self.pairs = pairs_selected

    def trade_pairs(self, test_data, divergence=2):

        # If np.array given as an input
        if isinstance(test_data, np.ndarray):
            test_data = pd.DataFrame(test_data, columns=self.min_normalize.index)

        # If the pairs formation step wasn't performed
        if self.pairs is None:
            raise Exception("Pairs are not defined. Please perform the form_pairs() step first.")

        # Normalizing the testing data with min and max values obtained from the training data
        self.normalized_data, _, _ = self.normalize_prices(test_data, self.min_normalize, self.max_normalize)

        # Creating portfolios for pairs chosen in the pairs formation stage
        self.portfolios = self.find_portfolios(self.normalized_data, self.pairs)

        # Creating trade signals for pair portfolios
        self.trading_signals = self.signals(self.portfolios, self.train_std, divergence)

    def get_signals(self):

        return self.trading_signals

    def get_portfolios(self):

        return self.portfolios

    def get_scaling_parameters(self):

        scale = pd.DataFrame()

        scale['min_value'] = self.min_normalize
        scale['max_value'] = self.max_normalize

        return scale

    def get_pairs(self):

        return self.pairs

    def get_num_crossing(self):

        return self.num_crossing

    def count_number_crossing(self):

        # Creating a dictionary for number of zero crossings
        num_zeros_dict = {}

        # Iterating through pairs
        for pair in self.train_portfolio:
            # Getting names of individual elements from dataframe column names
            pair_val = pair.strip('\')(\'').split('\', \'')
            pair_val = tuple(pair_val)

            # Check if portfolio price crossed zero
            portfolio = self.train_portfolio[pair].to_frame()
            pair_mult = portfolio * portfolio.shift(1)

            # Get the number of zero crossings for the portfolio
            num_zero_crossings = len(portfolio[pair_mult.iloc[:, 0] <= 0].index)

            # Adding the pair's number of zero crossings to the dictionary
            num_zeros_dict[pair_val] = num_zero_crossings

        return num_zeros_dict

    def plot_portfolio(self, num_pair):

        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 7))
        fig.suptitle('Distance Strategy results for portfolio' + self.trading_signals.columns[num_pair])

        axs[0].plot(self.portfolios[self.trading_signals.columns[num_pair]])
        axs[0].title.set_text('Portfolio value (the difference between element prices)')

        axs[1].plot(self.trading_signals[self.trading_signals.columns[num_pair]], '#b11a21')
        axs[1].title.set_text('Number of portfolio units to hold')

        return fig

    def plot_pair(self, num_pair):

        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 7))
        fig.suptitle('Distance Strategy results for pair' + self.trading_signals.columns[num_pair])

        pair_val = self.trading_signals.columns[num_pair].strip('\')(\'').split('\', \'')
        pair_val = tuple(pair_val)

        axs[0].plot(self.normalized_data[pair_val[0]], label="Long asset in a portfolio - " + pair_val[0])
        axs[0].plot(self.normalized_data[pair_val[1]], label="Short asset in a portfolio - " + pair_val[1])
        axs[0].legend()
        axs[0].title.set_text('Price of elements in a portfolio.')

        axs[1].plot(self.trading_signals[self.trading_signals.columns[num_pair]], '#b11a21')
        axs[1].title.set_text('Number of portfolio units to hold')

        return fig

    @staticmethod
    def normalize_prices(data, min_values=None, max_values=None):

        # If normalization parameters are not given, calculate
        if (max_values is None) or (min_values is None):
            max_values = data.max()
            min_values = data.min()

        # Normalizing the dataset
        data_copy = data.copy()
        normalized = (data_copy - min_values) / (max_values - min_values)

        return normalized, min_values, max_values

    @staticmethod
    def find_pair(data, industry_dict=None):

        # Creating a dictionary
        pairs = {}

        # Iterating through each element in dataframe
        for ticker in data:

            # Removing the chosen element from the dataframe
            data_excluded = data.drop([ticker], axis=1)

            # Removing tickers in different industry group if the industry dictionary is given
            if industry_dict is not None:
                # Getting the industry group for the ticker
                industry_group = industry_dict[ticker]
                # Getting the tickers within the same industry group
                tickers_same_industry = [ticker for ticker, industry in industry_dict.items()
                                         if industry == industry_group]
                # Removing other tickers in different industry group
                data_excluded = data_excluded.loc[:, data_excluded.columns.isin(tickers_same_industry)]

            # Calculating differences between prices
            data_diff = data_excluded.sub(data[ticker], axis=0)

            # Calculating the sum of square differences
            sum_sq_diff = (data_diff ** 2).sum()

            # Iterating through second elements
            for second_element in sum_sq_diff.index:
                # Adding all new pairs to the dictionary
                pairs[tuple(sorted((ticker, second_element)))] = sum_sq_diff[second_element]

        return pairs

    @staticmethod
    def sort_pairs(pairs, num_top=20, skip_top=0):

        # Sorting pairs from the dictionary by distances in an ascending order
        sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)

        # Picking top pairs
        top_pairs = sorted_pairs[skip_top:(skip_top + num_top)]

        # Removing distance values, so we have only tuples with elements
        top_pairs = [x[0] for x in top_pairs]

        return top_pairs

    @staticmethod
    def find_volatility(data, pairs):

        # Creating a dictionary
        volatility_dict = {}

        # Iterating through pairs of elements
        for pair in pairs:
            # Getting two price series for elements in a pair
            par = data[list(pair)]

            # Differences between picked price series
            par_diff = par.iloc[:, 0] - par.iloc[:, 1]

            # Calculating standard deviation for difference series
            st_div = par_diff.std()

            # Adding pair and volatility to dictionary
            volatility_dict[pair] = st_div

        return volatility_dict

    @staticmethod
    def find_portfolios(data, pairs):

        # 创建一个空的字典用于存储每个组合
        portfolio_dict = {}

        # 迭代每一个对
        for pair in pairs:
            # 计算价差
            par_diff = data.loc[:, pair[0]] - data.loc[:, pair[1]]
            # 将价差存入字典，键为对的名称
            portfolio_dict[str(pair)] = par_diff

        # 使用 pd.concat 一次性将所有组合添加到 DataFrame
        portfolios = pd.concat(portfolio_dict, axis=1)

        return portfolios

    @staticmethod
    def signals(portfolios, variation, divergence):

        # Creating a signals dataframe
        signals = pd.DataFrame()

        # Iterating through pairs
        for pair in portfolios:
            # Getting names of individual elements from dataframe column names
            pair_val = pair.strip('\')(\'').split('\', \'')
            pair_val = tuple(pair_val)

            # Historical standard deviation for a pair
            st_dev = variation[pair_val]

            # Check if portfolio price crossed zero
            portfolio = portfolios[pair].to_frame()
            pair_mult = portfolio * portfolio.shift(1)

            # Entering a short position when portfolio is higher than divergence * st_dev
            short_entry_index = portfolio[portfolio.iloc[:, 0] > divergence * st_dev].index
            short_exit_index = portfolio[pair_mult.iloc[:, 0] <= 0].index

            # Entering a long position in the opposite situation
            long_entry_index = portfolio[portfolio.iloc[:, 0] < -divergence * st_dev].index
            long_exit_index = portfolio[pair_mult.iloc[:, 0] <= 0].index

            # Transforming long and short trading signals into one signal - target quantity
            portfolio['long_units'] = np.nan
            portfolio['short_units'] = np.nan
            portfolio.iloc[0, portfolio.columns.get_loc('long_units')] = 0
            portfolio.iloc[0, portfolio.columns.get_loc('short_units')] = 0

            portfolio.loc[long_entry_index, 'long_units'] = 1
            portfolio.loc[long_exit_index, 'long_units'] = 0
            portfolio.loc[short_entry_index, 'short_units'] = -1
            portfolio.loc[short_exit_index, 'short_units'] = 0

            portfolio.ffill(inplace=True)  # 替换 fillna 为 ffill
            portfolio['target_quantity'] = portfolio['long_units'] + portfolio['short_units']

            # Adding target quantity to signals dataframe
            signals[str(pair)] = portfolio['target_quantity']

        return signals
    


    def get_portfolio_daily_return(self, test_data):

        # 获取每个选定的组合对
        pairs = self.get_pairs()
    
        # 使用(2 * 标准差)作为阈值进行交易信号生成
        self.trade_pairs(test_data, divergence=2)
    
        # 获取组合对的组合值、交易信号及元素的归一化价格系列
        index_columns = [str(pair) for pair in pairs]
        portfolio_series = self.get_portfolios()
        portfolio_series = portfolio_series.loc[:, portfolio_series.columns.isin(index_columns)]
        trading_signals = self.get_signals()
        trading_signals = trading_signals.loc[:, trading_signals.columns.isin(index_columns)]
    
        # 测试数据集中的元素收益率
        test_data_returns = (test_data / test_data.shift(1) - 1)[1:]
    
        # 创建空的Series和列表来存储组合对的回报
        total_daily_return = pd.Series(0, index=test_data_returns.index)
        pair_daily_return = []
    
        for pair in pairs:
            first_stock, second_stock = pair
            # 计算单独的每日回报率
            daily_return = test_data_returns[first_stock] * 0.5 - test_data_returns[second_stock] * 0.5
            # 调整每日回报根据信号
            adjusted_daily_return = daily_return * (trading_signals[str(pair)].shift(1).fillna(0))
            
            # 添加到列表
            pair_daily_return.append(adjusted_daily_return)
            # 将调整后的每日回报添加到总的每日回报
            total_daily_return = total_daily_return.add(adjusted_daily_return, fill_value=0)
    
        # 将组合对的回报重新格式化，便于后续可视化
        pair_daily_return = [f'{value:.4f}' for value in [pd.Series(x).sum() for x in pair_daily_return]]
    
        # 计算组合的平均每日回报
        total_daily_return = total_daily_return / len(pairs)
    
        return pair_daily_return, total_daily_return


    def get_portfolio_daily_return_with_fees(self, test_data):

        # 获取每个选定的组合对
        pairs = self.get_pairs()
    
        # 使用(2 * 标准差)作为阈值进行交易信号生成
        self.trade_pairs(test_data, divergence=2)
    
        # 获取组合对的组合值、交易信号及元素的归一化价格系列
        index_columns = [str(pair) for pair in pairs]
        portfolio_series = self.get_portfolios()
        portfolio_series = portfolio_series.loc[:, portfolio_series.columns.isin(index_columns)]
        trading_signals = self.get_signals()
        trading_signals = trading_signals.loc[:, trading_signals.columns.isin(index_columns)]
    
        # 测试数据集中的元素收益率
        test_data_returns = (test_data / test_data.shift(1) - 1)[1:]
    
        # 创建空的Series和列表来存储组合对的回报
        total_daily_return = pd.Series(0.0, index=test_data_returns.index)
        pair_daily_return = []
    
        # 定义买入和卖出的费用百分比
        buy_cost_bps = 0.0050  # 买入费用百分比
        sell_cost_bps = 0.0050  # 卖出费用百分比
    
        for pair in pairs:
            first_stock, second_stock = pair
            # 计算单独的每日回报率
            daily_return = test_data_returns[first_stock] * 0.5 - test_data_returns[second_stock] * 0.5
            # 调整每日回报根据信号
            adjusted_daily_return = daily_return * (trading_signals[str(pair)].shift(1).fillna(0))
    
            # 计算买入和卖出费用
            buy_value = test_data[first_stock] * 0.5
            sell_value = test_data[second_stock] * 0.5
            transaction_cost = (buy_cost_bps * buy_value + sell_cost_bps * sell_value) * abs(trading_signals[str(pair)].diff().fillna(0))
    
            # 调整每日回报
            for i in range(len(adjusted_daily_return)):
                total_cost = transaction_cost.iloc[i]
                total_value = (buy_value.iloc[i] + sell_value.iloc[i])
                transaction_cost_as_pct = total_cost / total_value if total_value != 0 else 0
                adjusted_daily_return.iloc[i] -= transaction_cost_as_pct
    
            # 添加到列表
            pair_daily_return.append(adjusted_daily_return)
            # 将调整后的每日回报添加到总的每日回报
            total_daily_return = total_daily_return.add(adjusted_daily_return, fill_value=0)
    
        # 将组合对的回报重新格式化，便于后续可视化
        pair_daily_return = [f'{value:.4f}' for value in [pd.Series(x).sum() for x in pair_daily_return]]
    
        # 计算组合的平均每日回报
        total_daily_return = total_daily_return / len(pairs)
    
        return pair_daily_return, total_daily_return
