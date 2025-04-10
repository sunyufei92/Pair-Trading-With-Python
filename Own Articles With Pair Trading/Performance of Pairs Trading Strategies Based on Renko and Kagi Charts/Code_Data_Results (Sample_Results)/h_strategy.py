from itertools import compress, combinations

import pandas as pd
import numpy as np
from tqdm import tqdm


class HConstruction:

    def __init__(self, series: pd.Series, threshold: float, method: str = "Kagi"):

        self.series = series
        self.threshold = threshold
        self.method = method

        self.tau_a_index = None
        self.tau_b_index = None
        self.tau_a_direction = None
        self.tau_b_direction = None
        self.signals_contrarian = None
        self.signals_momentum = None

        # Dictionary for storing the results of the H-construction
        self.results = {
            "h_series": [self.series[0]],
            "direction": [0],
            "index": [0],
            "tau_a": [False],
            "tau_b": [False]
        }

        # Building the H-construction
        self._construct()

    def _process(self, i: int):

        # Calculating the gap between the current price and the latest price of the H-construction
        gap = (self.series[i] - self.results["h_series"][-1])
        direction = np.sign(gap)
        pre_direction = self.results["direction"][-1]

        if self.method == "Kagi":
            over_thresholds = abs(gap) >= self.threshold
            same_direction = (pre_direction == direction)

            # Handling tau_a0 and tau_b0
            if pre_direction == 0:
                if self.series[:i + 1].max() - self.series[:i + 1].min() >= self.threshold:
                    argmax = self.series[:i + 1].argmax()
                    argmin = self.series[:i + 1].argmin()
                    tau_a0 = min(argmin, argmax)
                    tau_b0 = max(argmin, argmax)
                    a_direction = 1 if tau_a0 == argmax else -1
                    self._append(tau_a0, a_direction, False)
                    self._append(tau_b0, -a_direction, True)

                return

            # Do nothing if the direction is turned, but the amount is not enough
            if not same_direction and not over_thresholds:
                return

            # reverse == True if the direction is turned with enough amount
            reverse = (not same_direction) and over_thresholds
            self._append(i, direction, reverse)

        elif self.method == "Renko":
            # Calculating the number of bricks needed
            num_bricks = abs(gap) // self.threshold

            # Appending each bricks to the results
            for _ in range(int(num_bricks)):
                reverse = (pre_direction != direction)
                self._append(i, direction, reverse)

        else:
            raise Exception("Incorrect method. "
                            "Please use one of the options "
                            "[\"Kagi\", \"Renko\"].")

    def _append(self, i: int, direction: int, reverse: bool):

        # Determining the price to append based on the method
        if self.method == "Kagi":
            price = self.series[i]

        else:
            price = self.results["h_series"][-1] + direction * self.threshold

        # Marking turning point if reverse == True
        # tau_a indicates whether it is a turning point
        # and tau_b indicates whether it is a confirmation point for turning point
        if reverse:
            self.results["tau_a"][-1] = True
            self.results["tau_a"].append(False)
            self.results["tau_b"].append(True)

        else:
            self.results["tau_a"].append(False)
            self.results["tau_b"].append(False)

        # Appending other information
        self.results["h_series"].append(price)
        self.results["direction"].append(direction)
        self.results["index"].append(i)

    def _construct(self):

        # Processing each element one by one
        for i in range(len(self.results["h_series"]), len(self.series)):
            self._process(i)

        index = self.results["index"]
        direction = self.results["direction"]
        tau_a = self.results["tau_a"]
        tau_b = self.results["tau_b"]

        # Determining the index values and the directions of turning points and confirmation points
        self.tau_a_index = list(compress(index, tau_a))
        self.tau_b_index = list(compress(index, tau_b))
        self.tau_a_direction = list(compress(direction, tau_a))
        self.tau_b_direction = list(compress(direction, tau_b))

        # Determining the signals
        self.signals_contrarian = pd.Series(0, index=self.series.index)
        self.signals_momentum = pd.Series(0, index=self.series.index)

        # The signals will be opposite to the directions of the turning confirmation points
        self.signals_contrarian[self.tau_b_index] = [-d for d in self.tau_b_direction]

        # The signals will be same to the directions of the turning confirmation points
        self.signals_momentum[self.tau_b_index] = self.tau_b_direction

    def h_inversion(self) -> int:

        # The number of times the series changes its direction will equal to the number of the confirmation points
        return len(self.tau_b_index)

    def h_distances(self, p: int = 1) -> float:

        summation = 0
        for i in range(1, len(self.tau_a_index)):
            diff = self.series[self.tau_a_index[i]] - self.series[self.tau_a_index[i - 1]]
            summation += abs(diff) ** p

        return summation

    def h_volatility(self, p: int = 1) -> float:

        return self.h_distances(p)/self.h_inversion()

    def extend_series(self, series: pd.Series):

        self.series = pd.concat([self.series, series])
        self._construct()

    def get_signals(self, method: str = "contrarian") -> pd.Series:

        if method == "contrarian":
            signals = self.signals_contrarian

        elif method == "momentum":
            signals = self.signals_momentum

        else:
            raise Exception("Incorrect method. "
                            "Please use one of the options "
                            "[\"contrarian\", \"momentum\"].")

        return signals


class HSelection:

    def __init__(self, data: pd.DataFrame, method: str = "Kagi"):

        self.data = data
        self.method = method
        self.length = len(data)
        self.minimum_length = self.length
        self.results = None

        self.stock_pool = list(data.columns)
        self.possible_pairs = list(combinations(self.stock_pool, 2))

    def _get_h_inversion(self, pair: tuple) -> tuple:

        data_needed = self.data[list(pair)].dropna(axis = 0)

        # Return (0, 0) if the data length after removing rows with NaN is less than the minimum required length
        # or if there is any negative value.
        if len(data_needed) < self.minimum_length or (data_needed.values < 0).any():
            return 0, 0

        # Forming the spread series
        # Calculated as: log(Pt) - log(Qt)
        series = np.log(data_needed[pair[0]]) - np.log(data_needed[pair[1]])

        # Use the standard deviation of the spread series as the threshold of the H-construction
        std = series.std()
        hc = HConstruction(series, std, self.method)

        return hc.h_inversion(), std

    def select(self, minimum_length: int = None):

        if minimum_length is not None:
            self.minimum_length = minimum_length
        else:
            self.minimum_length = self.length

        results = []
        for pair in tqdm(self.possible_pairs):
            h_inversion, std = self._get_h_inversion(pair)
            if h_inversion != 0 and std != 0:
                results.append([h_inversion, std, pair])

        # Sorting the results by H-inversion statistic
        self.results = sorted(results, key=lambda i: i[0], reverse=True)

    def get_pairs(self, num: int, method: str = "highest", allow_repeat: bool = False) -> list:

        if method not in ['highest', 'lowest']:
            raise Exception("Incorrect method. "
                            "Please use one of the options "
                            "[\"highest\", \"lowest\"].")

        if allow_repeat:
            chose_pairs = self.results[:num] if method == "highest" else self.results[-num:]

        else:
            chose_tickers = []
            chose_pairs = []

            results = self.results.copy()

            if method == "lowest":
                results.reverse()

            i = 0
            while len(chose_pairs) < num and i < len(results):
                tickers = results[i][2]
                if tickers[0] not in chose_tickers and tickers[1] not in chose_tickers:
                    chose_tickers.extend(tickers)
                    chose_pairs.append(results[i])

                i += 1

        return chose_pairs
