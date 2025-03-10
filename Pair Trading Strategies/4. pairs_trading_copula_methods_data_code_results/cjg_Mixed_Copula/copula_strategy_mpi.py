from typing import Callable, Sequence
import numpy as np
import pandas as pd


class MPICopulaTradingRule:

    def __init__(self, opening_triggers: tuple = (-0.6, 0.6), stop_loss_positions: tuple = (-2, 2)):

        self.opening_triggers = opening_triggers
        self.stop_loss_positions = stop_loss_positions

        # Counters on how many times each position is triggered
        self._long_count = 0
        self._short_count = 0
        self._exit_count = 0

        self.copula = None  # Fit copula
        self.cdf_x = None
        self.cdf_y = None

    def set_copula(self, copula: object):

        self.copula = copula

    def set_cdf(self, cdf_x: Callable[[float], float], cdf_y: Callable[[float], float]):

        self.cdf_x = cdf_x
        self.cdf_y = cdf_y

    @staticmethod
    def to_returns(pair_prices: pd.DataFrame, fill_init_nan: Sequence[float] = (0, 0)) -> pd.DataFrame:

        returns = pair_prices.pct_change()
        returns.iloc[0, 0] = fill_init_nan[0]
        returns.iloc[0, 1] = fill_init_nan[1]

        return returns

    def calc_mpi(self, returns: pd.DataFrame) -> pd.DataFrame:

        # Convert to quantile data
        quantile_c1 = returns.iloc[:, 0].map(self.cdf_x)
        quantile_c2 = returns.iloc[:, 1].map(self.cdf_y)
        quantile_data = pd.concat([quantile_c1, quantile_c2], axis=1)
        # Calculate conditional probabilities using returns and cdfs. This is the definition of MPI
        mpis = self.get_condi_probs(quantile_data)

        return mpis

    def get_condi_probs(self, quantile_data: pd.DataFrame) -> pd.DataFrame:

        # Initiate a data frame with zeros and the same index
        condi_probs = pd.DataFrame(np.nan, index=quantile_data.index, columns=quantile_data.columns)

        for row_count, row in enumerate(quantile_data.iterrows()):
            condi_probs.iloc[row_count] = [self.copula.get_condi_prob(row[1].iloc[0], row[1].iloc[1]),
                                           self.copula.get_condi_prob(row[1].iloc[1], row[1].iloc[0])]

        return condi_probs

    
    @staticmethod
    def positions_to_units_dollar_neutral(prices_df: pd.DataFrame, positions: pd.Series,
                                          multiplier: float = 1) -> pd.DataFrame:

        units_df = pd.DataFrame(data=0.0, index=prices_df.index, columns=prices_df.columns)
        units_df.iloc[0, 0] = 0.5 * prices_df.iloc[0, 0] / prices_df.iloc[0, 0] * positions.iloc[0]
        units_df.iloc[0, 1] = -0.5 * prices_df.iloc[0, 1] / prices_df.iloc[0, 1] * positions.iloc[0]
        nums = len(positions)
        
        for i in range(1, nums):
            # By default the new amount of units to be held is the same as the previous step
            units_df.iloc[i, :] = units_df.iloc[i-1, :]
            # Updating if there are position changes
            # From not short to short
            if positions.iloc[i-1] != -1 and positions.iloc[i] == -1:  # Short 1, long 2
                long_units = 0.5 * prices_df.iloc[i, 1] / prices_df.iloc[i, 1]
                short_units = 0.5 * prices_df.iloc[i, 0] / prices_df.iloc[i, 0]
                units_df.iloc[i, 0] = - short_units
                units_df.iloc[i, 1] = long_units
            # From not long to long
            if positions.iloc[i-1] != 1 and positions.iloc[i] == 1:  # Short 2, long 1
                long_units = 0.5 * prices_df.iloc[i, 0] / prices_df.iloc[i, 0]
                short_units = 0.5 * prices_df.iloc[i, 1] / prices_df.iloc[i, 1]
                units_df.iloc[i, 0] = long_units
                units_df.iloc[i, 1] = - short_units
            # From long/short to none
            if positions.iloc[i-1] != 0 and positions.iloc[i] == 0:  # Exiting
                units_df.iloc[i, 0] = 0
                units_df.iloc[i, 1] = 0

        return units_df.multiply(multiplier)

    def get_positions_and_flags(self, returns: pd.DataFrame,
                                init_pos: int = 0, enable_reset_flag: bool = True,
                                open_rule: str = 'and', exit_rule: str = 'and') -> (pd.Series, pd.DataFrame):

        # Initialization
        open_based_on = [0, 0]  # Initially no position was opened based on stocks
        mpis = self.calc_mpi(returns)  # Mispricing indices from stock 1 and 2
        flags = pd.DataFrame(data=0.0, index=returns.index, columns=returns.columns)  # Initialize flag values
        positions = pd.Series(data=[np.nan]*len(returns), index=returns.index, dtype=float)  # 使用 float 类型
        positions.iloc[0] = init_pos
        # Reset the counters
        self._long_count, self._short_count, self._exit_count = 0, 0, 0

        # Calculate positions and flags
        for i in range(1, len(returns)):
            mpi = mpis.iloc[i, :]
            pre_flag = flags.iloc[i - 1, :]
            pre_position = positions.iloc[i - 1]

            cur_flag, cur_position, open_based_on = self._cur_flag_and_position(mpi, pre_flag, pre_position, open_based_on, enable_reset_flag, open_rule, exit_rule)
            flags.iloc[i, :] = cur_flag
            positions.iloc[i] = cur_position

        return positions, flags

    def _cur_flag_and_position(self, mpi: pd.Series, pre_flag: pd.Series, pre_position: int,
                               open_based_on: list, enable_reset_flag: bool,
                               open_rule: str, exit_rule: str) -> (pd.Series, int, list):

        centered_mpi = mpi.subtract(0.5)  # Center to 0
        # Raw value means it is not (potentially) reset by exit triggers
        raw_cur_flag = centered_mpi + pre_flag  # Definition

        cur_position, if_reset_flag, open_based_on = self._get_position_and_reset_flag(
            pre_flag, raw_cur_flag, pre_position, open_rule, exit_rule, open_based_on)

        # if if_reset_flag: reset
        # if not if_reset_flag: do nothing
        cur_flag = raw_cur_flag  # If not enable flag reset, then current flag value is just its raw value
        if enable_reset_flag:
            cur_flag = raw_cur_flag * int(not if_reset_flag)

        return cur_flag, cur_position, open_based_on

    def _get_position_and_reset_flag(self, pre_flag: pd.Series, raw_cur_flag: pd.Series,
                                     pre_position: int, open_rule: str, exit_rule: str,
                                     open_based_on: list = [0, 0],) -> (int, bool, list):

        flag_1 = raw_cur_flag.iloc[0]
        flag_2 = raw_cur_flag.iloc[1]
        lower_open_threshold = self.opening_triggers[0]
        upper_open_threshold = self.opening_triggers[1]

        # Check if positions should be open. If so, based on which stock
        # Uncomment for the next four lines to allow for openinig positions only when there's no position currently
        long_based_on_1 = (flag_1 <= lower_open_threshold)  # and (pre_position == 0)
        long_based_on_2 = (flag_2 >= upper_open_threshold)  # and (pre_position == 0)
        short_based_on_1 = (flag_1 >= upper_open_threshold)  # and (pre_position == 0)
        short_based_on_2 = (flag_2 <= lower_open_threshold)  # and (pre_position == 0)

        # Forming triggers, OR open logic
        if open_rule == 'or':
            long_trigger = (long_based_on_1 or long_based_on_2)
            short_trigger = (short_based_on_1 or short_based_on_2)
        # Forming triggers, AND open logic
        if open_rule == 'and':
            long_trigger = (long_based_on_1 and long_based_on_2)
            short_trigger = (short_based_on_1 and short_based_on_2)
        exit_trigger = self._exit_trigger_mpi(pre_flag, raw_cur_flag, open_based_on, open_rule, exit_rule)
        any_trigger = any([long_trigger, short_trigger, exit_trigger])
        # Updating trigger counts
        self._long_count += int(long_trigger)
        self._short_count += int(short_trigger)
        self._exit_count += int(exit_trigger)

        # Updating open_based_on variable.
        # This is only useful when used with OR-OR logic for open and exit. In other cases please ignore it.
        # Logic (and precedence. The sequence at which below are executed has influence on flag values.):
        # if long_based_on_1:
        #     open_based_on = [1, 1]
        # if short_based_on_1:
        #     open_based_on = [-1, 1]
        # if long_based_on_2:
        #     open_based_on = [1, 2]
        # if short_based_on_2:
        #     open_based_on = [-1, 2]
        # if exit_trigger:
        #     open_based_on = [0, 0]
        open_exit_triggers = [long_based_on_1, short_based_on_1, long_based_on_2, short_based_on_2, exit_trigger]
        open_based_on_values = [[1, 1], [-1, 1], [1, 2], [-1, 2], [0, 0]]
        for i in range(5):
            if open_exit_triggers[i]:
                open_based_on = open_based_on_values[i]

        # Update positions. Defaults to previous position unless there is a trigger to update it
        cur_position = pre_position
        # Updating logic:
        # If there is a long trigger, take long position (1);
        # If there is a short trigger, take short position (-1);
        # If there is an exit trigger, take no position (0).
        if any_trigger:
            cur_position = (int(long_trigger) - int(short_trigger)) * int(not exit_trigger)

        # When there is an exit_trigger, we reset the flag value
        if_reset_flag = exit_trigger

        return cur_position, if_reset_flag, open_based_on

    def _exit_trigger_mpi(self, pre_flag: pd.Series, raw_cur_flag: pd.Series, open_based_on: list,
                          open_rule: str, exit_rule: str) -> bool:

        pre_flag_1 = pre_flag.iloc[0]  # Previous flag1 value
        pre_flag_2 = pre_flag.iloc[1]  # Previous flag2 value
        raw_cur_flag_1 = raw_cur_flag.iloc[0]
        raw_cur_flag_2 = raw_cur_flag.iloc[1]


        slp_lower = self.stop_loss_positions[0]  # Lower end of stop loss value for flag
        slp_upper = self.stop_loss_positions[1]  # Upper end of stop loss value for flag

        # Check if crossing 0 from above & below 0
        stock_1_x_from_above = (pre_flag_1 > 0 >= raw_cur_flag_1)  # flag1 crossing 0 from above
        stock_1_x_from_below = (pre_flag_1 < 0 <= raw_cur_flag_1)  # flag1 crossing 0 from below
        stock_2_x_from_above = (pre_flag_2 > 0 >= raw_cur_flag_2)  # flag2 crossing 0 from above
        stock_2_x_from_below = (pre_flag_2 < 0 <= raw_cur_flag_2)  # flag2 crossing 0 from below

        # Check if current flag reaches stop-loss positions
        # If flag >= slp_upper or flag <= slp_lower, then it reaches the stop-loss position
        stock_1_stop_loss = (raw_cur_flag_1 <= slp_lower or raw_cur_flag_1 >= slp_upper)
        stock_2_stop_loss = (raw_cur_flag_2 <= slp_lower or raw_cur_flag_2 >= slp_upper)

        # Determine whether one should exit the current open position.
        exit_trigger = None
        # Case: open OR, exit OR (method in the paper [Xie et al. 2014])
        # If trades were open based on flag1, then they are closed if flag1 returns to 0, or reaches stop loss
        # position. Same for flag2. Thus in total there are 4 possibilities:
        # 1. If current pos is long based on 1: flag 1 returns to 0 from below, or reaches stop loss
        # 2. If current pos is short based on 1: flag 1 returns to 0 from above, or reaches stop loss
        # 3. If current pos is long based on 2: flag 2 returns to 0 from below, or reaches stop loss
        # 4. If current pos is short based on 2: flag 2 returns to 0 from above, or reaches stop loss
        # Hence, as long as 1 of the 4 exit condition is satisfied, we exit
        if open_rule == 'or' and exit_rule == 'or':
            exit_based_on_1 = any([open_based_on == [1, 1] and (stock_1_x_from_below or stock_1_stop_loss),
                                   open_based_on == [-1, 1] and (stock_1_x_from_above or stock_1_stop_loss)])
            exit_based_on_2 = any([open_based_on == [1, 2] and (stock_2_x_from_above or stock_2_stop_loss),
                                   open_based_on == [-1, 2] and (stock_2_x_from_below or stock_2_stop_loss)])
            exit_trigger = (exit_based_on_1 or exit_based_on_2)
            return exit_trigger

        exit_for_1 = any([stock_1_x_from_below or stock_1_stop_loss, stock_1_x_from_above or stock_1_stop_loss])
        exit_for_2 = any([stock_2_x_from_above or stock_2_stop_loss, stock_2_x_from_below or stock_2_stop_loss])
        # Case: open AND, exit OR (method in the paper [Rad et al. 2016])
        # In this case, it makes no sense to have the open_based_on variable. So we are just directly looking at the
        # thresholds. If the flag1 OR flag2 series reaches the thresholdsm, then exit
        if open_rule == 'and' and exit_rule == 'or':
            exit_trigger = exit_for_1 or exit_for_2

        # Case: open AND or OR, exit OR (method in the paper [Rad et al. 2016])
        # In this case, it makes no sense to have the open_based_on variable. So we are just directly looking at the
        # thresholds. If the flag1 AND flag2 series reaches the thresholdsm, then exit
        if exit_rule == 'and':
            exit_trigger = exit_for_1 and exit_for_2

        return exit_trigger
