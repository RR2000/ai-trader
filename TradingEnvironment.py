from abc import ABC

import gym
import numpy as np
import matplotlib.pyplot as plt


class TradingEnvironment(gym.Env, ABC):
    def __init__(self, data, window_size):
        super(TradingEnvironment, self).__init__()

        self.window_size = window_size
        self.current_step = window_size
        self.max_step = 0

        min_val = np.min(data)
        max_val = np.max(data)
        self.data = data = (data - min_val) / (max_val - min_val)

        self.data['Action'] = np.array(len(data))
        self.data['USD'] = np.array(len(data))
        self.data['Crypto'] = np.array(len(data))
        self.data['Steps'] = window_size

        self.initialTotal = 1
        self.USDBalance = self.initialTotal / 2
        self.CryptoBalance = (self.initialTotal / 2) / self.data.iloc[self.current_step]['Close']

        # Keep only the 'Steps' feature
        self.selected_features = ['Close', 'USD', 'Crypto']
        num_features = len(self.selected_features)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, num_features))
        self.action_space = gym.spaces.Discrete(3)  # buy, sell, hold
        self.data = self.data.reset_index(drop=True)

    # The rest of the class remains unchanged

    def reset(self, **kwargs):
        self.current_step = self.window_size
        self.USDBalance = self.initialTotal / 2
        self.CryptoBalance = (self.initialTotal / 2) / self.data.iloc[self.current_step]['Close']
        self.data['Action'] = np.array(len(self.data))
        self.data['USD'] = np.array(len(self.data))
        self.data['Crypto'] = np.array(len(self.data))
        self.data['Steps'] = self.window_size
        return self._get_observation()

    def step(self, action):
        info = {}
        self.data.at[self.current_step, 'Action'] = action
        self.data.at[self.current_step, 'Steps'] = self.current_step
        self.current_step += 1
        reward = 0

        if self.current_step + 2 > len(self.data):
            done = True
            reward = 0
        else:
            done = False
            current_price = self.data.iloc[self.current_step]['Close']
            next_price = self.data.iloc[self.current_step + 1]['Close']

            # Calculate the current total balance
            current_total_balance = self.USDBalance + self.CryptoBalance * current_price

            if action == 0:  # buy
                if self.USDBalance > 0:
                    buy_amount = self.USDBalance
                    self.CryptoBalance += buy_amount / current_price
                    self.USDBalance = 0
                    self.data.at[self.current_step, 'Crypto'] = self.CryptoBalance
                    self.data.at[self.current_step, 'USD'] = self.USDBalance

            elif action == 1:  # sell
                if self.CryptoBalance > 0:
                    sell_amount = self.CryptoBalance
                    self.USDBalance += sell_amount * current_price
                    self.CryptoBalance = 0
                    self.data.at[self.current_step, 'Crypto'] = self.CryptoBalance
                    self.data.at[self.current_step, 'USD'] = self.USDBalance

            new_total_balance = self.USDBalance + self.CryptoBalance * next_price

            reward += new_total_balance - current_total_balance

            self.data.at[self.current_step, 'Balance'] = new_total_balance

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        start_index = self.current_step - self.window_size
        end_index = self.current_step - 1
        obs = self.data.loc[start_index: end_index, self.selected_features].values

        if len(obs) < self.window_size:
            obs = np.pad(obs, ((self.window_size - len(obs), 0), (0, 0)), 'constant', constant_values=(0,))

        return obs

    def render(self):
        start = self.window_size + 1
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        ax1.plot(self.data['Close'][start:self.current_step], label='Close')
        buy_signals = self.data.iloc[start:self.current_step]['Action'] == 0
        sell_signals = self.data.iloc[start:self.current_step]['Action'] == 1

        ax1.scatter(self.data.iloc[start:self.current_step][buy_signals].index,
                    self.data.iloc[start:self.current_step][buy_signals]['Close'],
                    label='Buy', marker='^', c='b')
        ax1.scatter(self.data.iloc[start:self.current_step][sell_signals].index,
                    self.data.iloc[start:self.current_step][sell_signals]['Close'],
                    label='Sell', marker='v', c='r')

        ax1.set_title('Price History with Buy and Sell signals')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')

        ax2.plot(self.data['Balance'][start:self.current_step], label='Total Balance')
        ax2.set_title('Total Balance History')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Total Balance (USD)')

        plt.show()

