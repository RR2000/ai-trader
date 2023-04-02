from random import random, randint

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ta import trend, momentum
from tensorflow.python.keras.models import load_model

verbose = 1


def get_data_and_technical_indicators(source):
    tmp = pd.read_csv(source)
    tmp = tmp[['Close', 'Volume']]
    tmp['SMA_10'] = trend.sma_indicator(tmp['Close'], window=10)
    tmp['SMA_30'] = trend.sma_indicator(tmp['Close'], window=30)
    tmp['RSI'] = momentum.RSIIndicator(tmp['Close']).rsi()
    macd_indicator = trend.MACD(tmp['Close'])
    tmp['MACD'] = macd_indicator.macd()
    tmp['MACD_signal'] = macd_indicator.macd_signal()
    tmp.dropna(inplace=True)
    return tmp


def make_decision(state, model, window_size):
    state_array = state.to_numpy()
    state_array = np.reshape(state_array, (1, window_size, state_array.shape[1]))
    act_values = model.predict(state_array)
    return np.argmax(act_values[0]), np.round(act_values, 4)


def trading_simulation(model, data, scaled_data, initial_balance=1000.0, trade_fee=0.001):
    window_size = 30
    cash_balance = initial_balance
    crypto_balance = 0.0
    for i in range(window_size, len(data) - 1):
        next_price = data.iloc[i + 1]['Close']
        state = scaled_data.iloc[i - window_size:i]
        decision, values = make_decision(state, model, window_size)

        if decision == 0 and cash_balance > 0:  # Buy
            crypto_to_buy = cash_balance / next_price
            fee = crypto_to_buy * trade_fee
            crypto_to_buy -= fee
            cash_balance = 0
            crypto_balance += crypto_to_buy
            if verbose >= 2:
                print(f"{values} Buy: {crypto_to_buy} crypto at {next_price},   balance: {crypto_balance * next_price}")

        elif decision == 1 and crypto_balance > 0:  # Sell
            cash_to_receive = crypto_balance * next_price
            fee = cash_to_receive * trade_fee
            cash_to_receive -= fee
            crypto_balance = 0
            cash_balance += cash_to_receive
            if verbose >= 2:
                print(f"{values} Sell: {crypto_balance} crypto at {next_price}, balance: {cash_balance}")

        elif decision == 2:  # Hold
            if verbose >= 3:
                part = cash_balance + (crypto_balance * next_price)
                print(f"{values} Hodl:                                          balance: {part}")
            pass
        else:
            if verbose >= 3:
                part = cash_balance + (crypto_balance * next_price)
                print(f"{values} well...:                                       balance: {part}")
            pass

    final_balance = cash_balance + (crypto_balance * data.iloc[-1]['Close'])
    return initial_balance, final_balance


def main():
    # model = load_model('src/model/completed/trading_model.h5')
    model = load_model('src/model/checkpoint/trading_model_episode_30.h5')
    data = get_data_and_technical_indicators("src/data/historical_data.csv")
    # 10_080 one week in minutes
    one_week = 10_080
    n_of_weeks = 0.5
    weeks_in_minutes = int(one_week * n_of_weeks)

    total_percentage_difference = 0
    my_bot_total_gain_loss = 0
    benchmark_total_gain_loss = 0

    total_test_balance = 10000
    epochs = 10
    e_balance = total_test_balance / epochs
    for _ in range(epochs):
        rnd = randint(100, len(data) - weeks_in_minutes - 1)
        split_data = data[rnd: weeks_in_minutes + rnd]
        split_data = split_data.tail(weeks_in_minutes).reset_index(drop=True)

        scaler = MinMaxScaler(feature_range=(0.01, 1.01))
        scaled_data = pd.DataFrame(scaler.fit_transform(split_data), columns=split_data.columns)

        initial_balance, final_balance = trading_simulation(model, split_data, scaled_data, initial_balance=e_balance)
        initial_price, final_price = split_data.iloc[31]['Close'], split_data.iloc[-1]['Close']

        my_bot_gain = round(((final_balance / initial_balance) - 1) * 100, 2)
        benchmark = round(((final_price / initial_price) - 1) * 100, 2)
        difference = my_bot_gain - benchmark
        total_percentage_difference += difference

        my_bot_total_gain_loss += final_balance
        benchmark_total_gain_loss += (final_price / initial_price) * initial_balance

        if verbose >= 2:
            print(f"Initial balance: {initial_balance}$")
            print(f"Final balance: {final_balance}$")
            print(f"MyBot    {my_bot_gain}%")
            print(f"Benchmark {benchmark}%")
        if verbose >= 1:
            print(f"Difference from benchmark {difference:+.2} pt%")

    print()
    print()
    print("TEST FINISHED")
    print(f"{epochs} tests")
    print(f"Total difference from benchmark: {round(total_percentage_difference, 2)}%")
    print(f"My bot     balance $: {round(my_bot_total_gain_loss, 2)}$")
    print(f"My bot     balance %: {round(my_bot_total_gain_loss/total_test_balance, 2)}%")
    print(f"Benchmark balance  $: {round(benchmark_total_gain_loss, 2)}$")
    print(f"Benchmark balance  %: {round(benchmark_total_gain_loss/total_test_balance, 2)}%")


if __name__ == "__main__":
    main()
