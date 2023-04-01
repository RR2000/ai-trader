import numpy as np
import pandas as pd
from ta import trend, momentum
import tensorflow as tf
from tensorflow.python.keras.models import load_model


def add_technical_indicators(data):
    data['SMA_10'] = trend.sma_indicator(data['Close'], window=10)
    data['SMA_30'] = trend.sma_indicator(data['Close'], window=30)
    data['RSI'] = momentum.RSIIndicator(data['Close']).rsi()
    macd_indicator = trend.MACD(data['Close'])
    data['MACD'] = macd_indicator.macd()
    data['MACD_signal'] = macd_indicator.macd_signal()
    data.dropna(inplace=True)
    return data


def make_decision(state, model):
    state = np.reshape(state, (1, -1))
    act_values = model.predict(state)
    return np.argmax(act_values[0])


def trading_simulation(model, complete, initial_balance=1000.0, trade_fee=0.001, window_size=15):
    cash_balance = initial_balance
    crypto_balance = 0.0
    for i in range(window_size, len(complete) - 1):
        current_price = complete.iloc[i]['Close']
        next_price = complete.iloc[i + 1]['Close']
        #complete.at[i, 'Total'] = cash_balance + crypto_balance * current_price
        complete.at[i, 'USD'] = cash_balance
        complete.at[i, 'Crypto'] = crypto_balance
        state = complete.iloc[i - window_size:i].values
        decision = make_decision(state, model)

        if decision == 0:  # Buy
            crypto_to_buy = cash_balance / next_price
            fee = crypto_to_buy * trade_fee
            crypto_to_buy -= fee
            cash_balance = 0
            crypto_balance += crypto_to_buy
            print(f"Buy: {crypto_to_buy} crypto at {next_price}")

        elif decision == 1:  # Sell
            cash_to_receive = crypto_balance * next_price
            fee = cash_to_receive * trade_fee
            cash_to_receive -= fee
            crypto_balance = 0
            cash_balance += cash_to_receive
            print(f"Sell: {crypto_balance} crypto at {next_price}")

        else:  # Hold
            print("Hold")

    final_balance = cash_balance + (crypto_balance * complete.iloc[-1]['Close'])
    return initial_balance, final_balance


def main():
    # Load model configuration and weights
    '''
    with open('dqn_trading_config.json', 'r') as f:
        model_config = f.read()
    model = tf.keras.models.model_from_json(model_config)
    model.load_weights('dqn_trading_weights.h5')
    '''
    model = load_model('src/model/completed/trading_model.h5')
    plain = pd.read_csv("src/data/historical_data.csv")
    prices = plain[['Close']].tail(10000)
    complete = prices #add_technical_indicators(prices).reset_index(drop=True)

    initial_balance, final_balance = trading_simulation(model, complete)

    print(f"Initial balance: {initial_balance}")
    print(f"Final balance: {final_balance}")


if __name__ == "__main__":
    main()
