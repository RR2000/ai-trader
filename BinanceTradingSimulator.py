import time

import numpy as np
import pandas as pd
from binance import Client
from sklearn.preprocessing import MinMaxScaler
from ta import trend, momentum
from tensorflow.python.keras.models import load_model

api_key = 'REMOVED'
api_secret = 'REMOVED'
symbol = 'BTCBUSD'
interval = Client.KLINE_INTERVAL_1MINUTE
duration = "365 day"
client = Client(api_key, api_secret)


def add_technical_indicators(df):
    df['SMA_10'] = trend.sma_indicator(df['Close'], window=10)
    df['SMA_30'] = trend.sma_indicator(df['Close'], window=30)
    df['RSI'] = momentum.RSIIndicator(df['Close']).rsi()
    macd_indicator = trend.MACD(df['Close'])
    df['MACD'] = macd_indicator.macd()
    df['MACD_signal'] = macd_indicator.macd_signal()
    df.dropna(inplace=True)
    return df.reset_index(drop=True)


def get_data():
    print("Downloading last week...")
    klines = client.get_historical_klines(symbol, interval, "1 week ago UTC")  # todo change with 30
    klines = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                           'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                           'Taker buy quote asset volume', 'Ignore'])
    klines = klines[['Close', 'Volume']].astype(float)
    return klines.reset_index(drop=True)


def get_last_candle():
    klines = client.get_klines(symbol=symbol, interval=interval, limit=1)
    tmp = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                        'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                        'Taker buy quote asset volume', 'Ignore'])
    tmp = tmp[['Close', 'Volume']].astype(float)
    return tmp.reset_index(drop=True)


def make_decision(state, model, window_size):
    state_array = state.to_numpy()
    print(state_array.shape)
    state_array = np.reshape(state_array, (1, window_size, state_array.shape[1]))
    act_values = model.predict(state_array)
    return np.argmax(act_values[0]), np.round(act_values, 4)


def main():
    close_volume_data = get_data()
    scaler = MinMaxScaler(feature_range=(0.01, 1.01))
    model = load_model('src/model/completed/trading_model_episode_30.h5')
    # data_normalised = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    cash_balance = 1000
    crypto_balance = 0.0
    trade_fee = 0.001
    while True:
        new_candle = get_last_candle()
        close_volume_data = pd.concat([close_volume_data[1:], new_candle], ignore_index=True)
        technical_data = add_technical_indicators(close_volume_data)

        data_normalised = pd.DataFrame(scaler.fit_transform(technical_data), columns=technical_data.columns)

        this_price = technical_data.iloc[-1]['Close']
        next_price = this_price
        this_price_normalised = data_normalised.iloc[-1]['Close']

        print("not_scaled " + str(this_price))
        print("    scaled " + str(this_price_normalised))

        window_size = 30
        last_index = len(technical_data) - 1
        state = data_normalised.iloc[last_index - window_size:last_index]
        try:
            decision, values = make_decision(state, model, window_size)

            if decision == 0 and cash_balance > 0:  # Buy
                crypto_to_buy = cash_balance / next_price
                fee = crypto_to_buy * trade_fee
                crypto_to_buy -= fee
                cash_balance = 0
                crypto_balance += crypto_to_buy
                print(f"{values} Buy: {crypto_to_buy} crypto at {next_price},   balance: {crypto_balance * next_price}")

            elif decision == 1 and crypto_balance > 0:  # Sell
                cash_to_receive = crypto_balance * next_price
                fee = cash_to_receive * trade_fee
                cash_to_receive -= fee
                crypto_balance = 0
                cash_balance += cash_to_receive
                print(f"{values} Sell: {crypto_balance} crypto at {next_price}, balance: {cash_balance}")

            else:
                print(f"{values} Hold: {crypto_balance} crypto and            , balance: {cash_balance}")
            total_value = cash_balance + (crypto_balance * technical_data.iloc[-1]['Close'])
            print(f"Total value: {total_value}")
            time.sleep(60)
        except ValueError as e:
            print(e)
            print("Resetting...")
            close_volume_data = get_data()
            scaler = MinMaxScaler(feature_range=(0.01, 1.01))
            technical_data = add_technical_indicators(close_volume_data)
            data_normalised = pd.DataFrame(scaler.fit_transform(technical_data), columns=technical_data.columns)


if __name__ == "__main__":
    main()
