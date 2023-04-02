import pandas as pd
from ta import trend, momentum
from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from TradingEnvironment import TradingEnvironment
from DQN import train_dqn


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


if __name__ == '__main__':
    data = get_data_and_technical_indicators('src/data/historical_data.csv')
    data = data.tail(30 + 7 * 11).reset_index(drop=True)

    scaler = MinMaxScaler(feature_range=(0.01, 1.01))
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Create environment and train model
    env = TradingEnvironment(data, 30)
    episodes = 200
    batch_size = 64
    trained_model = train_dqn(env, episodes, batch_size, 30)

    # Save model weights
    trained_model.save('src/model/completed/trading_model.h5')

    model = load_model('src/model/completed/trading_model.h5')

    print(model.summary())
