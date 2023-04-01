import pandas as pd
from tensorflow.python.keras.models import load_model

from TradingEnvironment import TradingEnvironment
from DQN import train_dqn

if __name__ == '__main__':
    # Load data
    data = pd.read_csv('src/data/historical_data.csv')  # Replace with your data file
    data = data[['Close']]
    data = data.tail(120)
    # Create environment and train model
    window_size = 15
    env = TradingEnvironment(data, window_size)
    episodes = 300
    batch_size = 32
    print(batch_size)
    trained_model = train_dqn(env, episodes, batch_size, 99)

    # Save model weights
    trained_model.save('src/model/completed/trading_model.h5')

    model = load_model('src/model/completed/trading_model.h5')

    print(model.summary())
