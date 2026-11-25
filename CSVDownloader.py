import os
import pandas as pd
from binance.client import Client
from tqdm import tqdm


def download_historical_data(api_key, api_secret, symbol, interval, duration, file_name='historical_data.csv'):
    client = Client(api_key, api_secret)

    # Get the number of days to display in the progress bar
    num_days = int(duration.split()[0])

    klines = []
    for day in tqdm(range(num_days), desc="Downloading data"):
        current_duration = f"{day} day ago UTC"
        next_duration = f"{day + 1} day ago UTC"
        daily_klines = client.get_historical_klines(symbol, interval, next_duration, current_duration)
        klines.extend(daily_klines)

    data = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                         'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                         'Taker buy quote asset volume', 'Ignore'])

    data.to_csv(file_name, index=False)
    print(f"Historical data saved to {file_name}")


# Usage example
# Market data is public: no API key is needed. If you use one, take it from the environment, never commit it.
api_key = os.environ.get('BINANCE_API_KEY')
api_secret = os.environ.get('BINANCE_API_SECRET')
symbol = 'BTCBUSD'
interval = Client.KLINE_INTERVAL_1MINUTE
duration = "365 day"

download_historical_data(api_key, api_secret, symbol, interval, duration, "src/data/historical_data.csv")
