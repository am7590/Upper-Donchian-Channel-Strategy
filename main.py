import pandas as pd
import numpy as np
from binance import Client
import matplotlib.pyplot as plt
client = Client()

# Get hourly prices over last 400 hours
def get_data(symbol, interval='1h', lookback='400'):
    frame = pd.DataFrame(client.get_historical_klines(symbol, interval, lookback+' hours UTC'))
    frame = frame.iloc[:,0:6]
    frame.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    frame.set_index('Time', inplace=True)
    frame.index = pd.to_datetime(frame.index, unit='ms')
    frame = frame.astype(float)
    return frame


if __name__ == '__main__':
    print(get_data("BTCUSDT"))
