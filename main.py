import pandas as pd
import numpy as np
from binance import Client
import matplotlib.pyplot as plt

client = Client()


# Get hourly prices over last 400 hours
def get_data(symbol, interval='1h', lookback='400'):
    frame = pd.DataFrame(client.get_historical_klines(symbol, interval, lookback + ' hours UTC'))
    frame = frame.iloc[:, 0:6]
    frame.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    frame.set_index('Time', inplace=True)
    frame.index = pd.to_datetime(frame.index, unit='ms')
    frame = frame.astype(float)
    return frame


def populate_dataframe(df):
    df['rollhigh'] = df.High.rolling(15).max()  # rolling max over last 15 hours
    df['rolllow'] = df.Low.rolling(15).min()  # rolling low over last 15 hours
    df['mid'] = (df.rollhigh + df.rolllow) / 2
    df['high_approach'] = np.where(df.Close > df.rollhigh * 0.996, 1, 0)  # -0.4%
    df['close_above_mid'] = np.where(df.Close > df.mid, 1, 0)
    df['mid_cross'] = df.close_above_mid.diff() == 1


def entries_exits(df):
    in_position = False
    buy_dates, sell_dates = [], []

    for i in range(len(df)):
        if not in_position:  # Buy condition
            if df.iloc[i].mid_cross:
                buy_dates.append(df.iloc[i + 1].name)
                in_position = True
        if in_position:
            if df.iloc[i].high_approach:
                sell_dates.append(df.iloc[i + 1].name)
                in_position = False
    return buy_dates, sell_dates


def visualize(df, buy_dates, sell_dates):
    plt.plot(df[['Close', 'rollhigh', 'rolllow', 'mid']])
    plt.scatter(buy_dates, df.loc[buy_dates].Open, marker='^', color='g', s=200)
    plt.scatter(sell_dates, df.loc[sell_dates].Open, marker='v', color='r', s=200)
    plt.style.use("dark_background")
    plt.show()


if __name__ == '__main__':
    dataframe = get_data("BTCUSDT")
    populate_dataframe(dataframe)
    visualize(dataframe, entries_exits(dataframe)[0], entries_exits(dataframe)[1])
