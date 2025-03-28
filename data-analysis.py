import yfinance as yf
import pandas as pd

# Fetch historical data
data = yf.download('AAPL', start='2023-01-01', end='2024-03-27')

priceData = data['Close'].values

fileName = "GBPUSD_M1.csv"
data2 = pd.read_csv(fileName)

print(priceData.shape)

