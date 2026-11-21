import yfinance as yf
import pandas as pd

START_DATE = "2000-01-01"
SYMBOL = "GLD"

def get_data(symbol, start_date):
    df = yf.download(symbol, start=start_date, multi_level_index=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f'No data for {symbol} was loaded')

    df.to_parquet(f'data/{symbol}_data.parquet')
    return df

def load_data(symbol):
    return pd.read_parquet(f'data/{symbol}_data.parquet')

def get_details(df):
    print(df.head())
    print(df.columns)
    print(len(df))

if __name__ == "__main__":
    df = load_data(SYMBOL)
    

