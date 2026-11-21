import yfinance as yf
import pandas as pd

START_DATE = "2000-01-01"
SYMBOLS = ["GLD", "SLV", "USO", "CPER"]

def get_data(symbol, start_date):
    df = yf.download(symbol, start=start_date, multi_level_index=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f'No data for {symbol} was loaded')

    df.to_parquet(f'data/{symbol}_data.parquet')
    return df

def load_data(symbol):
    return pd.read_parquet(f'data/{symbol}_data.parquet')

def download_all(symbols, start_date):
    
    for symbol in symbols:
        try:
            df = get_data(symbol, start_date)
            print(df.head())
            print(len(df))
        except Exception as error:
            print(f'Error: {symbol}, {error}')

if __name__ == "__main__":

    download_all(SYMBOLS, START_DATE)
    

