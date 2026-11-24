import yfinance as yf
import pandas as pd

# Date beforer any instruments data in yahoo
START_DATE = "2000-01-01"
# Symbols aligned with yahooo finance documentation
SYMBOLS = ["GLD", "SLV", "USO", "CPER"]

# Downloading one, specific symbol data
def get_data(symbol, start_date):
    # Multitask     - 1 level columns (Close instead of ("Close", "GLD"))
    # auto_adjust   - data normalization by splits
    df = yf.download(symbol, start=start_date, multi_level_index=False, auto_adjust=True)

    # Check if empty before saving - no overwriting by empty file
    if df.empty:
        raise ValueError(f'No data for {symbol} was loaded')

    # Saving with dates index and column types, advantage over csv
    df.to_parquet(f'data/{symbol}_data.parquet')
    return df

# Loads specific data, used by other functions
def load_data(symbol):
    return pd.read_parquet(f'data/{symbol}_data.parquet')

# Function executed automaticaly by GitHub agent everyday after session is closed 
def download_all(symbols, start_date):
    
    for symbol in symbols:
        try:
            df = get_data(symbol, start_date)
        # Safety feature so that one error does not crush all process
        except Exception as error:
            print(f'Error: {symbol}, {error}')

if __name__ == "__main__":
    # Downloads all only when this file is being executed as main
    download_all(SYMBOLS, START_DATE)