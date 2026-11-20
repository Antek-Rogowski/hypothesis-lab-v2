import yfinance as yf

df = yf.download("GOLD", start="2000-01-01", auto_adjust=True, multi_level_index=False)

print(df.head())
print(df.columns)
print(len(df))