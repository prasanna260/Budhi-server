import yfinance as yf
import pandas as pd
import time
import json

# --------------------------
# 1. Load Nifty 500 symbols
# --------------------------

file_path = "nifty500.txt"   # Your uploaded file path
symbols = []

with open(file_path, "r") as f:
    for line in f:
        symbol = line.strip()
        if symbol:
            symbols.append(symbol)

print(f"Loaded {len(symbols)} symbols from Nifty 500 list")
# 2. Convert to yfinance tickers
yf_symbols = [f"{sym}.NS" for sym in symbols]

# --------------------------
# 3. Fetch info for each ticker and flatten JSON
# --------------------------

records = []

for original, yf_symbol in zip(symbols, yf_symbols):
    print(f"Fetching: {yf_symbol}")
    try:
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info

        if not info or info is None:
            # Create base record with error status
            record = {
                "symbol": original,
                "yf_symbol": yf_symbol,
                "status": "No data returned"
            }
        else:
            # Start with basic fields
            record = {
                "symbol": original,
                "yf_symbol": yf_symbol,
                "status": "OK"
            }
            
            # Flatten all info fields into the record
            for key, value in info.items():
                # Convert complex objects to strings to avoid CSV issues
                if isinstance(value, (list, dict)):
                    record[key] = json.dumps(value)
                else:
                    record[key] = value
        
        records.append(record)
        time.sleep(0.2)  # To avoid rate limiting

    except Exception as e:
        # Create error record
        record = {
            "symbol": original,
            "yf_symbol": yf_symbol,
            "status": f"Error: {str(e)}"
        }
        records.append(record)

# --------------------------
# 4. Write to CSV
# --------------------------

df = pd.DataFrame(records)
output_file = "nifty500_info_expanded.csv"
df.to_csv(output_file, index=False)

print(f"Saved expanded CSV file: {output_file}")
print(f"Total columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")