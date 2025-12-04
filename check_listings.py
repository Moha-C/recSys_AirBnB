import pandas as pd
from pathlib import Path

path = Path("data/processed/listings_v1.parquet")
df = pd.read_parquet(path)

print("Columns:")
print(df.columns.tolist())

print("\nSample rows:")
print(df.head(5)[["id", "name"]])

if "price" in df.columns:
    print("\nPrice column stats:")
    print(df["price"].describe())
    print("\nUnique price examples:")
    print(df["price"].dropna().astype(str).head(20))
else:
    print("\nNo 'price' column found!!")

import pandas as pd
from pathlib import Path

raw = pd.read_csv("data/raw/listings.csv.gz", low_memory=False)

print("\nRAW columns:")
print(raw.columns.tolist())

print("\nCheck raw price head:")
for col in raw.columns:
    if "price" in col.lower():
        print(f"\n---- {col} ----")
        print(raw[col].head(10))
