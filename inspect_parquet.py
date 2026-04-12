import pandas as pd

file_path = r"C:\Users\ashok\OneDrive\Desktop\gait\gait_dataset.parquet"

df = pd.read_parquet(file_path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())