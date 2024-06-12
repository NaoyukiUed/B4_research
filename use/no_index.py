import pandas as pd

input_dir = "result/short/lstm.csv"

df = pd.read_csv(input_dir)
df = df.drop(df.columns[0], axis=1)
df.to_csv(input_dir, index=False)
