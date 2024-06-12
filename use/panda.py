import pandas as pd


dir = "result/panda_means3.csv"

df = pd.read_csv(dir)
df = df.round(2)
df.to_csv(dir, index=False)
# df.to_csv(dir, index=False)
