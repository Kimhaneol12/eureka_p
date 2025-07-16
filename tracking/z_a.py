import pandas as pd

df = pd.read_csv("runs/detect/exp22/tracking_output.csv")
print(df.columns.tolist())
