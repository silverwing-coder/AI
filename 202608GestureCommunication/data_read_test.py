import pandas as pd

df = pd.read_csv('gesture_data.csv')
# print(df)
y = df['label'].to_numpy() - 1

print(y)