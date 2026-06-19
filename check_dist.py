import pandas as pd
import numpy as np

df = pd.read_csv('dataset/XAUUSDm_H1_features.csv')
df = df[df['regime'] == 'MEAN_REVERTING'].copy()
N_BARS = 5
THRESHOLD = 0.0020
df['future_return'] = (df['close'].shift(-N_BARS) - df['close']) / df['close']
conditions = [
    (df['future_return'] > THRESHOLD),
    (df['future_return'] < -THRESHOLD)
]
df['target'] = np.select(conditions, [1, -1], default=0)
print(df['target'].value_counts(normalize=True))
print(df['target'].value_counts())
