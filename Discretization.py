import pandas as pd
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold

# 讀取資料集
data = pd.read_csv('laptop_prices.csv')

# KBinsDiscretizer() - 將 Price_euros 分為 3 個區間

k_bins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
data['Price_bins'] = k_bins.fit_transform(data[['Price_euros']])

print("KBinsDiscretizer() 結果:")
print(data[['Price_euros', 'Price_bins']].head())
print("\n")
