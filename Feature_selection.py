import pandas as pd
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold

# 讀取資料集
data = pd.read_csv('laptop_prices.csv')


# VarianceThreshold() - 移除低變異的特徵

# 選擇 'Ram' 和 'CPU_freq' 作為範例特徵
selector = VarianceThreshold(threshold=0.1)
selected_features = selector.fit_transform(data[['Ram', 'CPU_freq']])

# 將結果轉為 DataFrame
selected_df = pd.DataFrame(selected_features, columns=['Ram', 'CPU_freq'])

print("VarianceThreshold() 結果:")
print(selected_df.head())
