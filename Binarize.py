import pandas as pd
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold

# 讀取資料集
data = pd.read_csv('laptop_prices.csv')

# OneHotEncoder() - 對 TypeName 和 OS 進行 one-hot encoding

encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(data[['TypeName', 'OS']])

# 將編碼結果轉為 DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['TypeName', 'OS']))

print("OneHotEncoder() 結果:")
print(encoded_df.head())
print("\n")

