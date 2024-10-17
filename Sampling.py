import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# 讀取資料集
data = pd.read_csv('laptop_prices.csv')


# 1. train_test_split() - 將資料集拆分為訓練集和測試集

# 選擇 'Price_euros' 作為目標變量 (y)，其他變量作為特徵 (X)
X = data[['Ram', 'CPU_freq', 'Inches', 'Weight']]  # 使用部分數值特徵作為範例
y = data['Price_euros']

# 將資料集拆分為 80% 訓練集和 20% 測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("train_test_split() 結果:")
print("X_train 前 5 筆資料:\n", X_train.head())
print("\nX_test 前 5 筆資料:\n", X_test.head())
print("\ny_train 前 5 筆資料:\n", y_train.head())
print("\ny_test 前 5 筆資料:\n", y_test.head())
print("\n")


# 2. sample_without_replacement() - 從資料集中無重複抽樣

# 從資料中隨機抽取 5 筆無重複樣本
sample = resample(data, replace=False, n_samples=5, random_state=42)

print("sample_without_replacement() 結果:")
print(sample)
