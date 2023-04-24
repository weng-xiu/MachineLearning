import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn import datasets
from sklearn.datasets import fetch_california_housing

# data = fetch_california_housing()
data = datasets.load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.DataFrame(data.target, columns=['MEDV'])
X = df
y = target

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
print(X_train.shape)
print(X_test.shape)

# 模型训练
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.coef_)
print(lr.intercept_)

# 模型评估
y_pred = lr.predict(X_test)
from sklearn import metrics

MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('MSE:', MSE)
print('RMSE:', RMSE)

# -----------图像绘制--------------
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 绘制图
plt.figure(figsize=(15, 5))

# plt.plot(range(len(y_test)), y_test, 'r', label='测试数据')

plt.plot(range(len(y_test)), y_pred, 'b', label='预测数据')
plt.legend()
plt.show()

# # 绘制散点图
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.show()