# coding:utf-8

from sklearn.linear_model import LinearRegression

# 1.获取数据
x = [[80, 86],
     [82, 80],
     [85, 78],
     [90, 90],
     [86, 82],
     [82, 90],
     [78, 80],
     [92, 94]]
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

# 2.模型训练
# 2.1 实例化一个估计器
estimator = LinearRegression()

# 2.2 使用fit方法进行训练
estimator.fit(x, y)


# 打印对应的系数:
print("线性回归的系数是:\n", estimator.coef_)

# 打印的预测结果是:
print("输出预测结果:\n", estimator.predict([[100, 80]]))