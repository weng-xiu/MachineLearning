import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------------1. 数据---------
# points = np.genfromtxt('data.csv', delimiter=',')
data = np.array([[32, 31], [53, 68], [61, 62], [47, 71], [59, 87], [55, 78], [52, 79], [39, 59], [48, 75], [52, 71],
                 [45, 55], [54, 82], [44, 62], [58, 75], [56, 81], [48, 60], [44, 82], [60, 97], [45, 48], [38, 56],
                 [66, 83], [65, 118], [47, 57], [41, 51], [51, 75], [59, 74], [57, 95], [63, 95], [46, 79],
                 [50, 83]])
# 提取points中的两列数据，分别作为x，y
x = data[:, 0]
y = data[:, 1]


# --------------2. 定义损失函数--------------
# 损失函数是系数的函数，另外还要传入数据的x，y
def compute_cost(w, b, data):
    total_cost = 0
    M = len(data)
    # 逐点计算平方损失误差，然后求平均数
    for i in range(M):
        x = data[i, 0]
        y = data[i, 1]
        total_cost += (y - w * x - b) ** 2
    return total_cost / M


lr = LinearRegression()
x_new = x.reshape(-1, 1)
y_new = y.reshape(-1, 1)
lr.fit(x_new, y_new)
# 从训练好的模型中提取系数和偏置
w = lr.coef_[0][0]
b = lr.intercept_[0]
print("w is: ", w)
print("b is: ", b)
cost = compute_cost(w, b, data)
print("cost is: ", cost)
plt.scatter(x, y)
# 针对每一个x，计算出预测的y值
pred_y = w * x + b
plt.plot(x, pred_y, c='r')
plt.show()