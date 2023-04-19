#使用matplotlib绘制图像，使用numpy准备数据集
import  matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

#准备自变量x，生成数据集，-3到3的区间均分间隔30份数
x = np.linspace(3,6,40)

#准备因变量y，这一个关于x的假设函数
y=3 * x + 2

x = x + np.random.rand(40)




#准备因变量y，这一个关于x的假设函数


#由于fit 需要传入二维矩阵数据，因此需要处理x，y数据格式,将每个样本信息单独作为矩阵的一行
x=[[i] for i in x]
y=[[i] for i in y]

model=linear_model.LinearRegression()
model.fit(x,y)
#准备测试数据 x_，这里准备了三组，如下：
x_=[[4],[5],[6]]
# 打印预测结果
y_=model.predict(x_)
print(y_)

#查看w和b的
print("w值为:",model.coef_)
print("b截距值为:",model.intercept_)

#数据集绘制,散点图，图像满足函假设函数图像
plt.scatter(x,y)
#绘制最佳拟合直线
plt.plot(x_,y_,color="red",linewidth=3.0,linestyle="-")
plt.legend(["func","Data"],loc=0)
plt.show()