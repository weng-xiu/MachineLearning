"""
1. 线性回归算法 Linear Regression

"""
from sklearn import linear_model,datasets

digits = datasets.load_digits()

clf = linear_model.LinearRegression()

x,y = digits.data[:-1],digits.target[:-1]

clf.fit(x,y)

y_pred = clf.predict([digits.data[-1]])
y_true = digits.target[-1]

print(y_pred)
print(y_true)



