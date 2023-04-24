"""
2. 支持向量机算法(Support Vector Machine,SVM)

"""
from sklearn import svm,datasets

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001)

x,y = digits.data[:-1],digits.target[:-1]

clf.fit(x,y)

y_pred = clf.predict([digits.data[-1]])
y_true = digits.target[-1]

print(y_pred)
print(y_true)




