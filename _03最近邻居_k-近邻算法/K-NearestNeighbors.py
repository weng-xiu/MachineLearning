"""
3. 最近邻居/k-近邻算法 (K-Nearest Neighbors,KNN)
"""
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()

clf = KNeighborsClassifier(n_neighbors=6)

x,y = digits.data[:-1],digits.target[:-1]

clf.fit(x,y)

y_pred = clf.predict([digits.data[-1]])
y_true = digits.target[-1]

print(y_pred)
print(y_true)


