# coding:utf-8
import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold

# 留一法
# data = [1,2,3,4]
# loo = LeaveOneOut()
#
# for train, test in loo.split(data):
#     print("%s %s" % (train, test))


# 交叉验证法 -- KFold, StratifiedKFold

X = np.array([
    [1,2,3,4],
    [11,12,13,14],
    [21,22,23,24],
    [31,32,33,34],
    [41,42,43,44],
    [51,52,53,54],
    [61,62,63,64],
    [71,72,73,74]
])

y = np.array([1,1,0,0,1,1,0,0])

folder = KFold(n_splits=4, random_state=0, shuffle=False)
sfolder = StratifiedKFold(n_splits=4, random_state=0, shuffle=False)

# Kfold:
print("Kfold:")
for train, test in folder.split(X, y):
    print("train:%s, test:%s" % (train, test))

# StratifiedKFold:
print("StratifiedKFold:")
for train, test in sfolder.split(X, y):
    print("train:%s, test:%s" % (train, test))