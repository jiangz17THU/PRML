from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from minepy import MINE
import matplotlib.pyplot as plt

x_raw=np.loadtxt('./data/feature_selection_X.txt')
y=np.loadtxt('./data/feature_selection_Y.txt')
y=list(map(int,y))
x_train, x_test, y_train, y_test = train_test_split(x_raw,y,test_size = 0.25,random_state = 33)

def fisher(x,y,n):# input:narrays,label; output:narrays after feature selection
    x=np.array(x)
    y=np.array(y)
    x_0=x[y==0]

    m0=x_0.shape[0]
    x_1=x[y==1]
    m1=x_1.shape[0]
    n_features=x.shape[1]
    J=[] #fisher判据
    for i in range(n_features):
        #print(x_0[:,i])
        miu0=np.sum(x_0[:,i])/m0
        miu1=np.sum(x_1[:,i])/m1
        list1=[j-miu0 for j in x_0[:,i] ]
        list2=[k**2 for k in list1]
        s0=np.sum(list2)
        list3 = [j - miu1 for j in x_1[:, i]]
        list4 = [k ** 2 for k in list3]
        s1=np.sum(list4)
        jfw=(miu0-miu1)**2/(s0+s1)
        J.append(jfw)
    J_sort=sorted(J)
    J=np.array(J)
    n_selected=n

    x_selected=x[:,J>=J_sort[n_features-n_selected]]
    return x_selected,J>=J_sort[n_features-n_selected]
n_selected=20 #可以改动 为1 5 10 20 50 100
x_fisher_selected,fisher_boollist=fisher(x_train,y_train,n_selected)

logreg = linear_model.LogisticRegression(solver='lbfgs',max_iter=3000)
logreg.fit(x_fisher_selected,y_train)
x_fisher_test=x_test[:,fisher_boollist]
print("基于类间类内距离特征选择后，利用logstics回归的准确率是：",logreg.score(x_fisher_test,y_test))
logreg.fit(x_train,y_train)
print("直接利用logstics回归的准确率是：",logreg.score(x_test,y_test))
m=MINE()

mic=[]
n_features=x_raw.shape[1]
for i in range(x_raw.shape[1]):
    x=x_train[:,i]
    m.compute_score(x,y_train)
    mic.append(m.mic())
mic_sorted=sorted(mic)
mic=np.array(mic)
x_mic_selected=x_train[:,mic>=mic_sorted[n_features-n_selected]]
mic_boollist=mic>=mic_sorted[n_features-n_selected]
x_mic_test=x_test[:,mic_boollist]
logreg.fit(x_mic_selected,y_train)
print("基于最大信息系数特征选择后，利用logstics回归的准确率是：",logreg.score(x_mic_test,y_test))
num=0
for i in range(n_features):
    if(mic_boollist[i]==True and fisher_boollist[i]==True):
        num=num+1
print("两种方法选取的相同特征个数：",num)

#前向算法
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size = 1/3,random_state = 33)
X=[]
X_val=[]
map=[]
for i in range(n_selected):
    max_j=0
    max_score=0
    for j in range(n_features):
        if j not in map:
            if j == 0 and i == 0:
                X.append(x_train[:, j])
                X_val.append(x_val[:, j])
            else:
                x_tran = np.array(x_train[:, j]).reshape(-1, 1)
                X = np.append(X, x_tran, axis=1)
                x_val_tran = np.array(x_val[:, j]).reshape(-1, 1)
                X_val = np.append(X_val, x_val_tran, axis=1)

            X = np.array(X).reshape(x_train.shape[0], -1)
            logreg.fit(X, y_train)

            X_val = np.array(X_val).reshape(x_val.shape[0], -1)
            score = logreg.score(X_val, y_val)
            if score > max_score:
                max_j = j
                max_score = score
            X = X[:, 0:i]
            X_val = X_val[:, 0:i]
    x_tran = np.array(x_train[:, max_j]).reshape(-1, 1)
    X = np.append(X, x_tran, axis=1)
    x_val_tran = np.array(x_val[:, max_j]).reshape(-1, 1)
    X_val = np.append(X_val, x_val_tran, axis=1)
    map.append(max_j)
x_test=x_test[:,map]
print('前向算法特征选择后，利用logstics回归的准确率是：',logreg.score(x_test,y_test))
num=0
for i in range(n_features):
    if i in map and fisher_boollist[i]==True:
        num=num+1
print('类间类内选择特征和前向算法选择特征相同个数为：',num)

num=0
for i in range(n_features):
    if i in map and mic_boollist[i]==True:
        num=num+1
print('最大信息系数选择特征和前向算法选择特征相同个数为：',num)

from sklearn.tree import DecisionTreeClassifier
import heapq
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)

list1=list(clf.feature_importances_)

list1_sorted=sorted(list1)
treemap=[]
for i in range(n_features):
    if list1[i]>=list1_sorted[n_features-n_selected]:
        treemap.append(i)
num=0
for i in range(n_features):
    if i in treemap and fisher_boollist[i]==True:
        num=num+1
print('类间类内选择特征和决策树算法选择特征相同个数为：',num)

num=0
for i in range(n_features):
    if i in treemap and mic_boollist[i]==True:
        num=num+1
print('最大信息系数选择特征和决策树算法选择特征相同个数为：',num)

num=0
for i in range(n_features):
    if i in treemap and i in map:
        num=num+1
print('前向算法选择特征和决策树算法选择特征相同个数为：',num)