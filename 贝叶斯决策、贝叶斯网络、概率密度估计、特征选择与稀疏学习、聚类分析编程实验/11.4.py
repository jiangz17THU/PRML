import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import linear_model

x=np.load('./data/mnist.npz')
x_train=x['X_train']
y_train=x['y_train']
x_test=x['X_test']
y_test=x['y_test']
X_train=[]
Y_train=[]
X_test=[]
Y_test=[]
train_label=[]
test_label=[]

for i in range(len(y_train)):
    if y_train[i][0]==1:
        X_train.append(x_train[i])
        Y_train.append(y_train[i])
        train_label.append(0)
    elif y_train[i][8]==1:
        X_train.append(x_train[i])
        Y_train.append(y_train[i])
        train_label.append(8)
for i in range(len(y_test)):
    if y_test[i][0]==1:
        X_test.append(x_test[i])
        Y_test.append(y_test[i])
        test_label.append(0)
    elif y_test[i][8]==1:
        X_test.append(x_test[i])
        Y_test.append(y_test[i])
        test_label.append(8)
Y_test=np.array(Y_test)
X_test=np.array(X_test).reshape(-1,28*28)
X_train=np.array(X_train).reshape(-1,28*28)
Y_train=np.array(Y_train)
X=np.append(X_train,X_test,axis=0)
np.random.shuffle(X)
#list1=[i==0 for i in train_label]
#list2=[i==8 for i in train_label]
pca = PCA(n_components=50) #此处需要修改
isomap = Isomap( n_components=2)
tsne = TSNE(n_components=2)
lle=LocallyLinearEmbedding(n_components=2)
logreg = linear_model.LogisticRegression(solver='lbfgs',max_iter=8000)
#pca-分类
logreg.fit(X_train,train_label)
print('不做降维利用logistic回归分类准确率为：',logreg.score(X_test,test_label))
pca.fit(X)
X_test=pca.transform(X_test)
X_train=pca.transform(X_train)
logreg.fit(X_train,train_label)
print('pca降维后得到logistic回归分类准确率为：',logreg.score(X_test,test_label))
#pca
'''''''''
x_train_dec=pca.fit_transform(X_train)
x=x_train_dec[:,0]
y=x_train_dec[:,1]
plt.scatter(x,y,c=train_label)
plt.show()
'''''''''

#isomap
'''''''''
x_train_isomap = isomap.fit_transform(X_train)
x=x_train_isomap[:,0]
y=x_train_isomap[:,1]
plt.scatter(x,y,c=train_label)
plt.show()
'''''''''

#t-sne
'''''''''
x_train_tsne=tsne.fit_transform(X_train)
x=x_train_tsne[:,0]
y=x_train_tsne[:,1]
plt.scatter(x,y,c=train_label)
plt.show()
'''''''''

#LLE
'''''''''
x_train_lle=lle.fit_transform(X_train)
x=x_train_lle[:,0]
y=x_train_lle[:,1]
plt.scatter(x,y,c=train_label)
plt.show()
'''''''''