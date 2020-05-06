import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn.cluster import KMeans #导入KMeans包
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.metrics.cluster.supervised", lineno=746)
x=np.load('./data/mnist.npz')
x_train=x['X_train']
y_train=x['y_train']
X_train=[]
Y_train=[]
train_label=[]

for i in range(len(y_train)):
    if y_train[i][0]==1:
        X_train.append(x_train[i])
        Y_train.append(y_train[i])
        train_label.append(0)
    elif y_train[i][1]==1:
        X_train.append(x_train[i])
        Y_train.append(y_train[i])
        train_label.append(1)
    elif y_train[i][2]==1:
        X_train.append(x_train[i])
        Y_train.append(y_train[i])
        train_label.append(2)

X_train=np.array(X_train).reshape(-1,28*28)
Y_train=np.array(Y_train)
K = range(1, 11)
meandistortions = []
silhouette=[]
#利用误差平方和评估聚类
'''''''''
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_train)
    meandistortions.append(kmeans.inertia_)
#sum(np.min(cdist(X_train, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X_train.shape[0]
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.show()
'''''''''

'''''''''
#利用轮廓系数评估聚类
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_train)
    pre_y=kmeans.predict(X_train)
    sil_score=sm.silhouette_score(X_train,pre_y,sample_size=len(X_train),metric='euclidean')
    silhouette.append(sil_score)
plt.plot(K, silhouette, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Coefficient')
plt.show()
'''''''''

'''''''''
NMI=[]
#print(X_train.shape[0]) 18623
initpoint=np.array(X_train[[0,100,200]])
#print(train_label)
for i in range(10):
    kmeans = KMeans(n_clusters=3, init='k-means++',n_init=1)
    labels_pred = kmeans.fit_predict(X_train)
    #print(labels_pred)
    nmi=sm.adjusted_mutual_info_score(train_label, labels_pred)
    NMI.append(nmi)
    print("第%d次的NMI值是%f"%(i+1,nmi))
print("初始化为'自定义'时，NMI为%f"%(np.sum(NMI)/10))
'''''''''


NMI=[]
#print(X_train.shape[0]) 18623
#print(train_label)
for i in range(10):
    clst=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='average')
    labels_pred = clst.fit_predict(X_train)
    #print(labels_pred)
    nmi=sm.adjusted_mutual_info_score(train_label, labels_pred)
    NMI.append(nmi)
    print("第%d次的NMI值是%f"%(i+1,nmi))
print("距离为'euclidean'时，NMI为%f"%(np.sum(NMI)/10))