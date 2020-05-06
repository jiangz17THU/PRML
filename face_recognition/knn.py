import matplotlib.pyplot as plt
from skimage import transform
from sklearn.linear_model import  LogisticRegression
import numpy as np
import skimage.io as io
import os, sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.neighbors import KNeighborsClassifier

dirlist=[]
label=[]
data=[]
dirs=os.listdir("./Pictures")
for dir in dirs:
    list=os.listdir("./Pictures/"+dir)
    dirlist.append(list)

for i in range(10):
    label_i=np.zeros((len(dirlist[i]),1))
    label_i[:,0]=i
    label.append(label_i)

for i in range(10):
    data_i = []
    for j in range(len(dirlist[i])):
        image=io.imread("./Pictures/"+str(i)+"/"+ dirlist[i][j],as_gray=True)
        data_i.append(image)
    data.append(data_i)

data=np.array(data)
data=np.transpose(data,(1,0,2,3))
label=np.array(label)
label=np.transpose(label,(1,0,2))  #转置，为了使得数据集中，图片的个数作为第一维，便于split

#此处数据集的选择可以人为地任意指定，比如想取第1，2，3，5，6个人的图像，只需在data和label的第二维输入【1，2，3，5，6】
x_train, x_test, y_train, y_test = train_test_split(data[:,0:10],label[:,0:10],test_size = 0.25,random_state = 33)
#判断选取的人数，if=2，执行roc
peoplenum=x_train.shape[1]

nsamples,n,nx, ny = x_train.shape
msamples,m,labelvalue=y_train.shape
x_train=x_train.reshape(n*nsamples,nx*ny)
y_train=y_train.reshape(m*msamples*labelvalue)

n,nsamples, nx, ny = x_test.shape
m,msamples,labelvalue=y_test.shape
x_test=x_test.reshape(n*nsamples,nx*ny)
y_test=y_test.reshape(m*msamples*labelvalue)

#"euclidean"（欧氏距离）, "chebyshev"（切比雪夫距离）, "manhattan"（曼哈顿距离），“cosine”余弦距离
clf= KNeighborsClassifier(n_neighbors=1,metric="euclidean")
clf.fit(x_train,y_train)
result=clf.score(x_test,y_test)
print(result)

if(peoplenum==2):

    y_pred_gbc = clf.predict_proba(x_test)[:, 1]  ###这玩意就是预测概率的
    #注意下面的pos_label是需要人为指定的，其为正例的标签值，比如二分类标签是1和6，选取大的值作为正例，即此处设为6即可
    fpr, tpr, threshold = roc_curve(y_test, y_pred_gbc)  ###画图的时候要用预测的概率

    roc_auc = auc(fpr,tpr)
    plt.title('ROC')
    plt.plot(fpr,tpr, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()