import numpy as np
from sklearn.linear_model import  LogisticRegression, SGDClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
def fisher(data):
    N=data.shape[1]
    #0 for benign, 1 for malignant
    maglignant=[] #阳性
    benign=[] #阴性
    for i in range(N):
        if(data[9,i]==0):
            benign.append(data[0:9,i])
        else:
            maglignant.append(data[0:9,i])
    benign=np.array(benign).T
    maglignant=np.array(maglignant).T
    benign=np.nan_to_num(benign)
    maglignant=np.nan_to_num(maglignant)
    #类均值向量（x空间）

    m0=np.mean(benign,axis=1)
    m1=np.mean(maglignant,axis=1)
    #print(benign.shape)
    s0=np.zeros([9,9])
    s1=np.zeros([9,9])
    for i in range(benign.shape[1]):
        tmp=(benign[:,i]-m0).reshape([9,1])
        s0+=np.dot(tmp,tmp.T)

    for i in range(maglignant.shape[1]):
        tmp=(maglignant[:,i]-m1).reshape([9,1])
        s1+=np.dot(tmp,tmp.T)

    sw=s0+s1
    #print(maglignant.shape[1])

    w=np.dot(np.linalg.inv(sw),m0-m1).reshape([9,1])
    y0=np.dot(w.T,benign)
    y1=np.dot(w.T,maglignant)

    m0_new=np.mean(y0,axis=1)
    m1_new=np.mean(y1,axis=1)
    w0=-1/2 *(m0_new+m1_new)
    return w,w0

def acc_test(data,w,w0):
    N=data.shape[1]
    right=0
    for i in range(data.shape[1]):
        ans=np.dot(w.T,data[0:9,i])+w0
        if ((ans>0) and (data[9,i]==0)):
            right=right+1
        if((ans<0) and (data[9,i]==1)):
            right=right+1
    return right/N

if __name__ == '__main__':
    data = np.genfromtxt('breast-cancer-wisconsin.txt', delimiter='\t')
    data=data.astype(float)
    train, test = train_test_split(data, test_size=0.3)
    train=train.T
    train=train[1:11] #刨除第一行code number 个人觉得无用
    #print(data.shape)
    test=test.T
    test=test[1:11]

    w,w0=fisher(train)
    print("fisher分类准确率:",acc_test(test,w,w0))

    #logitics回归
    data1=pd.read_csv('breast-cancer-wisconsin.txt',sep="\t")
    data1 = data1.replace(to_replace='?', value=np.nan)
    data1 = data1.dropna(how='any')
    data1 = data1.values
    x_train, x_test, y_train, y_test = train_test_split(data1[:,1:10],data1[:,10],test_size = 0.3,random_state = 33)
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)  # 对x_train进行标准化
    x_test = ss.transform(x_test)
    lr = LogisticRegression()  # 初始化逻辑斯蒂回归模型
    y_train=y_train.astype('int')

    lr.fit(x_train, y_train)
    lr_y_predict = lr.predict(x_test)
    #print(x_test,"\n",lr_y_predict)
    right = 0
    for i in range(x_test.shape[0]):

        if(y_test[i]==lr_y_predict[i]):
            right=right+1
    acc=right/x_test.shape[0]
    print("logistics回归准确率为：",acc)