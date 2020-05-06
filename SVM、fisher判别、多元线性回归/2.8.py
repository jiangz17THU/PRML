from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

#训练回归模型
train =pd.read_csv("prostate_train.txt",sep='\t')
train=np.array(train)
x=train[:,0:4]
y=train[:,4]

lire=LinearRegression()
lire.fit(x,y)
print('多元线性回归的各项系数:',lire.coef_)               #输出多元线性回归的各项系数
print('多元线性回归的常数项的值:',lire.intercept_)          #输出多元线性回归的常数项的值

#利用评价指标测评测试集
test =pd.read_csv("prostate_test.txt",sep='\t')
test=np.array(test)
x_test=test[:,0:4]
y_test=test[:,4]
y_predict=lire.predict(x_test)
print('MSE',mean_squared_error(y_test,y_predict))
print('MAE',mean_absolute_error(y_test,y_predict))
print('r方：',r2_score(y_test,y_predict))
#考虑交叉项
x01=x[:,0]*x[:,1]
x01=np.reshape(x01,(67,1))
x02=x[:,0]*x[:,2]
x02=np.reshape(x02,(67,1))
x12=x[:,1]*x[:,2]
x12=np.reshape(x12,(67,1))

x_=np.append(x,x01,axis=1)
x_=np.append(x_,x02,axis=1)
x_=np.append(x_,x12,axis=1)
print('引入交叉项后')
lire.fit(x_,y)
print('多元线性回归的各项系数:',lire.coef_)               #输出多元线性回归的各项系数
print('多元线性回归的常数项的值:',lire.intercept_)          #输出多元线性回归的常数项的值
x01=x_test[:,0]*x_test[:,1]
x01=np.reshape(x01,(30,1))
x02=x_test[:,0]*x_test[:,2]
x02=np.reshape(x02,(30,1))
x12=x_test[:,1]*x_test[:,2]
x12=np.reshape(x12,(30,1))

x_test=np.append(x_test,x01,axis=1)
x_test=np.append(x_test,x02,axis=1)
x_test=np.append(x_test,x12,axis=1)
y_predict=lire.predict(x_test)
print('MSE:',mean_squared_error(y_test,y_predict))
print('MAE:',mean_absolute_error(y_test,y_predict))
print("r方：",r2_score(y_test,y_predict))