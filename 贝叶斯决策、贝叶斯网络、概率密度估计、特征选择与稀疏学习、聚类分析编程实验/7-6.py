import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

pos = np.random.normal(-2.5,1, 250)
neg = np.random.normal(2.5,1, 250)
pos_train, pos_test, neg_train, neg_test = train_test_split(pos,neg,test_size = 0.3,random_state = 33)
pos_train=np.array(pos_train)
pos_train=pos_train.reshape(-1,1)
neg_train=np.array(neg_train)
neg_train=neg_train.reshape(-1,1)
pos_test=np.array(pos_test)
neg_test=np.array(neg_test)
X1_plot = np.linspace(-10, 10, 1001)[:, np.newaxis]
kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(pos_train)
log_dens1 = kde.score_samples(X1_plot)
plt.plot(X1_plot[:, 0], np.exp(log_dens1), 'red')

X2_plot = np.linspace(-10, 10, 1001)[:, np.newaxis]
kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(neg_train)
log_dens2 = kde.score_samples(X1_plot)
plt.plot(X1_plot[:, 0], np.exp(log_dens2), 'blue')


count = 0
for i in range(75):
    index = int((np.round(pos_test[i]*50)/50+10)/0.02)
    p1 = 1-sum(np.exp(log_dens1[0:index]))
    p2 = 1-sum(np.exp(log_dens2[index:1001]))
    if (p1 >= p2):
        label_predict = 0
    else:
        label_predict = 1
    if (label_predict == 0):
        count = count+1
for i in range(75):
    index = int((np.round(neg_test[i]*50)/50+10)/0.02)
    p1 = 1-sum(np.exp(log_dens1[0:index]))
    p2 = 1-sum(np.exp(log_dens2[index:1001]))
    if (p1 >= p2):
        label_predict = 0
    else:
        label_predict = 1
    if (label_predict == 1):
        count = count+1
accuracy = count/150
# print(count)
print("最小错误率贝叶斯决策正确率："+str(accuracy))

count = 0
for i in range(75):
    index = int((np.round(pos_test[i]*50)/50+10)/0.02)
    p1 = 10*(1-sum(np.exp(log_dens1[index:1001])))
    p2 = 1-sum(np.exp(log_dens2[0:index]))
    if (p1 <= p2):
        label_predict = 0
    else:
        label_predict = 1
    if (label_predict == 0):
        count = count+1

for i in range(75):
    index = int((np.round(neg_test[i]*50)/50+10)/0.02)
    p1 = 10*(1-sum(np.exp(log_dens1[index:1001])))
    p2 = 1-sum(np.exp(log_dens2[0:index]))
    if (p1 <= p2):
        label_predict = 0
    else:
        label_predict = 1
    if (label_predict == 1):
        count = count+1
accuracy = count/150
print("最小风险贝叶斯决策正确率："+str(accuracy))
plt.show()

