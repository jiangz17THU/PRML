import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc, roc_curve
data=[]
with open('./spambase.data','r') as f:
    X=f.readlines()
    for line in X:
        line=line.strip('\n').split(',')
        line=map(float,line)
        line=list(line)
        data.append(line)
data=np.array(data)

label=np.array(data[:,57])
data=data[:,0:57]
x_train, x_test, y_train, y_test = train_test_split(data,label,test_size = 1000/4601,random_state = 33)

clf = GaussianNB()
clf.fit(x_train,y_train)
print('正确率：',clf.score(x_test,y_test))
pred=clf.predict(x_test)
cm=confusion_matrix(y_test,pred)
print("混淆矩阵:\n", cm)


y_pred_gbc = clf.predict_proba(x_test)[:, 1]  ###这玩意就是预测概率的
# 注意下面的pos_label是需要人为指定的，其为正例的标签值，比如二分类标签是1和6，选取大的值作为正例，即此处设为6即可
fpr, tpr, threshold = roc_curve(y_test, y_pred_gbc,pos_label=1)  ###画图的时候要用预测的概率

roc_auc = auc(fpr,tpr)
plt.title('ROC')
plt.plot(fpr,tpr, 'b', label='AUC = %0.4f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()