import numpy as np
from sklearn.svm import SVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
import skimage.io as io
import os, sys
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt

def load_data(filename):
    dirlist = []
    label = []
    data = []
    dirs = os.listdir(filename)
    for dir in dirs:
        list = os.listdir(filename+ '/'+dir)
        dirlist.append(list)

    for i in range(10):
        label_i = np.zeros((len(dirlist[i]), 1))
        label_i[:, 0] = i
        label.append(label_i)

    for i in range(10):
        data_i = []
        for j in range(len(dirlist[i])):
            image = io.imread(filename + "/" +str(i) + "/" + dirlist[i][j], as_gray=True)
            data_i.append(image)
        data.append(data_i)

    data = np.array(data)
    data = np.transpose(data, (1, 0, 2, 3))
    label = np.array(label)
    label = np.transpose(label, (1, 0, 2))  # 转置，为了使得数据集中，图片的个数作为第一维，便于split

    # 此处数据集的选择可以人为地任意指定，比如想取第1，2，3，5，6个人的图像，只需在data和label的第二维输入【1，2，3，5，6】
    x_train, x_test, y_train, y_test = train_test_split(data[:, 0:10], label[:, 0:10],
                                                        test_size=0.25, random_state=33)
    # 判断选取的人数，if=2，执行roc
    peoplenum = x_train.shape[1]

    nsamples, n, nx, ny = x_train.shape
    msamples, m, labelvalue = y_train.shape
    x_train = x_train.reshape(n * nsamples, nx * ny)
    y_train = y_train.reshape(m * msamples * labelvalue)

    n, nsamples, nx, ny = x_test.shape
    m, msamples, labelvalue = y_test.shape
    x_test = x_test.reshape(n * nsamples, nx * ny)
    y_test = y_test.reshape(m * msamples * labelvalue)
    return x_train, x_test, y_train, y_test

def svm_c(x_train, x_test, y_train, y_test):
    # rbf核函数，设置数据权重
    '''svc = SVC(kernel='linear',C=1,tol=0.001)
    svc.fit(x_train, y_train)
    pred_y = svc.predict(x_test)
    print(classification_report(y_test,pred_y))'''


    c_range = [0.001,0.05,0.1,0.5,1,2.5,5,10,15]
    tol_range=[0.05,0.01,0.005,0.001,0.0005,0.0001]

    svc=SVC(probability=True)
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉
    param_grid = [{'kernel': ['linear'], 'C': c_range,'tol':tol_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    # 训练模型
    clf = grid.fit(x_train, y_train)
    # 计算测试集精度
    score = grid.score(x_test, y_test)

    print('精度为%s' % score)
    print("最优参数",clf.best_params_)


    if x_train.shape[0]==450:
        y_pred_gbc=clf.predict_proba(x_test)[:,1]
        # 注意下面的pos_label是需要人为指定的，其为正例的标签值，比如二分类标签是1和6，选取大的值作为正例，即此处设为6即可
        fpr, tpr, threshold = roc_curve(y_test, y_pred_gbc, pos_label=1)  ###画图的时候要用预测的概率

        roc_auc = auc(fpr, tpr)
        plt.title('ROC')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.show()



if __name__ == '__main__':
    svm_c(*load_data("./Pictures"))

