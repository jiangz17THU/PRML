#参考https://www.cnblogs.com/sdxk/p/4084711.html
import numpy as np

def normal(mean, std, x):
    return (1 / (np.sqrt(2*np.pi)*std)) * (np.exp(-np.square(x-mean) / (2*np.square(std))))

def parzen(X, sample):
    prob = []
    n = len(X)
    for sam in sample:
        p = 0
        for x in X :
            p += np.exp(-np.square(x-sam)/2/np.square(0.5))  /  np.sqrt(2*np.pi)/0.5
            #exp(-(x1(j)-x(i))^2/2/h^2)/sqrt(2*pi)/h
        prob.append(p/n)
    prob = np.array(prob)
    return prob

def Bayes(prob, X, lamda1, lamda2):
    pred_Y = []
    for x in X:
        if (x < -10):
            y = 1
        elif (x > 10):
            y = 0
        else:
            index = int(10 * (x - 10))
            if (prob[0, index] * lamda2 > prob[1, index] * lamda1):
                y = 1
            else:
                y = 0
        pred_Y.append(y)
    return pred_Y

X = []
Y = []
X.append(np.random.normal(loc=2.5, scale=2, size=250))
X.append(np.random.normal(loc=-2.5, scale=1, size=250))
Y.append(np.ones(250))
Y.append(np.zeros(250))
X = np.array(X)
Y = np.array(Y)

#抽取训练集和测试集
index = np.arange(250)
np.random.shuffle(index)
test_index = np.array(index[175:])
train_index = np.array(index[:175])

train_X = X[:, train_index]
train_Y = Y[:, train_index]
test_X = X[:, test_index]
test_Y = Y[:, test_index]

#概率密度函数
sample = np.linspace(-8,8,num=200)
prob = []
prob.append(parzen(train_X[0],sample))
prob.append(parzen(train_X[1],sample))
prob.append(normal(2.5,2,sample))
prob.append(normal(-2.5,1,sample))
prob = np.array(prob)

#prediction
pred_Y = []
pred_Y.append(Bayes(prob,test_X[0],1,1))
pred_Y.append(Bayes(prob,test_X[1],1,1))
pred_Y = np.array(pred_Y)

#计算正确率
test_Y = test_Y.reshape(-1)
pred_Y = pred_Y.reshape(-1)
amount = len(test_Y)
count = 0
for i in range(amount):
    if (test_Y[i] == pred_Y[i]):
        count += 1
accuracy = count / amount
print('accuracy:'+str(accuracy))