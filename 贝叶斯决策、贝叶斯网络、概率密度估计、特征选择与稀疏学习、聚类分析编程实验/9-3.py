import numpy as np
from hmmlearn import hmm

X=np.load("./sequences.npy")
X=X.reshape(6000,1)
X=X-1
model=hmm.MultinomialHMM(n_components=2,n_iter=1000,tol=0.0005)
model.fit(X,lengths=[30]*200)

print("初始概率",model.startprob_)
print('转移概率',model.transmat_)
print('发射概率',model.emissionprob_)

test=np.array([[3,2,1,3,4,5,6,3,1,4,1,6,6,2,6]]).T
test=test-1
print('预测隐藏状态',model.predict(test))