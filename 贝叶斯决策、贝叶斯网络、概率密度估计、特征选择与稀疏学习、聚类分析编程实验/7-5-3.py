import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def normfun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

def unifun(x,a,b):
    y=[]
    for i in range(len(x)):
        y.append(1/(b-a))
    return  y

def norm_xy():
    X = np.random.uniform(0, 1, 100)
    X.sort()
    x_mean, x_std = norm.fit(X)
    print('mean, ', x_mean)
    print('x_std, ', x_std)
    Y = normfun(X, x_mean, x_std)
    return X,Y

X1,Y1=norm_xy()
plt.plot(X1,Y1,'red')
X2,Y2=norm_xy()
plt.plot(X2,Y2,'green')
X3,Y3=norm_xy()
plt.plot(X3,Y3,'blue')
X4=np.random.uniform(0, 1, 100)
X4.sort()
Y4=unifun(X4,0,1)
plt.plot(X4,Y4,'black')
plt.show()