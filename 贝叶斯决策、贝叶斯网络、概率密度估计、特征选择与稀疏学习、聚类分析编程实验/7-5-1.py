import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#print(X)


def normfun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

def norm_xy(num):
    X = np.random.normal(0, 1, num)
    X.sort()
    x_mean, x_std = norm.fit(X)
    print('mean, ', x_mean)
    print('x_std, ', x_std)
    Y = normfun(X, x_mean, x_std)
    return X,Y
X1,Y1=norm_xy(1000)
plt.plot(X1,Y1,'red')
X2,Y2=norm_xy(1000)
plt.plot(X2,Y2,'green')
X3,Y3=norm_xy(1000)
plt.plot(X3,Y3,'blue')
X4=np.random.normal(0, 1, 1000)
X4.sort()
Y4=normfun(X4,0,1)
plt.plot(X4,Y4,'black')
plt.show()