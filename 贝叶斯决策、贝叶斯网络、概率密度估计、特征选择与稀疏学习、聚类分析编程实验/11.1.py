import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
x=np.linspace(4,6,30)
y1=12-x
y3=11.5-x
y2=11-x
pca = PCA(n_components=2)
x=x.reshape(-1,1)
y1=np.array(y1).reshape(-1,1)
y2=np.array(y2).reshape(-1,1)
y3=np.array(y3).reshape(-1,1)
X=np.append(x,x,axis=0)
X=np.append(X,x,axis=0)
y=np.append(y1,y2,axis=0)
y=np.append(y,y3,axis=0)
X=np.append(X,y,axis=1)
Xtran=pca.fit_transform(X)
print(Xtran)
xtran=Xtran[:,0]
ytran=Xtran[:,1]
plt.scatter(xtran,ytran)
plt.scatter(x,y1)
plt.scatter(x,y2)
plt.scatter(x,y3)
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.show()
