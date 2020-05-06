'''''''''
经典的 MDS（Multidimensional Scaling）方法起源于当我们仅能获取到物体之间的距离的时候，如何由此重构它的坐标。
附件 city_dist.xlsx 中是 34 个城 市之间的相对距离，请用 MDS方法得到城市的二维表示并作图，
简要分析你 的可视化结果与真实地图上各个城市相对位置的差异。 
'''''''''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.manifold import MDS
df = pd.read_excel('./data/city_dist.xlsx')
data = df.values
D=data[:,1:36] #距离矩阵 34*34
label=data[:,0]
mds = MDS(dissimilarity='precomputed')
result = mds.fit_transform(D)
x=result[:,0]
y=result[:,1]
plt.scatter(x,y)
for i in range(34):
    plt.text(x[i]*1.01, y[i]*1.01, label[i],
            fontsize=10, color = "r", style = "italic", weight = "light",
            verticalalignment='center', horizontalalignment='right',rotation=0) #给散点加标签

plt.show()
