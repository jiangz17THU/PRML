'''请编写代码生成以下仿真数据，探索线性回归、岭回归和LASSO回归模型对共线性问题的表现。
y = 3𝑥1 + 2 + 𝜀1,
𝑥1 = 1,…, 20
𝑥2 = 0.05𝑥1 + 𝜀2
𝜀1 ∈ N(0, 2), 𝜀2 ∈ N(0, 0.5)
若我们将与𝑥1有强相关关系的噪声𝑥2误认为是一维特征（即输入特征变为了[𝑥1, 𝑥2]），请同学们尝试使用上述三种模型对y进行回归，并回答以下问题。
(1)请给出𝑥1, 𝑥2的相关系数。
(2)请多次生成数据，观察正则化系数为1情况下三种模型拟合参数的稳定性。
(3)针对于岭回归和LASSO，调整正则化系数（调整范围不要过大，0~10之间即可），你能发现什么。'''
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

x_draw=np.linspace(1, 100, 100)
coef1=[]
coef2=[]
intercept=[]
for i in range(100):
    x1 = np.linspace(1, 20, 20)
    e1 = np.random.normal(0, np.sqrt(2), 20)
    e2 = np.random.normal(0, np.sqrt(0.5), 20)
    x2 = 0.05 * x1 + e2
    y = 3 * x1 + 2 + e1

    r = np.corrcoef(x1, x2)
    print("𝑥1, 𝑥2的相关系数:", r[0][1])
    x1 = np.array(x1).reshape(-1, 1)
    x2 = np.array(x2).reshape(-1, 1)

    y = np.array(y).reshape(-1, 1)
    x = np.append(x1, x2, axis=1)

    model1 = linear_model.LinearRegression()
    model1.fit(x, y)
    '''''''''
    coef1.append(model1.coef_[0][0])
    coef2.append(model1.coef_[0][1])
    intercept.append(model1.intercept_[0])
    '''''''''

    model2 = linear_model.Ridge(alpha=10)
    model2.fit(x, y)
    '''''''''
    coef1.append(model2.coef_[0][0])
    coef2.append(model2.coef_[0][1])
    intercept.append(model2.intercept_[0])
    '''''''''
    model3 = linear_model.Lasso(alpha=1)
    model3.fit(x, y)

    coef1.append(model3.coef_[0])
    coef2.append(model3.coef_[1])
    intercept.append(model3.intercept_[0])

plt.plot(x_draw,coef1,label=u"θ1变化曲线")
plt.plot(x_draw,coef2,label=u"θ2变化曲线")
plt.plot(x_draw,intercept,label=u"b变化曲线")

plt.ylim(-4,6)
plt.legend()
plt.show()