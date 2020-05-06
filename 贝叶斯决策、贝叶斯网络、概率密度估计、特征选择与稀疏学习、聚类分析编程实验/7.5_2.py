import numpy as np
import matplotlib.pyplot as plt

def normal(mean, std, x):
    return (1 / (np.sqrt(2*np.pi)*std)) * (np.exp(-np.square(x-mean) / (2*np.square(std))))

num=1000
x = np.linspace(-3, 3)
y = [normal(0, 1, x)]
sample = np.random.normal(loc=0, scale=1, size=1000)
mean = np.mean(sample)
var = np.var(sample)
std = np.std(sample)

for ratio in (0.01,0.1,1,10):
    varr = ratio*var
    meann = (num*varr)/(num*varr+var)*mean + (-5)*var/(num*varr+var)
    varr = varr+var
    y.append(normal(meann, np.sqrt(varr), x))

plt.plot(x, y[0], label='Normal Distribution')
plt.plot(x, y[1], label='σ0^2=0.01σ^2')
plt.plot(x, y[2], label='σ0^2=0.1σ^2')
plt.plot(x, y[3], label='σ0^2=σ^2')
plt.plot(x, y[4], label='σ0^2=10σ^2')
plt.legend()
plt.title("pdf of miu")
plt.savefig('7.5_2.png')
plt.show()