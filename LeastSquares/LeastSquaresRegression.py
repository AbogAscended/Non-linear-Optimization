import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 500
p = 1000
mean = 1
std = 100
x = np.linspace(-p,p,n)
y = np.random.uniform() + np.random.uniform()*x + np.random.normal(mean,std,n)
bstar = (n*np.mean(x)*np.mean(y) - np.dot(x,y)) / (n*(np.mean(x)**2) - np.dot(x,x))
astar = np.mean(y) - bstar*np.mean(x)
ystar = astar + bstar*x

plt.plot(x,y, label='Data',alpha=.7)
plt.plot(x,ystar,label='Regression line',color='red')
plt.title('Regression on a random data set')
plt.grid(True)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.show()

a = np.linspace(-1000,1000,500)
b = np.linspace(-1000,1000,500)
f = (a + b*x - y)**2
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x, y, f, c='blue', marker='o', label="Data Points")
fstar = (astar + bstar*np.mean(x) - np.mean(y))**2
ax.scatter(astar,bstar,fstar,s=400, color='red')
ax.set_xlabel("A")
ax.set_xlabel("B")
ax.set_xlabel("F")
plt.show()
