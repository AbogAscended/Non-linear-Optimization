"""
Let g(x)= x^2 - 2
I will use Newtons Method to approximate sqrt(2)
My initial guess will be x^{(0)} = 1
It will run until I have x^{(10)}
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# Establish initial guess and functions
x = np.zeros(10)
x[0] = 1
g = lambda x: x**2 - 2
gprime = lambda x: 2*x

print(f'Our initial guess is {x[0]}')
for i in range(1, len(x)):
    x[i] = x[i-1] - g(x[i-1])/gprime(x[i-1])
    print(f'X^({i}) = {x[i]}')

"""
Code Output:
Our initial guess is 1.0
X^(1) = 1.5
X^(2) = 1.4166666666666667
X^(3) = 1.4142156862745099
X^(4) = 1.4142135623746899
X^(5) = 1.4142135623730951
X^(6) = 1.414213562373095
X^(7) = 1.4142135623730951
X^(8) = 1.414213562373095
X^(9) = 1.4142135623730951
"""
k = np.linspace(0,9,10)
plt.plot(k,x)
plt.xlabel("$K^{th} iteration$")
plt.ylabel('$x^{(k)} value$')
plt.title("$g(x)=x^2 - 2$")
plt.savefig("NewtonsMethod/graphs/xsqrdvalues.png")
