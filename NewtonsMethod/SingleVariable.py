"""
Let g(x)= x^2 - 2
I will use Newtons Method to approximate sqrt(2)
My initial guess will be x^{(0)} = 1
It will run until I have x^{(10)}
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
print(mpl.rcParams['text.usetex'])
print(mpl.rcParams['text.latex.preamble'])

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

# Graph generation
k = np.linspace(0,9,10)
plt.plot(k,x)
plt.xlabel("$K^{th} iteration$")
plt.ylabel('$x^{(k)} value$')
plt.title("$g(x)=x^2 - 2$")
plt.savefig("NewtonsMethod/graphs/xsqrdvalues.png")


"""
Let g(x) = x^10 - 10,
I will then approximate the tenth root of 10 using newtons method.
My intial guess will be 100
and I will again run it 9 times.
"""

# Establish initial guess and functions
x = np.zeros(50)
x[0] = 100
g = lambda x: x**10 - 10
gprime = lambda x: 10*x**9

print(f'Our initial guess is {x[0]}')
for i in range(1, len(x)):
    x[i] = x[i-1] - g(x[i-1])/gprime(x[i-1])
    print(f'X^({i}) = {x[i]}')

"""
Code Output:
Our initial guess is 100.0
X^(1) = 90.0
X^(2) = 81.0
X^(3) = 72.9
X^(4) = 65.61
X^(5) = 59.049
X^(6) = 53.1441
X^(7) = 47.82969
X^(8) = 43.046721
X^(9) = 38.7420489
X^(10) = 34.867844010000006
X^(11) = 31.38105960900002
X^(12) = 28.242953648100052
X^(13) = 25.418658283290135
X^(14) = 22.876792454961347
X^(15) = 20.589113209465793
X^(16) = 18.53020188852072
X^(17) = 16.677181699672527
X^(18) = 15.009463529715296
X^(19) = 13.508517176769631
X^(20) = 12.15766545915943
X^(21) = 10.941898913415812
X^(22) = 9.847709022519032
X^(23) = 8.862938121415239
X^(24) = 7.976644312237185
X^(25) = 7.178979888662701
X^(26) = 6.461081919540441
X^(27) = 5.814973778549137
X^(28) = 5.233476532237954
X^(29) = 4.710129218551442
X^(30) = 4.239117173100808
X^(31) = 3.815207717939746
X^(32) = 3.4336927851166354
X^(33) = 3.0903385777787187
X^(34) = 2.781343619627187
X^(35) = 2.5033096517615743
X^(36) = 2.2532377277807867
X^(37) = 2.0285818941082954
X^(38) = 1.8274426699415742
X^(39) = 1.6490979315809702
X^(40) = 1.4952743197409994
X^(41) = 1.3725084910183205
X^(42) = 1.2931158049591427
X^(43) = 1.2627194672492403
X^(44) = 1.25897630239575
X^(45) = 1.2589254210501666
X^(46) = 1.2589254117941675
X^(47) = 1.2589254117941673
X^(48) = 1.2589254117941673
X^(49) = 1.2589254117941673
"""

# Graph generation
k = np.linspace(0,len(x)-1,len(x))
plt.plot(k,x)
plt.xlabel("$K^{th} iteration$")
plt.ylabel('$x^{(k)} value$')
plt.title("$g(x)=x^{10} - 10$")
plt.savefig("NewtonsMethod/graphs/xtenthvalues.png")