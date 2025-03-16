"""
Let f(x1,x2) = 2x1^4 + x2^2 - 4x1x2 + 5x2
gradient = [8*x1**3 - 4*x2, -4*x1 + 2*x2 + 5]
hessian = [[24*x1**2, -4], [-4, 2]]

[[24*(0)**2, -4], [-4, 2]](x^(1) - [0,0]) = -[8*(0)**3 - 4*(0), -4*(0) + 2*(0) + 5]
x^(1) = [1.25, 0]

[[24*(1.25)**2, -4], [-4, 2]](x^(2) - [1.25,0]) = -[8*(1.25)**3 - 4*(0), -4*(1.25) + 2*(0) + 5]
[[24*(1.25)**2, -4], [-4, 2]]([x1 - 1.25 , x2]) = -[8*(1.25)**3 - 4*(0), -4*(1.25) + 2*(0) + 5]
x^(2) = [0.72033898, -1.05932203]
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Setup our variables and function using sympy
x1, x2 = sp.symbols('x1 x2')
f = 2*(x1**4) + x2**2 - 4*x1*x2 + 5*x2

# Use sympy to calculate our gradient and hessian symbolically
gradient = [sp.diff(f,var) for var in (x1,x2)]
hessian  = [[sp.diff(g,var) for var in (x1,x2)] for g in gradient]

# Convert our symbolic solutions to lambda functions for solving
fn = sp.lambdify((x1,x2), f, 'numpy')
grad_fn = sp.lambdify((x1,x2), gradient, 'numpy')
hess_fn = sp.lambdify((x1,x2), hessian, 'numpy')



# Implementation of multivariable newtons method
x = np.zeros((10,2), dtype= np.float64)
for i in range(1,x.shape[0]):
    x[i:] = x[i-1,:] - np.linalg.inv(hess_fn(x[i-1,0],x[i-1,1]))@grad_fn(x[i-1,0],x[i-1,1])
    print(f'x^({i}) = {x[i,:]}')
"""
Output:
x^(1) = [1.25 0.  ]
x^(2) = [ 0.72033898 -1.05932203]
x^(3) = [-0.90260633 -4.30521267]
x^(4) = [-1.88402029 -6.26804059]
x^(5) = [-1.5157418 -5.5314836]
x^(6) = [-1.39412209 -5.28824418]
x^(7) = [-1.38057119 -5.26114239]
x^(8) = [-1.38040894 -5.26081788]
x^(9) = [-1.38040892 -5.26081783]
"""


x1Vals = np.linspace(-7,2,4000)
x2Vals = np.linspace(-7,2,4000)
X1, X2 = np.meshgrid(x1Vals,x2Vals)
Z = fn(X1,X2)

plt.figure(figsize=(8, 6))
plt.contour(X1, X2, Z, levels=25, cmap='viridis')
plt.colorbar(label='f(x1, x2)')

plt.plot(x[:, 0], x[:, 1], 'ro-', label="Newton's Iterates", linewidth=2, markersize=8)
plt.scatter(x[0, 0], x[0, 1], color='blue', label='Initial Guess', zorder=2)
plt.scatter(x[-1, 0], x[-1, 1], color='green', label='Final Estimate', zorder=2)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title("Convergence of Newton's Method")
plt.legend()
plt.grid(True)
plt.savefig('NewtonsMethod/graphs/multicontour.png')

"""
Let f(x1,x2) = 2x1^4 + x2^2 - 4x1x2 + 5x2
gradient = [8*x1**3 - 4*x2, -4*x1 + 2*x2 + 5]
[8*x1**3 - 4*x2, -4*x1 + 2*x2 + 5] = [0,0]
(x1, x2) = (-1.3804089170137677, -5.260817834027535)
There are also imaginary solutions but this is the only real solution
"""
solutions = sp.solve(gradient, (x1, x2))
for sol in solutions:
    print(f"x1 = {sol[0].evalf()}, x2 = {sol[1].evalf()}")

"""
Something that i find interesting is the methods based on newtons methods which
improve it that we studied in MAT 4010 such as broydens method. Broydens method
i find to be particularly interesting because as computing inverses is computationaly
expensive we instead approximate the inverse and update it each time using a secant approx.
This is known as broydens bad method opposed to broydens good method which approximatees the jacobian
instead of the inverse jacobian but then uses a system of linear equations to solve it.
"""



