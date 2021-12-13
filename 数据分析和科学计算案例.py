##-----解线性方程组
#例1
import numpy as np
from sympy.core import symbol
A = np.mat("-1,3,-5; 2,-2,4; 1,3,0")
A
b = np.mat("-3 8 6").T
b
r = np.linalg.solve(A,b)
r
#例2
from sympy.solvers.solveset import linsolve
x1 ,x2 ,x3, x4 =symbol.symbols("x1 x2 x3 x4")
linsolve([x1 + 3*x2 - 4*x3 + 2*x4, 
            3*x1 - x2 + 2*x3 - x4, 
            -2*x1 + 4*x2 - x3 + 3*x4, 
            3*x1 +9*x2 - 7*x3 + 6*x4], 
        (x1, x2, x3, x4))

##-------------假设检验
from scipy.stats import chisquare
chisquare([1e10-1e6, 1e10+1.5e6, 1e10-2e6, 1e10+4e6, 1e10-3e6, 1e10+0.5e6])
from scipy.stats import chi2
chi2.isf(0.05,(6-1))
p_value = 1 - chi2.cdf(3250,(6-1))
p_value
