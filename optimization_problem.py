import numpy as np
from scipy.optimize import  minimize
from RSD import *
import scipy
import math


c0     = 0.2
delta  = 0.05
deltad = 0.05
n      = 1000
k      = 50
R      = c0 * np.sqrt(k) * np.log(float(k)/delta)            # computation of R parameter


_, pdf, _ = Robust_Soliton_Distribution(n, k, c0, delta)  # See RSD doc
# Def of objective function
def objective(xd):
    func = np.sum( [xd[d]*(d+1)*pdf[d] for d in xrange(k)])
    return func


# Def constrain
def constraint1(xd,d):
    value = deltad
    for j in xrange(d):
        binomial = math.factorial(k) / ( math.factorial(j) * math.factorial(k - j))
        esponente = xd[d-1]*d/k
        p = 1-math.exp(-esponente)
        value -= binomial * (p**j) * ((1-p)**(k-j))

    return value

def constraint_larghi(xd,d):
    d += 1
    aa = (k/d)**d
    bb = math.exp(-xd[d-1]*d)
    value = deltad-aa * bb
    return value

def constraint_x(xd,d):
    return xd[d]-1


cons = []
for d in xrange(int(round(k/R))):
    con1 = {'type': 'ineq', 'fun': constraint_larghi, 'args': [d]}
    con2 = {'type': 'ineq', 'fun': constraint_x, 'args': [d]}
    print con1 , '\n',con2
    cons.append(con1)
    cons.append(con2)




# Def of initial points
x0 = np.ones(k)*1

b=[0]*k
for i in xrange(k):
    b[i] = (1.0,None)



# Solution
sol = minimize(objective, x0 , method='SLSQP',bounds=b ,  constraints=cons)
print sol


# L-BFGS-B
# 'SLSQP'
