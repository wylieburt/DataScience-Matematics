import numpy as np
import numpy.linalg as la

def findDiag(T, C) :
    print ('Find the diagonal matrix given T and C')
    Cinv = la.inv(C)
    D = Cinv @ T @ C
    print('T equals', T)
    print('C equals', C)
    print('The inverse of C', Cinv)
    print('D = Cinv * T * C')
    print('D the diagonal matrix', D)
    return D

def findT(C, D) :
    print('Finding T')
    Cinv = la.inv(C)
    T = C @ D @ Cinv
    print('Inverse of C', Cinv)
    print('T equals', T)
    return T

def findPowerOfT(T, power) :
    Tpower = la.matrix_power(T, power)
    print('T to the power of', power,Tpower)
    return Tpower

# Inputs
# Depending on the function you want to call uncomment the input variables needed.
T = np.array([[3/2, -1], [-1/2, 1/2]])
C = np.array([[(1 - math.sqrt(5)), (1 + math.sqrt(5))], [1, 1]])
D = findDiag(T, C)
# D = np.array([[1, 0], [0, 1]])
# T = np.array([[1, 0], [2, -1]])
# T = findT(C, D)
# power = 5
# Tp = findPowerOfT(T, power)

# This is Eigen stuff
#print('Eigen stuff for A')
#eigenvals, eigenmat = la.eig(T)
#print(eigenmat)
#print(eigenvals)
#print(np.round(eigenmat, decimals=1))
#print(np.round(eigenvals, decimals=1))
