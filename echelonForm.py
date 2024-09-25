# Python code to transform a 4x4 matrix to echelon form.
# Also, catch a matrix that is not linearly independent.
# Requires 4x4 matrix
# Wylie Burt 9/18/2024
# Adapted from the course Mathematics for Machine Learning: linear Algebra from the Imperial College, London

import numpy as np

# class MatrixIsSingular(Exception): pass

# set up the matrix
A = np.array([
        [0, 7, -5, 3],
        [2, 8, 0, 4],
        [3, 12, 0, 5],
        [1, 3, 1, 3]
    ], dtype=np.float64)
print ('starting matrix')
print(A)

# Fix row 0. Transform [0,0] to 1.
if A[0,0] == 0 :
    A[0] = A[0] + A[1]
if A[0,0] == 0 :
    A[0] = A[0] + A[2]
if A[0,0] == 0 :
    A[0] = A[0] + A[3]
if A[0,0] == 0 :
    print ('MatrixIsSingular')
A[0] = A[0] / A[0,0]
print('Fixed row 0')
print(A)

#fix row 1. Transform [1,0] to 0, and [1,1] to 1
A[1] = A[1] - A[1, 0] * A[0]
print('fix sub diag of row 1')
print(A)
if A[1,1] == 0 :
    A[1] = A[1] + A[2]
    A[1] = A[1] - A[1,0] * A[0]
if A[1,1] == 0 :
    A[1] = A[1] + A[3]
    A[1] = A[1] - A[1,0] * A[0]
if A[1,1] == 0 :
    print('matrixsingular')
A[1] = A[1] / A[1,1]
print('fixed sub diag and main diag in row 1')
print(A)

#fix row 2.  Transform [2,0] and [2,1] to 0.  Also, transform [2,2] to 1.
A[2] = A[2] - A[2, 0] * A[0]
A[2] = A[2] - A[2, 1] * A[1]
print('fixed sub diag of row 2')
print(A)

if A[2,2] == 0 :
    A[2] = A[2] + A[3]
    print('step 1 in first if of fixing row 2')
    print(A)
    A[2] = A[2] - A[2,0] * A[0]
    A[2] = A[2] - A[2, 1] * A[1]
    print('step 2 in first if on row 2 main diag')
    print(A)
if A[2,2] == 0 :
    A[2] = A[2] + A[1]
    print('step 1 in second if of fixing row 2')
    print(A)
    A[2] = A[2] - A[2,0] * A[0]
    A[2] = A[2] - A[2, 1] * A[1]
    print('step 2 in second if of fixing row 2')
    print(A)
if A[2,2] == 0 :
    print('matrixsingular')

else :
    A[2] = A[2] / A[2,2]
    print('fixed sub diag and main diag row 2')
    print(A)

#fix row 3. Transform [3,0], [3,1], and [3,2] to 0.  Also transform [3,3] to 1.
A[3] = A[3] - A[3, 0] * A[0]
A[3] = A[3] - A[3, 1] * A[1]
A[3] = A[3] - A[3, 2] * A[2]
print('fixed sub diag of row 3')
print(A)

if A[3,3] == 0 :
    A[3] = A[3] + A[1]
    print('step 1 in first if of fixing row 3')
    print(A)
    A[3] = A[3] - A[3, 0] * A[0]
    A[3] = A[3] - A[3, 1] * A[1]
    A[3] = A[3] - A[3,0] * A[2]
    print('step 2 in first if on row 3 main diag')
    print(A)
if A[3,3] == 0 :
    A[3] = A[3] + A[2]
    print('step 1 in second if of fixing row 3')
    print(A)
    A[3] = A[3] - A[3, 0] * A[0]
    A[3] = A[3] - A[3, 1] * A[1]
    A[3] = A[3] - A[3,0] * A[2]
    print('step 2 in second if of fixing row 3')
    print(A)
if A[3,3] == 0 :
    print('matrixsingular')
else :
    A[3] = A[3] / A[3,3]
    print('fixed sub diag and main diag row 3')
    print(A)
