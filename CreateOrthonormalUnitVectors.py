# perform the Gram-Schmidt procedure, which takes a list of vectors
# and forms an orthonormal basis from this set.
# As a corollary, the procedure allows us to determine the dimension
# of the space spanned by the basis vectors,
# which is equal to or less than the space which the vectors sit.
# Wylie Burt 9/20/2024 adapted from lab exercise in week 4 of
# Mathematics for Machine Learning: Linear Algebra from Imperial London College

# Test matrices
###################################################
# V = np.array([[1,0,2,6],
#               [0,1,8,2],
#               [2,8,3,1],
#               [1,-6,2,3]], dtype=np.float_)
# gsBasis4(V)
# Result:
# array([[ 0.40824829, -0.1814885 ,  0.04982278,  0.89325973],
#        [ 0.        ,  0.1088931 ,  0.99349591, -0.03328918],
#        [ 0.81649658,  0.50816781, -0.06462163, -0.26631346],
#        [ 0.40824829, -0.83484711,  0.07942048, -0.36063281]])
###################################################
# gsBasis(V)
# Result:
# Same as above
###################################################
# See what happens for non-square matrices
# A = np.array([[3,2,3],
#              [2,5,-1],
#              [2,4,8],
#              [12,2,1]], dtype=np.float_)
# gsBasis(A)
# Results:
# array([[ 0.23643312,  0.18771349,  0.22132104],
#        [ 0.15762208,  0.74769023, -0.64395812],
#        [ 0.15762208,  0.57790444,  0.72904263],
#        [ 0.94573249, -0.26786082, -0.06951101]])
###################################################
# dimensions(A)
# Results:
# 3.0
###################################################
# B = np.array([[6,2,1,7,5],
#               [2,8,5,-4,1],
#               [1,-6,3,2,8]], dtype=np.float_)
# gsBasis(B)
# Results:
# array([[ 0.93704257, -0.12700832, -0.32530002,  0.        ,  0.        ],
#        [ 0.31234752,  0.72140727,  0.61807005,  0.        ,  0.        ],
#        [ 0.15617376, -0.6807646 ,  0.71566005,  0.        ,  0.        ]])
###################################################
# dimensions(B)
# Results:
# 3.0
###################################################
# Now let's see what happens when we have one vector that is a linear combination of the others.
# C = np.array([[1,0,2],
#               [0,1,-3],
#               [1,0,2]], dtype=np.float_)
###################################################
# gsBasis(C)
# Results:
# array([[ 0.70710678,  0.        ,  0.        ],
#        [ 0.        ,  1.        ,  0.        ],
#        [ 0.70710678,  0.        ,  0.        ]])
# dimensions(C)
# Results:
# 2.0
###################################################

# GRADED FUNCTION
import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14 # That's 1×10⁻¹⁴ = 0.00000000000001

# Our first function will perform the Gram-Schmidt procedure for 4 basis vectors.
# We'll take this list of vectors as the columns of a matrix, A.
# We'll then go through the vectors one at a time and set them to be orthogonal
# to all the vectors that came before it. Before normalising.
# Follow the instructions inside the function at each comment.
# You will be told where to add code to complete the function.
# Better to use the function gsBasis to handle any size matrix
def gsBasis4(A) :
    B = np.array(A, dtype=np.float_) # Make B as a copy of A, since we're going to alter it's values.
    # The zeroth column is easy, since it has no other vectors to make it normal to.
    # All that needs to be done is to normalise it. I.e. divide by its modulus, or norm.
    B[:, 0] = B[:, 0] / la.norm(B[:, 0])
    # For the first column, we need to subtract any overlap with our new zeroth vector.
    B[:, 1] = B[:, 1] - B[:, 1] @ B[:, 0] * B[:, 0]
    # If there's anything left after that subtraction, then B[:, 1] is linearly independant of B[:, 0]
    # If this is the case, we can normalise it. Otherwise we'll set that vector to zero.
    if la.norm(B[:, 1]) > verySmallNumber :
        B[:, 1] = B[:, 1] / la.norm(B[:, 1])
    else :
        B[:, 1] = np.zeros_like(B[:, 1])
    # Now we need to repeat the process for column 2.
    # Insert two lines of code, the first to subtract the overlap with the zeroth vector,
    # and the second to subtract the overlap with the first.
    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 0] * B[:, 0]
    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 1] * B[:, 1]

    # Again we'll need to normalise our new vector.
    # Copy and adapt the normalisation fragment from above to column 2.
    if la.norm(B[:, 2]) > verySmallNumber :
        B[:, 2] = B[:, 2] / la.norm(B[:, 2])
    else :
        B[:, 2] = np.zeros_like(B[:, 2])


    # Finally, column three:
    # Insert code to subtract the overlap with the first three vectors.
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 0] * B[:, 0]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 1] * B[:, 1]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 2] * B[:, 2]

    # Now normalise if possible
    if la.norm(B[:, 3]) > verySmallNumber :
        B[:, 3] = B[:, 3] / la.norm(B[:, 3])
    else :
        B[:, 3] = np.zeros_like(B[:, 3])

    # Finally, we return the result:
    return B

# The second part of this exercise will generalise the procedure.
# Previously, we could only have four vectors, and there was a lot of repeating in the code.
# We'll use a for-loop here to iterate the process for each vector.
def gsBasis(A) :
    B = np.array(A, dtype=np.float_) # Make B as a copy of A, since we're going to alter it's values.
    # Loop over all vectors, starting with zero, label them with i
    for i in range(B.shape[1]) :
        # Inside that loop, loop over all previous vectors, j, to subtract.
        for j in range(i) :
            # Complete the code to subtract the overlap with previous vectors.
            # you'll need the current vector B[:, i] and a previous vector B[:, j]
            B[:, i] = B[:, i] - B[:, i] @ B[:, j] * B[:, j]
        # Next insert code to do the normalisation test for B[:, i]
        if la.norm(B[:, i]) > verySmallNumber :
            B[:, i] = B[:, i] / la.norm(B[:, i])
        else :
            B[:, i] = np.zeros_like(B[:, i])

    # Finally, we return the result:
    return B

# This function uses the Gram-schmidt process to calculate the dimension
# spanned by a list of vectors.
# Since each vector is normalised to one, or is zero,
# the sum of all the norms will be the dimension.
def dimensions(A) :
    return np.sum(la.norm(gsBasis(A), axis=0))
