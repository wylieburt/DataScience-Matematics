# Takes a 3x3 and a 3x1 matrices and performs gaussian elimination
# followed by back substitution to solve 3 simultaneous linear equations.
import numpy as np

def print_matrix(matrix):
    if matrix.ndim == 1:
        # If it's a 1D array, print it as a column vector
        for value in matrix:
            print(f"{value:7.2f}")
    else:
        # If it's a 2D array, print it as before
        for row in matrix:
            print(" ".join(f"{x:7.2f}" for x in row))
    print()

def gaussian_elimination(A, b):
    # Combine A and b into an augmented matrix
    augmented = np.column_stack((A, b))
    n = len(b)

    for i in range(n):
        # Find pivot
        max_element = abs(augmented[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > max_element:
                max_element = abs(augmented[k][i])
                max_row = k

        # Swap maximum row with current row
        augmented[i], augmented[max_row] = augmented[max_row], augmented[i].copy()

        print(f"Step {i+1}:")
        print("Current matrix:")
        print_matrix(augmented)

        # Make all rows below this one 0 in current column
        for k in range(i + 1, n):
            c = -augmented[k][i] / augmented[i][i]
            for j in range(i, n + 1):
                if i == j:
                    augmented[k][j] = 0
                else:
                    augmented[k][j] += c * augmented[i][j]

            print(f"Eliminating below row {i+1}:")
            print_matrix(augmented)

    return augmented[:, :n], augmented[:, n]

def back_substitution(A, b):
    n = len(b)
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= A[i, j] * x[j]
        x[i] /= A[i, i]

        print(f"Step {n-i}: Solved for x[{i}] = {x[i]:.2f}")

    return x

# Example usage
A = np.array([[0, 4],
              [0, 2]], dtype=float)

b = np.array([0, 0], dtype=float)

print("Original system:")
print("Coefficient matrix A:")
print_matrix(A)
print("Constants vector b:")
print_matrix(b)

print("Performing Gaussian elimination...")
A_echelon, b_echelon = gaussian_elimination(A, b)

print("Final result:")
print("Coefficient matrix A in echelon form:")
print_matrix(A_echelon)
print("Constants vector b in echelon form:")
print_matrix(b_echelon)

print("Performing back substitution...")
solution = back_substitution(A_echelon, b_echelon)

print("\nSolution:")
for i, value in enumerate(solution):
    print(f"x[{i}] = {value:.2f}")

# Verify the solution
print("\nVerifying the solution:")
for i, equation in enumerate(A):
    result = np.dot(equation, solution)
    print(f"Equation {i+1}: {result:.2f} = {b[i]:.2f}")
