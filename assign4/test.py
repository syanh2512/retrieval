import numpy as np

# A = np.array([[0,1/2,0,0,0],[1/4,0,1,1,1/2],[1/4,0,0,0,0],[1/4,0,0,0,1/2],[1/4,1/2,0,0,0]])
A = np.array([[0,1,1,1,1],[1,0,0,0,1],[0,1,0,0,0],[0,1,0,0,0],[0,1,0,1,0]])
# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("eigenvalue:\n", np.abs(eigenvalues))
print("eigenvectors:\n", np.abs(eigenvectors))

max_eigenvalue_index = np.argmax(eigenvalues)

# Retrieve the eigenvector corresponding to the maximum eigenvalue
max_eigenvalue = eigenvalues[max_eigenvalue_index]
max_eigenvector = eigenvectors[:, max_eigenvalue_index]

print("\nMaximum Eigenvalue:\n", np.abs(max_eigenvalue))
print("Corresponding Eigenvector:\n", np.abs(max_eigenvector))