"""This program solves the Eigenvalue problem

    - div grad u(x, y) = labda u(x, y)

on the unit circle and boundary conditions given by

    u(x, y) = 0        
"""


from __future__ import print_function
from fenics import *
from mshr import *
import matplotlib.pyplot as plt

# Create mesh 
domain = Circle(Point(0, 0), 1) 
mesh = generate_mesh(domain, 16)

# Define function space
V_h = 

#############################################
# Define boundary conditions
#############################################

# define overal expression
u_D = 

# define where the expression will be evaluated
def boundary(x, on_boundary):
    return on_boundary

# define the BC using the space and the function above
bc = DirichletBC(V_h, u_D, boundary)

#############################################
# Define variational problem
#############################################

# define trial space
u = 

# define test space
v = 

# define the bilinear form
a = 

#############################################
# Assemble the final matrix
#############################################

# define where to store the matrix
A = PETScMatrix()

# assemble the stiffness matrix and store in A
assemble(a, tensor=A)

# apply the boundary conditions
bc.apply(A)

#############################################
# Compute eigenvalues
#############################################

# define eigen solver
eigensolver = SLEPcEigenSolver(A)

# Compute all eigenvalues of A x = \lambda x
print("Computing eigenvalues. This can take a minute.")
eigensolver.solve()

#############################################
# Extract and plot eigenfunctions
#############################################
# Extract smallest (last) eigenpair
r, c, rx, cx = eigensolver.get_eigenpair(A.array().shape[0]-1)

# print eigenvalue
print("Smallest eigenvalue: ", r)

# Initialize function and assign eigenvector
u = Function(V_h)
u.vector()[:] = rx

# Plot eigenfunction
plot(u)
plt.show()
