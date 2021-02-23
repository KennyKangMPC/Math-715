"""This script program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit cricle with source f given by

    f(x, y) = 4

and boundary conditions given by

    u(x, y) = 0        
"""

from __future__ import print_function
from fenics import *
from mshr import *
import matplotlib.pyplot as plt

# Create mesh 
domain = Circle(Point(0, 0), 1) 
mesh = generate_mesh(domain, 8)

# Define function spae
V_h = FunctionSpace(mesh, 'P', 1)

#############################################
# Define boundary conditions
#############################################

# define overal expression
u_D = Expression('1 - x[0]*x[0] - x[1]*x[1]', degree=3)

# define where the expression will be evaluated
def boundary(x, on_boundary):
    return on_boundary

# define the BC using the space and the function above
bc = DirichletBC(V_h, u_D, boundary)

#############################################
# Define variational problem
#############################################

# define trial space

  
# define test space


# define right-hand side



# define the bilinear form
a =

# define the linear form
L = 

#############################################
# Compute the solution 
#############################################

# define function where we will store the solution
u_h = 

# solve the variational problem
solve(a == L, u_h, bc)

#############################################
# Plot and compute error
#############################################

# Plot solution and mesh
plot(u_h)
plot(mesh)

# Compute error in L2 norm
error_L2 = errornorm(u_D, u_h, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u_h.compute_vertex_values(mesh)
import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

# Hold plot
plt.show()