# functions to help you
import numpy as np
import scipy.sparse as spsp
from scipy.sparse.linalg import spsolve
import scipy.integrate as integrate

class Mesh:
  def __init__(self, points):
    # self.p    array with the node points (sorted) type : np.array dim: (n_p)
    # self.n_p  number of node points               type : int
    # self.s    array with indices of points per    type : np.array dim: (n_s, 2) 
    #           segment  
    # self.n_s  number of segments                  type : int
    # self.bc.  array with the indices of boundary  type : np.array dim: (2)
    #           points

    self.p   =
    self.n_p = 
    
    self.s   = 
    self.n_s = 
    
    self.bc  = 


class V_h:
  def __init__(self, mesh):
    # self.mesh Mesh object containg geometric info type: Mesh
    # self.sim  dimension of the space              type: in

    self.mesh = 
    self.dim  = 

  def eval(self, xi, x):
    """ evaluation of the piece wise local polynomial given by
       the coefficients xi, at the point x 
    """

    # compute the index of the interval in which x is contained


    # compute the size of the interval


    return # here return the value of the fucnciton 

class Function:
  def __init__(self, xi, v_h):
    self.xi  = 
    self.v_h = 

  def __call__(self,x):
    # wrapper for calling eval in V_h
    
    # use the fucntion defined in v_h
    return 


def mass_matrix(v_h):

  # sparse matrix easy to change sparsity pattern
  # this initializes an empty sparse matrix of 
  # size v_h.dim x v_h.dim
  M = spsp.lil_matrix((v_h.dim,v_h.dim))

  # for loop
  for i in range(v_h.mesh.n_s):
    # extract the indices


    # compute the lengh of the segment


    # add the values to the matrix



  return M

def stiffness_matrix(v_h, sigma):

  # matrix easy to change sparsity pattern
  S = spsp.lil_matrix((v_h.dim,v_h.dim))

  # for loop
  for i in range(v_h.mesh.n_s):
    # extract the indices


    # compute the lengh of the segment


    # sample sigma


    # update the stiffness matrix
 

  return S

# show differences between Trapezoidal rule and Simpson rule
def load_vector(v_h, f):

  # allocate the vector
  b = np.zeros(v_h.dim)

  # for loop over the segments
  for i in range(v_h.mesh.n_s):
    # extracting the indices


    # computing the lenght of the interval 


    # update b


  return b


def source_assembler(v_h, f, u_dirichlet):
  # computing the load vector (use the function above)


  # extract the interval index for left boundary


  # compute the lenght of the interval


  # sample sigma at the middle point


  # update the source_vector



  # extract the interval index for the right boudanry



  # compute the length of the interval



  # sample sigma at the middle point



  # update the source_vector


  # return only the interior nodes
  return b[1:-1]


def solve_poisson_dirichelet(v_h, f, sigma, 
                             u_dirichlet=np.zeros((2)) ):
  """ function to solbe the Poisson equation with 
  Dirichlet boundary conditions
  input:  v_h         function space
          f           load (python function)
          sigma       conductivity
          u_dirichlet boundary conditions
  output: u           approximation (Function class)
  """  

  # we compute the stiffness matrix, we only use the  
  # the interior dof, and we need to convert it to 
  # a csc_matrix
  S = 
  
  # we build the source
  b = 

  # solve for the interior degrees of freedom
  u_interior = spsolve(S,b)

  # concatenate the solution to add the boundary 
  # conditions
  xi_u = np.concatenate([u_dirichlet[:1], 
                         u_interior, 
                         u_dirichlet[1:]])

  # return the function
  return Function(xi_u, v_h)


def pi_h(v_h, f):
  """interpolation function
    input:  v_h   function space
            f     function to project
    output: pih_f function that is the interpolation 
                  of f into v_h
  """
  pi_h_f = 


  return pi_h_f


def p_h(v_h, f):
  """projection function
    input:  v_h   function space
            f     function to project
    output: ph_f  function that is the projection 
                  of f into v_h
  """
  # compute load vector
  b = 

  # compute Mass matrix and convert it to csc type
  M = 

  # solve the system
  xi = spsolve(M,b)

  # create the new function (this needs to be an instance)
  # of a Function class
  ph_f = 

  return ph_f



if __name__ == "__main__":

  """ This is the main function, which will run 
  if you try to run this script, this code was provided 
  to you to help you debug your code. 
  """ 

  x = np.linspace(0,1,11)

  mesh = Mesh(x)
  v_h  = V_h(mesh)

  
  f_load = lambda x: 2+0*x
  xi = f_load(x) # linear function

  u = Function(xi, v_h) 

  assert np.abs(u(x[5]) - f_load(x[5])) < 1.e-6

  # check if this is projection
  ph_f = p_h(v_h, f_load)
  ph_f2 = p_h(v_h, ph_f)

  # 
  assert np.max(ph_f.xi - ph_f2.xi) < 1.e-6

  # using analytical solution
  u = lambda x : np.sin(4*np.pi*x)
  # building the correct source file
  f = lambda x : (4*np.pi)**2*np.sin(4*np.pi*x)
  # conductivity is constant
  sigma = lambda x : 1 + 0*x  

  u_sol = solve_poisson_dirichelet(v_h, f, sigma)

  err = lambda x: np.square(u_sol(x) - u(x))
  # we use an fearly accurate quadrature 
  l2_err = np.sqrt(integrate.quad(err, 0.0,1.)[0])

  print("L^2 error using %d points is %.6f"% (v_h.dim, l2_err))
  # this should be quite large

  # define a finer partition 
  x = np.linspace(0,1,21)
  # init mesh and fucntion space
  mesh = Mesh(x)
  v_h  = V_h(mesh)

  u_sol = solve_poisson_dirichelet(v_h, f, sigma)

  err = lambda x: np.square(u_sol(x) - u(x))
  # we use an fearly accurate quadrature 
  l2_err = np.sqrt(integrate.quad(err, 0.0,1.)[0])

  # print the error
  print("L^2 error using %d points is %.6f"% (v_h.dim, l2_err))






