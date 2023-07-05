from pyomo.contrib.incidence_analysis.interface import IncidenceGraphInterface
from pyomo.common.dependencies import scipy as sc
from pyomo.contrib.incidence_restructuring.graph_partitioning_algo import graph_partitioning_algorithm
from pyomo.contrib.incidence_restructuring.BBBD_algorithm import bbbd_algo
import matplotlib.pyplot as plt

def get_incidence_matrix(m):
  igraph = IncidenceGraphInterface(m, include_inequality=False)
  incidence_matrix = igraph.incidence_matrix
  return igraph, incidence_matrix

def create_adj_list_from_matrix(permuted_matrix):
  x_adjacency = {i : [] for i in range(permuted_matrix.shape[1])}
  y_adjacency = {i : [] for i in range(permuted_matrix.shape[0])}
  for i, j in zip(*permuted_matrix.nonzero()):
    x_adjacency[j].append(i)
    y_adjacency[i].append(j)
  overall_adjacency = {i : [j for j in range(max(permuted_matrix.shape[1], permuted_matrix.shape[0])) if (j in x_adjacency[i] or j in y_adjacency[i])] for i in x_adjacency} 
  return overall_adjacency

def create_perfect_matching(igraph, incidence_matrix):
  perfect_matching = igraph.maximum_matching()
  col_order = [0]*incidence_matrix.shape[0]
  for constraint in perfect_matching:
    col_order[igraph.get_matrix_coord(perfect_matching[constraint])] = \
     igraph.get_matrix_coord(constraint)
  col_map = {igraph.get_matrix_coord(constraint) : igraph.get_matrix_coord(
      perfect_matching[constraint]) for constraint in perfect_matching}
  permutation_matrix = sc.sparse.eye(incidence_matrix.shape[1]).tocoo()
  permutation_matrix.col = permutation_matrix.col[col_order]
  permuted_matrix = incidence_matrix.dot(permutation_matrix)
  return permuted_matrix, col_map

def get_adjacency_and_map_pyomo_model(m):
  # get igraph object and incidence matrix for model
  igraph, incidence_matrix = get_incidence_matrix(m)
  # reorganize the matrix to have perfect matching
  permuted_matrix, col_map = create_perfect_matching(igraph, incidence_matrix)
  # create the adjacency list
  overall_adjacency = create_adj_list_from_matrix(permuted_matrix)
  return overall_adjacency, col_map, permuted_matrix

# needed for bbbd algorithm initialization
def get_adjacency_size(adjacency_list):
  return {i : len(adjacency_list[i]) for i in adjacency_list}

def reorder_matrix(column_order, matrix):
  rearranged_matrix = matrix[:, column_order]
  rearranged_matrix = rearranged_matrix[column_order, :]
  return rearranged_matrix

def show_matrix_structure(matrix):
  plt.spy(matrix)
  plt.show()

# method = 1 --> BBBD
# method = 2 --> GP + LP
def plot_matrix_structures(m, method=1, num_part=4, d_max=10, n_max = 2):
  # first plot original matrix
  igraph, incidence_matrix = get_incidence_matrix(m)
  show_matrix_structure(incidence_matrix)
  
  # second, plot perfectly matched matrix
  permuted_matrix, col_map = create_perfect_matching(igraph, incidence_matrix)
  show_matrix_structure(permuted_matrix)
  overall_adjacency = create_adj_list_from_matrix(permuted_matrix)

  # restructure the matrix
  if method == 1:
    column_order, all_blocks = bbbd_algo(overall_adjacency, 
        get_adjacency_size(overall_adjacency), d_max, n_max)
  
  if method == 2:
    column_order, partitions = graph_partitioning_algorithm(num_part, 
                                                            overall_adjacency)
  # plot the final structure
  show_matrix_structure(reorder_matrix(column_order, permuted_matrix))

  if method == 1:
      print("Number of partitions = ", len(all_blocks))
      print("Size of partitions = ", [all_blocks[i].size for i in all_blocks])
  if method == 2:
      print("Number of partitions = ", num_part)
      print("Size of partitions = ", [len(partitions[i]) for i in partitions])


