from pyomo.contrib.incidence_analysis.interface import IncidenceGraphInterface
from pyomo.common.dependencies import scipy as sc
from pyomo.contrib.incidence_restructuring.graph_partitioning_algo import graph_partitioning_algorithm
from pyomo.contrib.incidence_restructuring.BBBD_general import BBBD_algo
from pyomo.common.collections import ComponentMap
from pyomo.util.subsystems import create_subsystem_block
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import numpy as np
import os
import sys 

# NOTE: IN ALL REFERENCES TO "ORDER" THE "ORDER" ARRAY [a_0, a_1, a_2, ... , a_n] MEANS
# THAT ROW/COLUMN a_i WILL BE MOVED TO ROW/COLUMN i   

# TO REORDER SPARSE MATRICES: 
# FOR COLUMN REORDERING: REPLACE ROWS OF IDENTITY MATRIX WITH COL_ORDER, RIGHT MULTIPLY MATRIX
# FOR ROW REORDERING: REPLACE COLS OF IDENTITY MATRIX WITH ROW_ORDER, LEFT MULTIPLY MATRIX
# THERE ARE OTHER EQUIVALENT METHODS, BUT WE STICK TO THIS CONVENTION FOR REORDERING

def get_incidence_matrix(m):
  igraph = IncidenceGraphInterface(m, include_inequality=False)
  incidence_matrix = igraph.incidence_matrix
  return igraph, incidence_matrix

def create_adj_list_from_matrix(matrix):
  return [(i,j) for i,j in zip(*matrix.nonzero())]

def reorder_sparse_matrix(m, n, row_order, col_order, target_matrix):
  permutation_matrix = sc.sparse.eye(m).tocoo()
  permutation_matrix.col = permutation_matrix.col[row_order]
  permuted_matrix = permutation_matrix.dot(target_matrix)
  permutation_matrix = sc.sparse.eye(n).tocoo()
  permutation_matrix.row = permutation_matrix.row[col_order]
  return permuted_matrix.dot(permutation_matrix)

def save_matrix_structure(matrix, params):
  fraction, num_blocks, size_border, size_system, index = params
  print("Image # = ", index)
  if index < 10:
    beginning = "00{}".format(index)
  else:
    beginning = "0{}".format(index)
  name_file = beginning + "_1D_HX.png"
  plt.figure(figsize=(12,8))
  plt.spy(matrix)
  plt.title("Fraction : {:.2f}  Num Blocks : {}  Size Border :  {}  Size System :  {}".format(fraction,
                                                                        num_blocks, size_border, size_system))
  plt.savefig("/nfs/home/strahlw/Documents/parallel_initialization_project/algorithm_hyperparameter_images2/"
           + name_file, dpi=300)
  plt.close()
  #plt.show()

def show_matrix_structure(matrix):
  plt.spy(matrix)
  plt.show()

def get_variable_constraint_maps(m, igraph):
  # variables = m.component_objects(pyo.Var)
  # constraints = m.component_objects(pyo.Constraint)
  index_variable_map = {igraph.get_matrix_coord(var) : var for var in igraph._variables}
  constraint_variable_map = {igraph.get_matrix_coord(constr) : constr for constr in igraph._constraints}
  return index_variable_map, constraint_variable_map

def get_restructured_matrix(model, method=1, fraction=0.9):
  # igraph = IncidenceGraphInterface(model, include_inequality=False)
  # incidence_matrix = igraph.incidence_matrix
  igraph, incidence_matrix = get_incidence_matrix(model)
  m, n = incidence_matrix.shape
  edge_list = create_adj_list_from_matrix(incidence_matrix)
  assert method == 1
  if method == 1:
    # no graph partitioning, just algorithmic
    bbbd_algo = BBBD_algo(edge_list, m, n, fraction)
    # col_order, row_order, blocks
    return *bbbd_algo.solve(), *get_variable_constraint_maps(model, igraph)
  
def show_decomposed_matrix(model, method=1, fraction=0.9):
  igraph, incidence_matrix = get_incidence_matrix(model)
  show_matrix_structure(incidence_matrix)
  col_order, row_order, blocks, idx_var_map, idx_constr_map = \
    get_restructured_matrix(model, method, 1.1)
  print(len(blocks))
  print([len(i[0]) for i in blocks])
  reordered_incidence_matrix = reorder_sparse_matrix(len(row_order),
     len(col_order), row_order, col_order, incidence_matrix)
  show_matrix_structure(reordered_incidence_matrix)

def save_algorithm_decomposition_images(model):
  igraph, incidence_matrix = get_incidence_matrix(model)
  for index, fraction in enumerate(np.linspace(0.0, 1.0, 100)):
    col_order, row_order, blocks, idx_var_map, idx_constr_map = \
      get_restructured_matrix(model, fraction=fraction)
    num_blocks = len(blocks)
    size_system = sum([len(blocks[i][0]) for i in range(len(blocks))])
    size_border = len(idx_var_map) - size_system
    reordered_incidence_matrix = reorder_sparse_matrix(len(row_order),
      len(col_order), row_order, col_order, incidence_matrix)
    save_matrix_structure(reordered_incidence_matrix, [fraction, num_blocks, size_border, size_system, index])


def filter_small_blocks(blocks):
  # only use blocks that are at least 10% of the largest block size
  # eliminates erratic behavior of solving very small systems
  return [i for i in blocks if len(i) >= 0.1*max(len(j) for j in blocks)]   

def create_subsystem_from_block(block, constr_map):
  # block is list of indices
  # what to do about inequalities???
  # get constraint names
  constr_obj = [constr_map[i] for i in block]
  # the subsystem function will automatically populate the variables
  return create_subsystem_block(constr_obj, include_fixed=True)

def create_subsystems(blocks, constr_map):
  return [create_subsystem_from_block(i, constr_map) for i in blocks]

def solve_subsystems_sequential(subsystems):
  solver_success = [True]*len(subsystems)
  solver = pyo.SolverFactory('ipopt')
  for idx, model in enumerate(subsystems):
    try:
      solver.solve(model, tee=True)
    except:
      solver_success[idx] = False
      pass
    
  return solver_success

def create_subfolder(folder_name):
  if not os.path.exists(folder_name):
    os.mkdir(folder_name)

def solve_subsystems_sequential_independent(subsystems, border_indices, idx_var_map, folder):
  solver_success = [True]*len(subsystems)
  solver = pyo.SolverFactory('ipopt')
  solver.options['max_iter'] = 300
  initial_val_border_vars = ComponentMap()
  # get original value
  for index in border_indices:
    var = idx_var_map[index]
   
    if var.value == None:
      initial_val_border_vars[var] = None
    else:
      initial_val_border_vars[var] = var.value  

    if var.name == "fs.heat_exchanger.cold_side.length":
      print(initial_val_border_vars[var])
      print(var.value)
      print(var.bounds)
    
  sys.exit(0)
  # solve the subsystems and reset the values of the border variables
  # this simulates a parallel solution strategy
  for idx, model in enumerate(subsystems):
    for var in model.component_data_objects(pyo.Var):
      if var.name == "fs.heat_exchanger.cold_side.length":
        print(var.value)
        sys.exit(0)
    # border variables will not have values loaded
    for var in initial_val_border_vars:
      var.value = initial_val_border_vars[var]
    try:
      solver.solve(model, logfile=os.path.join(folder,"block_{}_logfile.log".format(idx)))
    except:
      solver_success[idx] = False
      pass
  
  # reset border value after last solve
  for var in initial_val_border_vars:
    var.value = initial_val_border_vars[var]

  return solver_success

def BBBD_initializer_sequential(model, method=1, num_part=4, d_max=10, n_max=2, folder="Results"):
  create_subfolder(folder)

  
  order, blocks, col_map, method, idx_var_map, idx_constr_map, border_indices = get_restructured_matrix(model, method, num_part, 
                                                                    d_max, n_max)
  original_mapping_constr, original_mapping_vars = get_mappings_to_original(
    order, col_map, idx_var_map, idx_constr_map
  )
  blocks = filter_small_blocks(reformat_blocks(method, blocks))
  subsystems = create_subsystems(blocks, original_mapping_constr)

  solve_subsystems_sequential_independent(subsystems, border_indices, original_mapping_vars, folder)
  return 
# method = 1 --> BBBD
# method = 2 --> GP + LP
def plot_matrix_structures(m, method=1, num_part=4, d_max=10, n_max = 2):
  # first plot original matrix
  igraph, incidence_matrix = get_incidence_matrix(m)
  show_matrix_structure(incidence_matrix)
  
  # second, plot perfectly matched matrix
  permuted_matrix, col_map, idx_var_map, idx_constr_map = create_perfect_matching(igraph, incidence_matrix)
  show_matrix_structure(permuted_matrix)
  overall_adjacency = create_adj_list_from_matrix(permuted_matrix)
  # restructure the matrix
  if method == 1:
    column_order, all_blocks, border_indices = bbbd_algo(overall_adjacency, 
        get_adjacency_size(overall_adjacency), d_max, n_max)
    # print("COLUMN ORDER = ", column_order)
  
  if method == 2:
    column_order, partitions, border_incides = graph_partitioning_algorithm(num_part, 
                                                            overall_adjacency)
  # plot the final structure
  # restructured_matrix = reorder_matrix(column_order, column_order, permuted_matrix)
  # show_matrix_structure(restructured_matrix)

  restructured_matrix = reorder_sparse_matrix(len(column_order), column_order, column_order, permuted_matrix)
  show_matrix_structure(restructured_matrix)


  if method == 1:
      print("Number of partitions = ", len(all_blocks))
      print("Size of partitions = ", [all_blocks[i].size for i in all_blocks])
  if method == 2:
      print("Number of partitions = ", num_part)
      print("Size of partitions = ", [len(partitions[i]) for i in partitions])
  

