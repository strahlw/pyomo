from pyomo.contrib.incidence_analysis.interface import IncidenceGraphInterface
from pyomo.common.dependencies import scipy as sc
from pyomo.contrib.incidence_restructuring.graph_partitioning_algo import graph_partitioning_algorithm
from pyomo.contrib.incidence_restructuring.BBBD_algorithm import bbbd_algo
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

def create_adj_list_from_matrix(permuted_matrix):
  return [(i,j) for i,j in zip(*permuted_matrix.nonzero())]

def create_perfect_matching(igraph, incidence_matrix):
  # get perfect matching and maps to variables
  perfect_matching = igraph.maximum_matching()
  index_variable_map = {igraph.get_matrix_coord(perfect_matching[constraint]) : 
                        perfect_matching[constraint] for constraint in perfect_matching}
  index_constraint_map = {igraph.get_matrix_coord(constraint) : 
                             constraint for constraint in perfect_matching}
  # reorder the columns
  col_order = [0]*incidence_matrix.shape[0]
  for constraint in perfect_matching:
    col_order[igraph.get_matrix_coord(constraint)] = \
     igraph.get_matrix_coord(perfect_matching[constraint])
  # # keep track of reordering
  # col_map = {igraph.get_matrix_coord(constraint) : igraph.get_matrix_coord(
  #     perfect_matching[constraint]) for constraint in perfect_matching}
  # matrix operations to reorder
  permutation_matrix = sc.sparse.eye(incidence_matrix.shape[1]).tocoo()
  permutation_matrix.row = permutation_matrix.row[col_order]
  permuted_matrix = incidence_matrix.dot(permutation_matrix)
  return permuted_matrix, col_order, index_variable_map, index_constraint_map

def get_adjacency_and_map_pyomo_model(m):
  # get igraph object and incidence matrix for model
  igraph, incidence_matrix = get_incidence_matrix(m)
  # reorganize the matrix to have perfect matching
  permuted_matrix, col_map, idx_var_map, idx_constr_map = create_perfect_matching(igraph, incidence_matrix)
  # create the adjacency list
  overall_adjacency = create_adj_list_from_matrix(permuted_matrix)
  return overall_adjacency, col_map, permuted_matrix, idx_var_map, idx_constr_map


def reorder_sparse_matrix(size, row_order, col_order, target_matrix):
  permutation_matrix = sc.sparse.eye(size).tocoo()
  permutation_matrix.col = permutation_matrix.col[row_order]
  permuted_matrix = permutation_matrix.dot(target_matrix)
  permutation_matrix = sc.sparse.eye(size).tocoo()
  permutation_matrix.row = permutation_matrix.row[col_order]
  return permuted_matrix.dot(permutation_matrix)

def show_matrix_structure(matrix):
  plt.spy(matrix)
  plt.show()


def get_restructured_matrix(m, method=1, num_part=4, d_max=10, n_max = 2):
  igraph, incidence_matrix = get_incidence_matrix(m)
  matched_matrix, col_map, idx_var_map, idx_constr_map = create_perfect_matching(igraph, incidence_matrix)
  overall_adjacency = create_adj_list_from_matrix(matched_matrix)

  # restructure the matrix
  if method == 1:
    column_order, all_blocks, border_indices = bbbd_algo(overall_adjacency, 
        get_adjacency_size(overall_adjacency), d_max, n_max)
    
    return column_order, all_blocks, col_map, method, idx_var_map, idx_constr_map, border_indices
  
  if method == 2:
    column_order, partitions, border_indices = graph_partitioning_algorithm(num_part, 
                                                      overall_adjacency)
    
    return column_order, partitions, col_map, method, idx_var_map, idx_constr_map, border_indices
  
def invert_order(order, array1):
  return [order.index(elem) for elem in array1]

def get_perfect_matched_cols_rows(order):
  # indices in final restructuring
  row_2 = range(len(order))
  # indices from final restructuring to perfect matching
  row_1 = invert_order(order, row_2)
  col_1 = invert_order(order, row_2)
  return row_1, col_1

def update_keys(dictionary, array):
  # returns a dictionary with updated keys
  assert len(array) == len(dictionary)
  return {i : dictionary[array[i]] for i in range(len(array))}

def get_perfect_match_mapping(idx_var_map, perfect_matching_order):
  # column update
  return update_keys(idx_var_map, perfect_matching_order)

def get_restructured_mapping(idx_var_map, idx_constr_map, bbbd_order):
  # return constr, then var (row then column)
  return update_keys(idx_constr_map, bbbd_order), update_keys(idx_var_map, bbbd_order)

def get_mappings_to_original(bbbd_order, perfect_matching_order, idx_var_map, idx_constr_map):
  # create a mapping from final indices to original variables/constraints
  # start with original indices and update based on perfect matching
  perfect_match_mapping_cols = get_perfect_match_mapping(idx_var_map, perfect_matching_order)
  return idx_constr_map, perfect_match_mapping_cols
  # turns out the final matrix is only restructured for visualization, so we just need perfect matching mapping
  #return get_restructured_mapping(perfect_match_mapping_cols, idx_constr_map, bbbd_order)

def reformat_blocks(method, blocks):
  if method == 1:
    # BBBD algorithm
    return [[key for key in blocks[i].vertices] for i in blocks]
  if method == 2:
    # GP algorithm
    return [blocks[key] for key in blocks]

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
  

