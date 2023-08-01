from pyomo.contrib.incidence_analysis.interface import IncidenceGraphInterface
from pyomo.common.dependencies import scipy as sc
from pyomo.contrib.incidence_restructuring.graph_partitioning_algo import (
    graph_partitioning_algorithm_general, graph_partitioning_algorithm
)
from pyomo.contrib.incidence_restructuring.BBBD_general import BBBD_algo
from pyomo.contrib.incidence_restructuring.BBBD_algorithm import bbbd_algo
from pyomo.common.collections import ComponentMap
from pyomo.util.subsystems import create_subsystem_block
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.core.base.var import IndexedVar
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import numpy as np
import os
import sys 
import random 
random.seed(8900)
###############
### PERFECT MATCHING FUNCTIONS
##############
def create_adj_list_from_matrix(permuted_matrix):
  # x_adjacency = {i : [] for i in range(permuted_matrix.shape[1])}
  # y_adjacency = {i : [] for i in range(permuted_matrix.shape[0])}
  overall_adjacency = {i : [] for i in range(permuted_matrix.shape[0])}
  for i, j in zip(*permuted_matrix.nonzero()):
    if i not in overall_adjacency[j]:
      overall_adjacency[j].append(i)
    if i != j and j not in overall_adjacency[i]:
      overall_adjacency[i].append(j)
  #overall_adjacency = {i : [j for j in range(max(permuted_matrix.shape[1], permuted_matrix.shape[0])) if (j in x_adjacency[i] or j in y_adjacency[i])] for i in x_adjacency} 
  return overall_adjacency

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

# needed for bbbd algorithm initialization
def get_adjacency_size(adjacency_list):
  return {i : len(adjacency_list[i]) for i in adjacency_list}

def get_restructured_matrix_matched(incidence_matrix, igraph, folder, method=1, num_part=4, d_max=10, n_max = 2):
  matched_matrix, col_map, idx_var_map, idx_constr_map = create_perfect_matching(igraph, incidence_matrix)
  overall_adjacency = create_adj_list_from_matrix(matched_matrix)
  assert method == 1 or method == 2
  # restructure the matrix
  if method == 1:
    column_order, all_blocks, border_indices = bbbd_algo(overall_adjacency, 
        get_adjacency_size(overall_adjacency), d_max, n_max)
    
    return column_order, column_order, reformat_blocks(method, all_blocks), \
      col_map, idx_var_map, idx_constr_map
  
  if method == 2:
    column_order, partitions, border_indices = graph_partitioning_algorithm(num_part, 
                                                      overall_adjacency, folder)
    
    return column_order, column_order, reformat_blocks(method, partitions), \
      col_map, idx_var_map, idx_constr_map
  
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

def reformat_blocks(method, blocks):
  if method == 1:
    # BBBD algorithm
    return [[[key for key in blocks[i].vertices]]*2 for i in blocks]
  if method == 2:
    # GP algorithm
    return [[blocks[key]]*2 for key in blocks]

###############
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

def create_edge_list_from_matrix(matrix):
  return [(i,j) for i,j in zip(*matrix.nonzero())]

def create_adj_list_from_edge_list(edge_list, num_vars, num_constr):
  adj_list = [[] for i in range(num_vars + num_constr)]
  for edge in edge_list:
    adj_list[edge[0]].append(edge[1]+num_vars)
    adj_list[edge[1]+num_vars].append(edge[0])
  return adj_list

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
  plt.title("Num Part : {:.2f}  Num Blocks : {}  Size Border :  {}  Size System :  {}".format(fraction,
                                                                        num_blocks, size_border, size_system))
  plt.savefig("/nfs/home/strahlw/Documents/parallel_initialization_project/algorithm_hyperparameter_images3/"
           + name_file, dpi=300)
  plt.close()
  #plt.show()

def show_matrix_structure(matrix):
  plt.spy(matrix)
  plt.show()

def get_variable_constraint_maps(igraph):
  # variables = m.component_objects(pyo.Var)
  # constraints = m.component_objects(pyo.Constraint)
  index_variable_map = {igraph.get_matrix_coord(var) : var for var in igraph._variables}
  constraint_variable_map = {igraph.get_matrix_coord(constr) : constr for constr in igraph._constraints}
  return index_variable_map, constraint_variable_map

def get_restructured_matrix(incidence_matrix, igraph, model, folder, 
    method=1, fraction=0.9, num_part=4):
  # igraph = IncidenceGraphInterface(model, include_inequality=False)
  # incidence_matrix = igraph.incidence_matrix
  m, n = incidence_matrix.shape
  edge_list = create_edge_list_from_matrix(incidence_matrix)
  assert method == 1 or method == 2
  if method == 1:
    # no graph partitioning, just algorithmic
    bbbd_algo = BBBD_algo(edge_list, m, n, fraction)
    # col_order, row_order, blocks
    return *bbbd_algo.solve(), *get_variable_constraint_maps(igraph)

  if method == 2:
    adjacency_list = create_adj_list_from_edge_list(edge_list, n, m)
    return *graph_partitioning_algorithm_general(num_part, edge_list, adjacency_list, n, m, folder),\
        *get_variable_constraint_maps(igraph)

def get_restructured_matrix_general(incidence_matrix, igraph, model, folder, 
    method=1, fraction=0.9, num_part=4, matched=False, d_max=2, n_max=1):
  # wrapper for the matched version and non-matched versions of the algorithm
  if not matched:
    return get_restructured_matrix(incidence_matrix, igraph, model, folder, 
    method, fraction, num_part)
  else: # not matched 
    # need to return col order, row order, blocks, idx_var_map, idx_constr_map
    col_order, row_order, blocks, col_map, idx_var_map, idx_constr_map = \
      get_restructured_matrix_matched(incidence_matrix, igraph, folder, method=method, 
          num_part=num_part, d_max=d_max, n_max=n_max)
    idx_constr_map, idx_var_map = get_mappings_to_original(col_order, col_map, idx_var_map, idx_constr_map)
    return col_order, row_order, blocks, idx_var_map, idx_constr_map

def get_complicating_variables(blocks, col_order):
  size_blocks = sum(len(i[0] for i in blocks))
  return col_order[size_blocks:]

def get_complicating_constraints(blocks, row_order):
  # assertion for square blocks
  assert all([len(i[0]) == len(i[1]) for i in blocks])
  size_blocks = sum(len(i[0] for i in blocks))
  return row_order[size_blocks:]
  
def show_decomposed_matrix(model, method=1, fraction=0.9, num_part=4):
  igraph, incidence_matrix = get_incidence_matrix(model)
  show_matrix_structure(incidence_matrix)
  col_order, row_order, blocks, idx_var_map, idx_constr_map = \
    get_restructured_matrix(incidence_matrix, igraph, model, "test_problem", method, fraction, num_part)
  print(len(blocks))
  print([len(i[0]) for i in blocks])
  # print([len(i[1]) for i in blocks])
  reordered_incidence_matrix = reorder_sparse_matrix(len(row_order),
     len(col_order), row_order, col_order, incidence_matrix)
  show_matrix_structure(reordered_incidence_matrix)

def save_algorithm_decomposition_images(model):
  igraph, incidence_matrix = get_incidence_matrix(model)
  for index, fraction in enumerate(np.linspace(0.0, 1.0, 100)):
    col_order, row_order, blocks, idx_var_map, idx_constr_map = \
      get_restructured_matrix(incidence_matrix, igraph, model, fraction=fraction)
    num_blocks = len(blocks)
    size_system = sum([len(blocks[i][0]) for i in range(len(blocks))])
    size_border = len(idx_var_map) - size_system
    reordered_incidence_matrix = reorder_sparse_matrix(len(row_order),
      len(col_order), row_order, col_order, incidence_matrix)
    save_matrix_structure(reordered_incidence_matrix, [fraction, num_blocks, size_border, size_system, index])

def save_algorithm_decomposition_images_gp(model):
  igraph, incidence_matrix = get_incidence_matrix(model)
  for index, num_part in enumerate(range(1,31)):
    col_order, row_order, blocks, idx_var_map, idx_constr_map = \
      get_restructured_matrix(incidence_matrix, igraph, model, method=2, num_part=num_part)
    num_blocks = len(blocks)
    size_system = sum([len(blocks[i][0]) for i in range(len(blocks))])
    size_border = len(idx_var_map) - size_system
    reordered_incidence_matrix = reorder_sparse_matrix(len(row_order),
      len(col_order), row_order, col_order, incidence_matrix)
    save_matrix_structure(reordered_incidence_matrix, [num_part, num_blocks, size_border, size_system, index])


# def filter_small_blocks(blocks):
#   # only use blocks that are at least 10% of the largest block size
#   # eliminates erratic behavior of solving very small systems
#   return [i for i in blocks if len(i) >= 0.1*max(len(j) for j in blocks)]   

def create_subsystem_from_constr_list(constr_idx_list, constr_map):
  # get constraint names
  constr_obj = [constr_map[i] for i in constr_idx_list]
  block = create_subsystem_block(constr_obj, include_fixed=True)
  block.obj = pyo.Objective(expr = create_residual_objective_expression(constr_obj))
  for constr in constr_obj:
    constr.deactivate()
  return block

def create_subsystem_from_constr_list_no_obj(constr_idx_list, constr_map):
  constr_obj = [constr_map[i] for i in constr_idx_list]
  block = create_subsystem_block(constr_obj, include_fixed=True)
  return block

def create_subsystem_from_block(block, constr_map):
  # block is list of indices
  # what to do about inequalities???
  # get constraint names
  constr_obj = [constr_map[i] for i in block]
  # the subsystem function will automatically populate the variables
  return create_subsystem_block(constr_obj, include_fixed=True)

def create_subsystems_block(blocks, constr_map):
  return [create_subsystem_from_block(j, constr_map) for i,j in blocks]


def create_subsystems(constr_lists, constr_map, no_objective=True):
  if no_objective:
    return [create_subsystem_from_constr_list_no_obj(i, constr_map) for i in constr_lists]
  return [create_subsystem_from_constr_list(i, constr_map) for i in constr_lists]

def solve_subsystems_sequential(subsystems, folder):
  solver_success = [True]*len(subsystems)
  for idx, model in enumerate(subsystems):
    solver_success[idx] = solve_subsystem(model, folder, idx)
  return solver_success

def solve_subsystem(subsystem, folder, id):
  solver = pyo.SolverFactory('ipopt')
  # try:
  solver.options['max_iter'] = 300
  solver.solve(subsystem, logfile=os.path.join(folder,"block_{}_logfile.log".format(id)))
  # except:
  #   print("Solver ERROR")
  #   return False
  return True


def uniform_random_heuristic(var):
  # if the variable is doubly bounded: set at midpoint + Δ,
  # if the variable has a single lower bound: set at bound + Δ,
  # if the variable has a single upper bound: set at bound − Δ,
  # if the variable is unbounded in both directions: set at zero + Δ,
  perturbation = random.random()
  if var.ub != None and var.lb != None:
    if abs(var.ub - var.lb) < 1e-6:
      return var.ub
    avg = (var.ub + var.lb)/2 
    perturbation = random.uniform(0, min(var.ub - avg, 1.0))
    return avg + perturbation
  if var.ub == None and var.lb == None:
    return 0 + perturbation 
  if var.lb == None:
    # has upper bound
    return var.ub - perturbation
  if var.ub == None:
    return var.lb + perturbation 
  assert False

def get_initial_values_guesses(model, incidenceGraph):
  initial_bounds = ComponentMap()
  for var in incidenceGraph.variables:
    initial_bounds[var] = var.bounds
  fbbt(model, feasibility_tol = 1e-4)
  initial_vals = ComponentMap()
  for var in incidenceGraph.variables:
    initial_val = uniform_random_heuristic(var)
    initial_vals[var] = initial_val
  for var in incidenceGraph.variables:
    var.bounds = initial_bounds[var]
  return initial_vals

def create_subfolder(folder_name):
  if not os.path.exists(folder_name):
    os.mkdir(folder_name)
  
def get_list_of_simple_variables(blocks, idx_var_map):
  simple_variables = []
  for vars, constraints in blocks:
    for var_index in vars:
      simple_variables.append(idx_var_map[var_index])
  return simple_variables 

def get_list_of_complicating_variables(blocks, idx_var_map):
  complicating_var_idx = [i for i in range(len(idx_var_map)) if i not in [j for system in blocks for j in system[0]]]
  return [idx_var_map[i] for i in complicating_var_idx]

def get_list_of_complicating_constraints(blocks, idx_constr_map):
  complicating_constr_idx = [i for i in range(len(idx_constr_map)) if i not in [j for system in blocks for j in system[1]]]
  return [idx_constr_map[i] for i in complicating_constr_idx]

def get_list_complicating_constraint_indices(blocks, idx_constr_map):
  return [i for i in range(len(idx_constr_map)) if i not in [j for system in blocks for j in system[1]]]

def fix_variables(vars):
  for var in vars:
    var.fix()

def unfix_variables(vars):
  for var in vars:
    var.unfix()

def convert_constr_to_residual_form(constraint):
  assert constraint.upper == constraint.lower
  return constraint.body - constraint.upper

def create_residual_objective_expression(list_constraints):
  return sum(convert_constr_to_residual_form(constr)**2 for constr in list_constraints)
  
def phase_I(complicating_variables, simple_variables, subsystems, folder, iteration):
  # assume all the variables are unfixed
  unfix_variables(simple_variables)
  assert all(i.fixed == False for i in complicating_variables)
  assert all(i.fixed == False for i in simple_variables)
  folder += "_Phase_I_{}".format(iteration)
  create_subfolder(folder)
  fix_variables(complicating_variables)
  # solve the subsystems and keep track of solver status
  return solve_subsystems_sequential(subsystems, folder)

def get_normalized_change_var(old_val, new_val):
  if old_val == new_val:
    return 0
  if old_val == 0:
    return abs(new_val)
  # measures the normalized change of each variable
  return abs(new_val - old_val) / abs(old_val)

def get_list_normalized_change(list_vars, list_old):
  return [get_normalized_change_var(list_old[list_vars[i]], list_vars[i].value) for i in range(len(list_old))]

def phase_II(complicating_constr_subsystem, complicating_vars, complicating_constr, simple_vars, folder):
  # unfix the complicating variables
  unfix_variables(complicating_vars)
  fix_variables(simple_vars)
  return solve_subsystem(complicating_constr_subsystem, folder, "complicating_constr")

def initialization_strategy(model, folder, method=2, num_part=4, fraction=0.5, matched=False, d_max=1, n_max=2):
  create_subfolder(folder)
  folder = os.path.join(folder, "Results")
  create_subfolder(folder)
  igraph, incidence_matrix = get_incidence_matrix(model)
  col_order, row_order, blocks, idx_var_map, idx_constr_map = \
    get_restructured_matrix_general(incidence_matrix, igraph, model, folder, method, fraction, num_part, matched, 
      d_max, n_max)
  constr_list = [i[1] for i in blocks]
  complicating_constr = get_list_of_complicating_constraints(blocks, idx_constr_map)
  subsystems = create_subsystems(constr_list, idx_constr_map, no_objective=True)
  complicating_constr_subsystem = create_subsystem_from_constr_list_no_obj(
    get_list_complicating_constraint_indices(blocks, idx_constr_map), idx_constr_map)
  complicating_constr_subsystem.obj = pyo.Objective(expr = create_residual_objective_expression([idx_constr_map[i] for i in idx_constr_map]))
  # # deactivate the linking constraints, this will not affect the system solves
  # for constr in complicating_constr_subsystem.component_objects(pyo.Constraint):
  #   constr.deactivate()
  complicating_vars = get_list_of_complicating_variables(blocks, idx_var_map)
  simple_vars = get_list_of_simple_variables(blocks, idx_var_map)
  initial_vals = get_initial_values_guesses(model, igraph)
  old_vals = ComponentMap()
  for var in initial_vals:
    # var.value = initial_vals[var]
    old_vals[var] = initial_vals[var]
  
  iteration = 0
  maximum_change = 1
  while True:
    if iteration >= 5:
      print("Reached maximum iteration limit")
      break
    if iteration >= 1:
      print("Iteration = ", iteration)
      #print(get_list_normalized_change([var for var in initial_vals], old_vals))
      maximum_change = max(get_list_normalized_change([var for var in initial_vals], old_vals))
      
      print("Maximum change = ", maximum_change)
                           
    tolerance = 1e-2 # change by 1%
    for var in initial_vals:
      old_vals[var] = float(var.value)
    if maximum_change < tolerance:
      break

    phase_I(complicating_vars, simple_vars, subsystems, folder, iteration)
    #TODO handle unsuccessful solves - what do we do? start at point of 
    # local infeasibility?
    # print("Vals after Phase I")
    # for var in initial_vals:
    #   print(var, var.value)

    phase_II(complicating_constr_subsystem, complicating_vars, complicating_constr, simple_vars, folder)
    # print("Vals after Phase II")
    # for var in initial_vals:
    #   print(var, var.value)
    
    iteration += 1
  # for subsystem in subsystems:
  #   subsystem.pprint()
  # sys.exit(0)
  print("number of iterations = ", iteration)
  print("final max change = ", maximum_change)
  unfix_variables(simple_vars)
  for idx in idx_constr_map:
    idx_constr_map[idx].activate()
  # print("final_vals")
  # for var in initial_vals:
  #   print(var, var.value)

  # iteration = 1
  # print(phase_I(complicating_vars, simple_vars, subsystems, folder, iteration))
  # #TODO - handle unsuccessful solves
  # print("Vals after Phase I")
  # for var in initial_vals:
  #   print(var, var.value)

  # phase_II(complicating_constr_subsystem, complicating_vars, complicating_constr, simple_vars, folder)
  # print("Vals after Phase II")
  # for var in initial_vals:
  #   print(var, var.value)

  # iteration = 2
  # print(phase_I(complicating_vars, simple_vars, subsystems, folder, iteration))
  # # until termination criteria - just do one iteration for now
  # #phase_II(complicating_constr_subsystem, complicating_vars)
  # print("Vals after Phase I iteration 2")
  # for var in initial_vals:
  #   print(var, var.value)

  # phase_II(complicating_constr_subsystem, complicating_vars, complicating_constr, simple_vars, folder)
  # print("Vals after Phase II iteration 2")
  # for var in initial_vals:
  #   print(var, var.value)


# m = pyo.ConcreteModel()
# m.x1 = pyo.Var(name="x1", bounds=(0,1))
# m.x2 = pyo.Var(name="x2", bounds=(0,1))
# m.x3 = pyo.Var(name="x3", bounds=(0,1))
# m.x4 = pyo.Var(name="x4", bounds=(0,1))
# m.x5 = pyo.Var(name="x5", bounds=(0,1))
# m.x6 = pyo.Var(name="x6", bounds=(0,1))

# m.cons1 = pyo.Constraint(expr=m.x2 + m.x4  == 1.5 - m.x5)
# m.cons2 = pyo.Constraint(expr=m.x1 + m.x3 == 1)
# m.cons3 = pyo.Constraint(expr=m.x3 == 0.5)
# m.cons4 = pyo.Constraint(expr=m.x2 == 0.5)
# m.cons5 = pyo.Constraint(expr=m.x1 + m.x3 == 1.5 - m.x5)
# m.cons6 = pyo.Constraint(expr=m.x2 + m.x4 + m.x6 == 1.5)

# initialization_strategy(m, method=1, num_part=2, fraction=0.6, matched=True, d_max=1, n_max=2)
# sys.exit(0)











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
 
# perfect matching code
# method = 1 --> BBBD
# method = 2 --> GP + LP
# def plot_matrix_structures(m, method=1, num_part=4, d_max=10, n_max = 2):
#   # first plot original matrix
#   igraph, incidence_matrix = get_incidence_matrix(m)
#   show_matrix_structure(incidence_matrix)
  
#   # second, plot perfectly matched matrix
#   permuted_matrix, col_map, idx_var_map, idx_constr_map = create_perfect_matching(igraph, incidence_matrix)
#   show_matrix_structure(permuted_matrix)
#   overall_adjacency = create_edge_list_from_matrix(permuted_matrix)
#   # restructure the matrix
#   if method == 1:
#     column_order, all_blocks, border_indices = bbbd_algo(overall_adjacency, 
#         get_adjacency_size(overall_adjacency), d_max, n_max)
#     # print("COLUMN ORDER = ", column_order)
  
#   if method == 2:
#     column_order, partitions, border_incides = graph_partitioning_algorithm(num_part, 
#                                                             overall_adjacency)
#   # plot the final structure
#   # restructured_matrix = reorder_matrix(column_order, column_order, permuted_matrix)
#   # show_matrix_structure(restructured_matrix)

#   restructured_matrix = reorder_sparse_matrix(len(column_order), column_order, column_order, permuted_matrix)
#   show_matrix_structure(restructured_matrix)


#   if method == 1:
#       print("Number of partitions = ", len(all_blocks))
#       print("Size of partitions = ", [all_blocks[i].size for i in all_blocks])
#   if method == 2:
#       print("Number of partitions = ", num_part)
#       print("Size of partitions = ", [len(partitions[i]) for i in partitions])
  

