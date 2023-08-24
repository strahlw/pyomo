from pyomo.contrib.incidence_analysis.interface import IncidenceGraphInterface
from pyomo.common.dependencies import scipy as sc
from pyomo.contrib.incidence_restructuring.graph_partitioning_algo import (
    graph_partitioning_algorithm_general, graph_partitioning_algorithm
)
from pyomo.core.expr import differentiate
from pyomo.core.expr.current import identify_variables
from pyomo.contrib.incidence_restructuring.BBBD_general import BBBD_algo
from pyomo.contrib.incidence_restructuring.BBBD_algorithm import bbbd_algo
from pyomo.common.collections import ComponentMap
from pyomo.util.subsystems import create_subsystem_block
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.opt.results.solver import check_optimal_termination, TerminationCondition, SolverStatus
# comp,
#     deactivate_satisfied_constraints=False,
#     integer_tol=1e-5,
#     feasibility_tol=1e-8,
#     max_iter=10,
#     improvement_tol=1e-4,
#     descend_into=True,
from pyomo.core.base.var import IndexedVar
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import numpy as np
import os
import sys 
import math
import random 
random.seed(8900)

class algorithmConfiguration(object):
  def __init__(self):
    # objective 0 -> feasibility (i.e. obj = 0)
    # objective 1 -> min sum of squared violations 
    self.phase_I = {"subproblem_objective" : 0, "nonlinear_program" : 1}
    # constraints 0 -> just the complicating constraints
    # constraints 1 -> all the constraints (only applicable to minimizing violations)
    self.phase_II = {"subproblem_objective" : 0, "constraints" : 0, "nonlinear_program" : 1}
    self.config = 1
    self.constraint_consensus = False

  def set_configuration(self, config):
    self.config = config
    # configuration 1 - default construction 
    # config 2 - systems minimize squared violations
    # config 3 - complicating constraint objective is minimizing squared violations
    # config 4 - both problems' objectives are minimizing squared violations
    if config in [2,4,6,8]:
      self.phase_I["subproblem_objective"] = 1
    if config in [3,4,5,6]:
      self.phase_II["subproblem_objective"] = 1
    if config in [5,6]:
      self.phase_II["constraints"] = 1
    if config in [7,8]:
      self.constraint_consensus = True
      self.phase_II["nonlinear_program"] = False
    return
  
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

def get_restructured_matrix_matched(incidence_matrix, igraph, folder, method=1, num_part=4, d_max=2, n_max=1, border_fraction=0.5, show=False):
  matched_matrix, col_map, idx_var_map, idx_constr_map = create_perfect_matching(igraph, incidence_matrix)
  overall_adjacency = create_adj_list_from_matrix(matched_matrix)
  assert method == 1 or method == 2
  # restructure the matrix
  if method == 1:
    column_order, all_blocks, border_indices = bbbd_algo(overall_adjacency, 
        get_adjacency_size(overall_adjacency), d_max, n_max, border_fraction)

    if show:
      display_reordered_matrix(column_order, column_order, matched_matrix)
    
    return column_order, column_order, reformat_blocks(method, all_blocks), \
      col_map, idx_var_map, idx_constr_map
  
  if method == 2:
    column_order, partitions, border_indices = graph_partitioning_algorithm(num_part, 
                                                      overall_adjacency, folder)
    #print(column_order, num_part)
    if show:
      display_reordered_matrix(column_order, column_order, matched_matrix)
   
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

# note: for the matched matrices, the component maps mapping the indices to the variables and constraints
# are to the reordering for the MATCHED matrix, which the blocks reference.
# to display the reordered matrix structure, the column and row order need to be used with the MATCHED matrix
# The implication of this is that the row order and column order SHOULD ONLY BE USED FOR DISPLAYING RESTRUCTURED MATRICES
# and not for identifying complicating variables and/or constraints.
def get_restructured_matrix_general(incidence_matrix, igraph, model, folder, 
    method=1, fraction=0.9, num_part=4, matched=False, d_max=2, n_max=1, border_fraction=0.5):
  # wrapper for the matched version and non-matched versions of the algorithm
  if not matched:
    return get_restructured_matrix(incidence_matrix, igraph, model, folder, 
    method, fraction, num_part)
  else: # matched 
    # need to return col order, row order, blocks, idx_var_map, idx_constr_map
    col_order, row_order, blocks, col_map, idx_var_map, idx_constr_map = \
      get_restructured_matrix_matched(incidence_matrix, igraph, folder, method=method, 
          num_part=num_part, d_max=d_max, n_max=n_max, border_fraction=border_fraction)
    idx_constr_map, idx_var_map = get_mappings_to_original(col_order, col_map, idx_var_map, idx_constr_map)
    return col_order, row_order, blocks, idx_var_map, idx_constr_map

def show_decomposed_matrix(model, method=1, fraction=0.9, num_part=4):
  igraph, incidence_matrix = get_incidence_matrix(model)
  show_matrix_structure(incidence_matrix)
  col_order, row_order, blocks, idx_var_map, idx_constr_map = \
    get_restructured_matrix_matched(incidence_matrix, igraph, "test_problem", method, border_fraction=fraction, num_part=num_part)
  print(len(blocks))
  print([len(i[0]) for i in blocks])
  # print([len(i[1]) for i in blocks])
  reordered_incidence_matrix = reorder_sparse_matrix(len(row_order),
     len(col_order), row_order, col_order, incidence_matrix)
  show_matrix_structure(reordered_incidence_matrix)

def display_reordered_matrix(row_order, col_order, incidence_matrix):
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

def create_subsystem_from_constr_list(model, constr_idx_list, constr_map, name):
  # get constraint names
  constr_obj = [constr_map[i] for i in constr_idx_list]
  model.add_component(name, create_subsystem_block(constr_obj))
  # model.find_component(name).obj = pyo.Objective(expr = 1/1e10 *create_residual_objective_expression(constr_obj))
  model.find_component(name).obj = pyo.Objective(expr = create_residual_objective_expression(constr_obj))
  for constr in constr_obj:
    constr.deactivate()
  return

def create_subsystem_from_constr_list_no_obj(model, constr_idx_list, constr_map, name):
  constr_obj = [constr_map[i] for i in constr_idx_list]
  model.add_component(name, create_subsystem_block(constr_obj))
  model.find_component(name).obj = pyo.Objective(expr = 0)
  return

def create_subsystems(model, constr_lists, constr_map, no_objective=True):
  names = []
  for idx, constr_list in enumerate(constr_lists):
    name = f"subsystem_{idx}"
    names.append(name)
    if no_objective:
      create_subsystem_from_constr_list_no_obj(model, constr_list, constr_map, name)
    else:
      create_subsystem_from_constr_list(model, constr_list, constr_map, name)
  return names

def solve_subsystems_sequential(model, subsystem_names, folder, solver="ipopt"):
  assert solver == "ipopt" or solver == "conopt"
  solver_success = [True]*len(subsystem_names)
  for idx, name in enumerate(subsystem_names):
    print(f"solving {name}")
    solver_success[idx] = solve_subsystem(model.find_component(name), folder, solver, idx)

  return solver_success

def solve_subsystem(subsystem, folder, solver, idx):
    assert solver == "ipopt" or solver == "conopt"
    # subsystem.pprint()
    if solver == "ipopt":
      return solve_subsystem_ipopt(subsystem, folder, idx)
    if solver == "conopt":
      return solve_subsystem_conopt(subsystem, folder, idx)


def solve_subsystem_conopt(subsystem, folder, id):
  # subsystem.pprint()
  # from pyomo.common.tempfiles import TempfileManager
  # TempfileManager.tempdir = "/.nfs/home/strahlw/GAMS/pyomoTmp/conopt"
  solver =  pyo.SolverFactory('gams')
  opts = {}
  opts["solver"] = "conopt"
  opts['add_options'] = ["Option IterLim=1000;"]#, "GAMS_MODEL.optfile = 1;"]
  try:
    # 1000 iterations for conopt
    results = solver.solve(subsystem, io_options=opts, 
      #tmpdir= "/.nfs/home/strahlw/GAMS/pyomoTmp/conopt", keepfiles=True,
      logfile=os.path.join(folder,"block_{}_logfile_conopt.log".format(id)))
    if results.solver.status == SolverStatus.error:
      return False
    if results.solver.status == SolverStatus.warning:
      if results.solver.termination_condition != TerminationCondition.infeasible and \
         results.solver.termination_condition != TerminationCondition.feasible and \
         results.solver.termination_condition != TerminationCondition.maxIterations:
         return False
  except Exception as e:
    print("Solver ERROR")
    print(e)
    return False  
  return True

def solve_subsystem_ipopt(subsystem, folder, id):
  # from pyomo.common.tempfiles import TempfileManager
  # TempfileManager.tempdir = "/.nfs/home/strahlw/GAMS/pyomoTmp/ipopt"
  solver = pyo.SolverFactory('ipopt')
  # solver.options['OF_mu_init'] = 0.0001
  # solver.options['OF_bound_push'] = 1e-10
  # solver.options['OF_bound_frac'] = 1e-10
  # solver.options['OF_bound_relax_factor'] = 0
  solver.options['OF_max_iter'] = 300
  # solver.options['OF_halt_on_ampl_error'] = 'yes'
  try:
    results = solver.solve(subsystem, logfile=os.path.join(folder,"block_{}_logfile_ipopt.log".format(id)))
    if results.solver.status == SolverStatus.error:
      return False
    if results.solver.status == SolverStatus.warning:
      if results.solver.termination_condition != TerminationCondition.infeasible and \
         results.solver.termination_condition != TerminationCondition.feasible and \
         results.solver.termination_condition != TerminationCondition.maxIterations:
         return False
  except Exception as e:
    print("Solver ERROR")
    print(e)
    return False
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

def get_original_bounds(model, incidenceGraph):
  initial_bounds = ComponentMap()
  for var in incidenceGraph.variables:
    initial_bounds[var] = var.bounds
  return initial_bounds

def execute_fbbt(model):
  fbbt(model, feasibility_tol = 1e-4, max_iter = 100, improvement_tol = 1e-2)
  
def get_initial_values_guesses(model, incidenceGraph):
  initial_vals = ComponentMap()
  for var in incidenceGraph.variables:
    initial_val = uniform_random_heuristic(var)
    initial_vals[var] = initial_val
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
  
def phase_I(model, complicating_variables, simple_variables, subsystem_names, folder, iteration, solver):
  # assume all the variables are unfixed
  unfix_variables(simple_variables)
  assert all(i.fixed == False for i in complicating_variables)
  assert all(i.fixed == False for i in simple_variables)
  folder += "_Phase_I_{}".format(iteration)
  create_subfolder(folder)
  fix_variables(complicating_variables)
  # solve the subsystems and keep track of solver status
  return solve_subsystems_sequential(model, subsystem_names, folder, solver)

def get_normalized_change_var(old_val, new_val):
  if old_val == new_val:
    return 0
  if old_val == 0:
    return abs(new_val)
  # measures the normalized change of each variable
  return abs(new_val - old_val) / abs(old_val)

def get_initial_values_default(model, igraph):
  initial_vals = ComponentMap()
  for var in igraph.variables:
    if var.value == None:
      # maybe change this to 1?
      initial_vals[var] = 0
    else:
      initial_vals[var] = var.value
  return initial_vals

def get_list_normalized_change(list_vars, list_old):
  to_return = []
  for i in range(len(list_vars)):
    normalized_change = get_normalized_change_var(list_old[list_vars[i]], list_vars[i].value)
    if normalized_change > 1 and abs(list_vars[i].value) > 1:
      print("Name = ", list_vars[i].name)
      print("Old value = ", list_old[list_vars[i]])
      print("New value = ", list_vars[i].value)
  return [get_normalized_change_var(list_old[list_vars[i]], list_vars[i].value) for i in range(len(list_vars))]

def phase_II(model, complicating_constr_subsystem_name, complicating_vars, complicating_constr, simple_vars, folder, solver, iteration):
  # unfix the complicating variables
  unfix_variables(complicating_vars)
  fix_variables(simple_vars)
  # print(f"solving {complicating_constr_subsystem_name}")
  name = complicating_constr_subsystem_name
  # print("Number active constraints = ", 
  # len([constr for constr in model.find_component(name).component_data_objects(pyo.Constraint) if constr.active]))
  # print("Number of unfixed vars = ",
  # len([var for var in model.find_component(name).component_data_objects(pyo.Var) if not var.fixed]))
  return solve_subsystem(model.find_component(complicating_constr_subsystem_name), folder, solver, f"complicating_constr_{iteration}")

def initialize_old_vals(variables):
  old_vals = ComponentMap()
  for var in variables:
    if var.value == None:
      old_vals[var] = 0
    else:
      old_vals[var] = var.value
  return old_vals

def reset_bounds(variables, initial_bounds):
  for var in variables:
    var.bounds = initial_bounds[var]

def strip_bounds(variables):
  # remove bounds on all the variables
  for var in variables:
    var.setlb(None)
    var.setub(None)

def deactivate_constraints(model):
  for constr in model.component_data_objects(pyo.Constraint):
    constr.deactivate()

def activate_constraints(model):
  for constr in model.component_data_objects(pyo.Constraint):
    constr.activate()

def initialize_model_to_heuristic_point(model, use_fbbt):
    igraph, incidence_matrix = get_incidence_matrix(model)
    if use_fbbt:
      execute_fbbt(model)
    initial_vals = get_initial_values_guesses(model, igraph)
    for var in igraph.variables:
        var.value = initial_vals[var]
    return

def assign_initial_point_distance(parameter, initial_vals):
    # feasible solution loaded in var
    # parameter = 1 is the random point
    # parameter = 0 is the feasible point
    for var in initial_vals:
      var.value = (1-parameter)*var.value + parameter*initial_vals[var]
    return


def strip_bounds_tf(model):
  pyo.TransformationFactory('contrib.strip_var_bounds').apply_to(model)
  return

def print_subsystem_var(var, blocks, idx_var_map, complicating_vars):
  subsystem_counter = 0
  for vars, constraints in blocks:
    for var_index in vars:
      if idx_var_map[var_index].name == var.name:
        print(var.name, "in subsystem : ", subsystem_counter)
    subsystem_counter += 1
  for vars in complicating_vars:
    if vars.name == var.name:
      print(var.name , " In Complicating Constraint Subsystem")

def initialization_strategy(model, folder, method=2, num_part=4, fraction=0.5, 
  matched=False, d_max=1, n_max=2, solver="ipopt", use_init_heur="True", use_fbbt="True",
  max_iteration=100, border_fraction=0.5, algo_configuration=algorithmConfiguration(), test=0,
  strip_model_bounds=False, distance=None):
  # the solver option default is a warm-start or an initial point
  # this provides a heuristic to give the solvers a starting point
  #assert use_init_heur == True 
  final_folder = "FinalResultsRunsDistance"
  if not os.path.exists(final_folder):
    os.mkdir(final_folder)
  first_folder = folder
  create_subfolder(folder)
  folder = os.path.join(folder, "Results")
  create_subfolder(folder)
  igraph, incidence_matrix = get_incidence_matrix(model)
  list_variables = igraph.variables
  list_constraints = igraph.constraints

  # print("Num variables = ", len(list_variables))
  # print("Num constraints = ", len(list_constraints))
  # for var in list_variables:
  #   if var.lb != None and var.lb > 0 and var.lb < 1e-10:
  #     # print("lb is ", var.lb)
  #     # print("var name is ", var.name)
  #     var.setlb(1e-10)
  # sys.exit(0)

  initial_bounds = get_original_bounds(model, igraph)
  if use_init_heur:
    if use_fbbt:
      execute_fbbt(model)
    initial_vals = get_initial_values_guesses(model, igraph)
    if distance == None:
      for var in igraph.variables:
        var.value = initial_vals[var]
  else:
    initial_vals = get_initial_values_default(model, igraph)
  
  old_vals = initialize_old_vals(list_variables)

  # output the value of the variables for feasible point
  # for var in initial_vals:
  #   print("Feasible point = ", var.value)
  #   print("Initial point = ", initial_vals[var])
    
  if distance != None:
    assign_initial_point_distance(distance, initial_vals)
    old_vals = initialize_old_vals(list_variables)

  
  # for var in initial_vals:
  #   print("Combined point = ", var.value)
  # sys.exit(0)


  # if solver == "ipopt":
  reset_bounds(list_variables, initial_bounds)

  if strip_model_bounds:
    strip_bounds_tf(model)
  # for var in list_variables:
  #   if var.lb == 0:
  #     if var.ub == 0: # make sure consistent for variables with equivalent lower and upper bounds
  #       var.setub(1e-15)
  #     # set small tolerance
  #     var.setlb(1e-15)
  col_order, row_order, blocks, idx_var_map, idx_constr_map = \
    get_restructured_matrix_general(incidence_matrix, igraph, model, folder, method, fraction, num_part, matched, 
      d_max, n_max, border_fraction)
  constr_list = [i[1] for i in blocks]
  complicating_constr = get_list_of_complicating_constraints(blocks, idx_constr_map)
  complicating_constr_idxs = get_list_complicating_constraint_indices(blocks, idx_constr_map)
  complicating_vars = get_list_of_complicating_variables(blocks, idx_var_map)
  simple_vars = get_list_of_simple_variables(blocks, idx_var_map)
  complicating_constr_subsystem_name = "complicating constraints"


  # var_to_check = [
  #     model.fs.heat_exchanger.hot_side.properties[0.0,0.4].temperature,
  #     model.fs.heat_exchanger.hot_side._enthalpy_flow[0.0,0.35,'Liq'],
  #     model.fs.heat_exchanger.hot_side._enthalpy_flow[0.0,0.85,'Liq'],
  #     model.fs.heat_exchanger.hot_side.enthalpy_flow_dx[0.0,0.35,'Liq'],
  #     model.fs.heat_exchanger.hot_side.enthalpy_flow_dx[0.0,0.85,'Liq']]
  # for var in var_to_check:
  #   print_subsystem_var(var, blocks, idx_var_map, complicating_vars)
  # sys.exit(0)

  # algorithmic configurations
  phase_I_objective = algo_configuration.phase_I["subproblem_objective"]
  phase_II_objective = algo_configuration.phase_II["subproblem_objective"]
  phase_II_objective_constraints = algo_configuration.phase_II["constraints"]
  deactivate_constraints_phase_II = False
  deactivate_constraints_phase_I = False

  if phase_I_objective == 0:
    print("option A")
    subsystems = create_subsystems(model, constr_list, idx_constr_map, no_objective=True)
  if phase_I_objective == 1:
    print("option B")
    subsystems = create_subsystems(model, constr_list, idx_constr_map, no_objective=False)
    deactivate_constraints_phase_I = True

  if phase_II_objective == 0:
    print("Option B.5")
    assert phase_II_objective_constraints == 0 # don't want to solve the whole problem
    create_subsystem_from_constr_list_no_obj(model, complicating_constr_idxs, idx_constr_map,
      complicating_constr_subsystem_name)


  if phase_II_objective == 1 and phase_II_objective_constraints == 0:
    print("option C")
    # minimizing the sum of squared residuals of the linking constraints
    create_subsystem_from_constr_list(model, complicating_constr_idxs, idx_constr_map,
      complicating_constr_subsystem_name)
    deactivate_constraints_phase_II = True
  if phase_II_objective == 1 and phase_II_objective_constraints == 1:
    print("option D")
    # more complicated, we want to minimize the residual of the entire problem in the 
    # final subproblem, but we need to reactivate constraints for the subsystem solve
    create_subsystem_from_constr_list_no_obj(model,
      [i for i in idx_constr_map], idx_constr_map,
      complicating_constr_subsystem_name)
    deactivate_constraints_phase_II = True
    model.find_component(complicating_constr_subsystem_name).del_component("obj")
    # scale the sum of squares, should not affect the optimization
    model.find_component(complicating_constr_subsystem_name).add_component("obj",
    # pyo.Objective(expr = create_residual_objective_expression([idx_constr_map[i] for i in idx_constr_map])))
    #pyo.Objective(expr = create_residual_objective_expression(complicating_constr)))
    pyo.Objective(expr = 1/1e10 * create_residual_objective_expression(complicating_constr)))
  
  print("Configuration = ", algo_configuration.config)
  print("Subsystem_objectives = ") 
  model.find_component(subsystems[0]).obj.pprint()
  print("Number active constraints = ", 
    len([constr for constr in model.find_component(subsystems[0]).component_data_objects(pyo.Constraint) if constr.active]))
  print("Complicating constraints objective  = ") 
  model.find_component(complicating_constr_subsystem_name).obj.pprint()
  print("Number active constraints = ", 
    len([constr for constr in model.find_component(complicating_constr_subsystem_name).component_data_objects(pyo.Constraint) if constr.active]))
  # data collection
  data_maximum_change = [0]
  # data_maximum_change_var = ["None"]
  data_sum_violation = []
  data_subsystem_success = ["None"]


  init_sum_violation = sum(get_constraint_violation(constr) for constr in list_constraints)
  print("initial sum of violation = ", init_sum_violation)
  data_sum_violation.append(init_sum_violation)


  iteration = 0
  maximum_change = 1

  while True:
    iteration += 1
    if iteration > max_iteration:
      print("Reached maximum iteration limit")
      break
                           
    tolerance = 1e-2 # change by 1%
    for var in initial_vals:
      if var.value == None:
        old_vals[var] = 0
      else:
        old_vals[var] = float(var.value)
    if maximum_change < tolerance:
      break

    if algo_configuration.constraint_consensus and algo_configuration.config in [7,8]:
      print("Constraint consensus on subsystems with complicating variables")
      # constraint consensus on the subsystem constraints and the complicating variables
      constraint_list_cc1 = [idx_constr_map[constr] for block_constr in constr_list for constr in block_constr]
      get_closer_initial_point(constraint_list_cc1, complicating_vars, 0.1, 1.0, 20)
      unfix_variables(complicating_vars)

    subsystem_success = [True]
    activate_constraints(model)
    if algo_configuration.phase_I["nonlinear_program"] == 1:
      if deactivate_constraints_phase_I:
        deactivate_constraints(model)
      subsystem_success = phase_I(model, complicating_vars, simple_vars, subsystems, folder, iteration, solver)
      get_list_normalized_change(simple_vars, old_vals)

    activate_constraints(model)
    if algo_configuration.phase_II["nonlinear_program"] == 1:
      if deactivate_constraints_phase_II:
        deactivate_constraints(model)
      subsystem_success.append(phase_II(model, complicating_constr_subsystem_name, complicating_vars, complicating_constr, simple_vars, folder, solver, iteration))
      # get_list_normalized_change(complicating_vars, old_vals)

    # reactivate the constraints
      # if deactivate_constraints_phase_II:
      #   activate_constraints(model)

    if algo_configuration.constraint_consensus and algo_configuration.config in [7,8]:
      # constraint consensus on the subsystem constraints and the complicating variables
      constraint_list_cc2 = [i for i in complicating_constr]
      print("Constraint consensus on complicating constraints with simple variables")
      get_closer_initial_point(constraint_list_cc2, simple_vars, 0.1, 1.0, 20)
    
   
    if iteration >= 1:
      print("Iteration = ", iteration)
      sum_violation = sum(get_constraint_violation(constr) for constr in list_constraints)
      print("Maximum change = ", maximum_change)
      print("sum of violation = ", sum_violation)
      #print(get_list_normalized_change([var for var in initial_vals], old_vals))
      maximum_change = max(get_list_normalized_change([var for var in initial_vals], old_vals))
      data_maximum_change.append(maximum_change)
      data_sum_violation.append(sum_violation)
      data_subsystem_success.append({i: subsystem_success[i] for i in range(len(subsystem_success))})
  # for subsystem in subsystems:
  #   subsystem.pprint()
  # sys.exit(0)
  # sum_violation = sum(get_constraint_violation(constr) for constr in list_constraints)
  # print("number of iterations = ", iteration)
  # print("final max change = ", maximum_change)
  # print("final sum of violation = ", sum_violation)
  # unfix_variables(simple_vars)
  # data_maximum_change.append(maximum_change)
  # data_sum_violation.append(sum_violation)
  # data_subsystem_success.append({i: subsystem_success[i] for i in range(len(subsystem_success))})
  # for a final solve
  for idx in idx_constr_map:
    idx_constr_map[idx].activate()
  for block in model.component_data_objects(pyo.Block):
    block.del_component(block.find_component("obj"))
  #model.obj = pyo.Objective(expr=0)
  unfix_variables(simple_vars)
  unfix_variables(complicating_vars)

  with open(os.path.join(folder, "Results.txt"), 'w') as file:
    for i in range(len(data_maximum_change)):
      file.write(f"Iteration : {i}\n")
      file.write(f"maximum change : {data_maximum_change[i]}\n")
      file.write(f"sum violation : {data_sum_violation[i]}\n")
      if i > 0:
        file.write(f"difference violation : {data_sum_violation[i] - data_sum_violation[i-1]}\n")
        for j in range(len(data_subsystem_success[i])):
          if j == len(data_subsystem_success[i]) - 1:
            file.write(f"Complicating constraint subsystem : {data_subsystem_success[i][j]}\n")
          else:
            file.write(f"Subsystem {j} : {data_subsystem_success[i][j]}\n")
      file.write("\n")
  
  # write the final results file with all the pertinent information for data visualization
  with open(os.path.join(final_folder, first_folder + ".txt"), 'w') as file:
    # information needed - all the parameters :
    # restructuring algorithm type
    file.write(f"method : {method}\n")
    # matched
    file.write("matched : {}\n".format(1 if matched else 0))
    # restructuring algorithm parameters
    file.write(f"d_max : {d_max}\n")
    file.write(f"n_max : {n_max}\n")
    file.write(f"border_fraction : {border_fraction}\n")
    file.write(f"fraction : {fraction}\n")
    file.write(f"num_part : {num_part}\n")
    # size of the border - for now assume square
    size_border = len(complicating_constr)
    file.write(f"size_border : {size_border}\n")
    # number of blocks
    num_blocks = len(blocks)
    file.write(f"num_blocks : {num_blocks}\n")
    # size largest block
    largest_block = max(len(blocks[i][0]) for i in range(len(blocks)))
    file.write(f"largest_block : {largest_block}\n") 
    # final infeasibility
    sum_violation = sum(get_constraint_violation(constr) for constr in list_constraints)
    file.write(f"initial_infeasibility : {init_sum_violation}\n")
    file.write(f"final_infeasibility : {sum_violation}\n")
    # subproblem failures
    failure = 0
    for i in range(len(data_subsystem_success)):
      for j in range(len(data_subsystem_success[i])):
        if not data_subsystem_success[i][j]:
          failure += 1
    file.write(f"subproblem_failures : {failure}\n")
    # algorithm configuration
    file.write(f"algo_configuration : {algo_configuration.config}\n")
    # initial point strategy
    use_init_heur_str = "0" if not use_init_heur else "1"
    use_fbbt_str = "0" if not use_fbbt else "1"
    strip_bounds_str = "0" if not strip_model_bounds else "1"
    file.write(f"initialization_strategy : {use_init_heur_str}{use_fbbt_str}{strip_bounds_str}\n")
    # number of iterations
    file.write(f"iterations : {iteration-1}\n")
    # which solver
    file.write(f"solver : {solver}\n")
    # which test
    file.write(f"test : {test}\n")


def initialization_strategy_LC_overlap(model, folder, method=2, num_part=4, fraction=0.5, 
  matched=False, d_max=1, n_max=2, solver="ipopt", use_init_heur="True", use_fbbt="True",
  max_iteration=100, border_fraction=0.5, algo_configuration=algorithmConfiguration(), test=0,
  strip_model_bounds=False, distance=None):
  final_folder = "FinalResultsRunsDistance"
  if not os.path.exists(final_folder):
    os.mkdir(final_folder)
  first_folder = folder
  create_subfolder(folder)
  folder = os.path.join(folder, "Results")
  create_subfolder(folder)
  igraph, incidence_matrix = get_incidence_matrix(model)
  list_variables = igraph.variables
  list_constraints = igraph.constraints
  initial_bounds = get_original_bounds(model, igraph)
  if use_init_heur:
    if use_fbbt:
      execute_fbbt(model)
    initial_vals = get_initial_values_guesses(model, igraph)
    if distance == None:
      for var in igraph.variables:
        var.value = initial_vals[var]
  else:
    initial_vals = get_initial_values_default(model, igraph)
  
  old_vals = initialize_old_vals(list_variables)

  if distance != None:
    assign_initial_point_distance(distance, initial_vals)
    old_vals = initialize_old_vals(list_variables)

  reset_bounds(list_variables, initial_bounds)

  if strip_model_bounds:
    strip_bounds_tf(model)

  col_order, row_order, blocks, idx_var_map, idx_constr_map = \
  get_restructured_matrix_general(incidence_matrix, igraph, model, folder, method, fraction, num_part, matched, 
    d_max, n_max, border_fraction)
  constr_list = [i[1] for i in blocks]
  var_list = [i[0] for i in blocks]
  complicating_constr = get_list_of_complicating_constraints(blocks, idx_constr_map)
  complicating_constr_idxs = get_list_complicating_constraint_indices(blocks, idx_constr_map)
  complicating_vars = get_list_of_complicating_variables(blocks, idx_var_map)
  simple_vars = get_list_of_simple_variables(blocks, idx_var_map)
  complicating_constr_subsystem_name = "complicating constraints"

  subsystem_constr_list = [i + complicating_constr_idxs for i in constr_list]
  subsystems = create_subsystems(model, subsystem_constr_list, idx_constr_map, no_objective=True)
  constr_list_all = [idx_constr_map[constr] for constr in idx_constr_map]

  # create "global approximation" - shared solution
  # initialize to the value the variables are initialized to
  global_approximation = ComponentMap()
  for idx in idx_var_map:
      global_approximation[idx_var_map[idx]] = idx_var_map[idx].value
    
  # for var in global_approximation:
  #   print(var.name, " : ", var.value)
  
  subsystem_solutions = []
  for subsystem in subsystems:
      solutions = ComponentMap()
      for idx in idx_var_map:
          solutions[idx_var_map[idx]] = global_approximation[idx_var_map[idx]]
      subsystem_solutions.append(solutions)
  
  # the iteration part of the algorithm 
  iteration = 0
  while True:
    if iteration >= max_iteration:
      break

    for idx, subsystem in enumerate(subsystems):
        solve_subsystem(model.find_component(subsystem), folder, solver, f"{idx}")
        print("Solve subsystem {}".format(idx))
        for var in subsystem_solutions[idx]:
            # print(var.name, " : ", var.value)
            subsystem_solutions[idx][var] = var.value
            var.value = global_approximation[var]
    
    # update the global approximation from each subsystem
    for i in range(len(var_list)):
        # print("Updates from subsystem {}".format(i))
        for var_idx in var_list[i]:
            # print(idx_var_map[var_idx].name, " new value = ", subsystem_solutions[i][idx_var_map[var_idx]])
            idx_var_map[var_idx].value = set_within_bounds(subsystem_solutions[i][idx_var_map[var_idx]], 
              idx_var_map[var_idx].lb, idx_var_map[var_idx].ub)
    
    # update the linking constraint variables
    for var in complicating_vars:
        var.value = set_within_bounds(sum([subsystem_solutions[i][var] for i in range(len(subsystem_solutions))])/len(subsystem_solutions), var.lb, var.ub)
    
    # update the global approximation
    for var in global_approximation:
        global_approximation[var] = var.value
    
    # print("updated global approximation")
    # for var in global_approximation:
    #     print(var.name, " : ", var.value)
    
    print("sum of violation : ", sum(get_constraint_violation(constr) for constr in constr_list_all))
    iteration += 1

def get_closer_initial_point(list_constraints, list_variables, alpha, beta, iter_lim):
  iteration = 0
  fv_consensus, distances, s = constraint_consensus(list_constraints, list_variables, alpha)
  current_violation = sum(get_constraint_violation(constr) for constr in list_constraints)
  print("initial_violation = ", current_violation)
  while True:
    if sum(s) == 0:
      print("Exited on feasibility vector distance tolerance or on feasibility - alpha")
      return
    if norm_l2(fv_consensus) < beta:
      print("Exited on length of move, beta")
      return
    if iteration >= iter_lim:
      print("Reached maximum # of iterations")
      return
    for i in range(len(list_variables)):
      list_variables[i].value = adjust_consensus_value(list_variables[i].value + fv_consensus[i], list_variables[i].lb, list_variables[i].ub)
    iteration_violation = sum(get_constraint_violation(constr) for constr in list_constraints)
    if abs(current_violation - iteration_violation)/current_violation < 1e-2:
      print("Change is less than 1 percent of violation - terminate")
      return 
    current_violation = iteration_violation
    print("iteration : {}".format(iteration))
    print("violation = ", current_violation) 
    iteration += 1
    fv_consensus, distances, s = constraint_consensus(list_constraints, list_variables, alpha)
    
  return 

def constraint_consensus(list_of_constraints, list_of_variables, alpha):
  #np.seterr(all='raise')
  # keeps track of how many contribute component-wise
  # s_var = {i : 0 for i in range(len(vars))}
  grad_constr = list(np.array(get_constraint_gradient_all_vars(constr, list_of_variables)) 
    for constr in list_of_constraints)
  
  norm_grad_constr = list(norm_l2(grad_c) for grad_c in grad_constr)
  # remove any constraint with 0 norm
  idx_to_remove = list(i for i in range(len(norm_grad_constr)) if norm_grad_constr[i]==0)
  adjust = 0
  for idx in idx_to_remove:
    del norm_grad_constr[idx-adjust]
    del list_of_constraints[idx-adjust]
    del grad_constr[idx-adjust]
    adjust += 1
  # filter out by distance
  v = np.array(list(get_constraint_violation(constr) for constr in list_of_constraints))
  distances = list(v[i] / norm_grad_constr[i] if norm_grad_constr[i] != 0 else 0 for i in range(len(norm_grad_constr)))

  # eliminate any contributions of feasibility vectors that don't have a sufficient feasibility distance
  v = np.array([v[i] if distances[i] >= alpha else 0 for i in range(len(v))])
  d = np.array(list(determine_d_constr(constr) for constr in list_of_constraints))
  squared_norm_grad_constr = np.array(list(i**2 for i in norm_grad_constr))
  # s is indexed by variable
  s = list(sum(1 if grad_constr[j][i] != 0 and v[j] != 0 else 0 for j in range(len(grad_constr)))
           for i in range(len(list_of_variables)))
  # fv = feasibility_vector, checked all divide by zeros
  multiplier = d*v/squared_norm_grad_constr 
  fv_list = list(multiplier[i]*grad_constr[i] for i in range(len(multiplier)))  # indexed by variable
  fv_consensus = np.sum(fv_list, axis=0)
  fv_consensus = list(fv_consensus[i]/s[i] if s[i] != 0 else 0 for i in range(len(s)))
  # print("Fv consensus, distances, s, multiplier, squared_norm_grad_constr, norm_grad_constr, grad_constr : ")
  # print(fv_consensus)
  # print(distances)
  # print(s)
  # print("multiplier = ", multiplier)
  # print("squared norm grad constr = ", squared_norm_grad_constr)
  # print("norm grad constr = ", norm_grad_constr)
  # print("grad_constr = ", grad_constr)
  return fv_consensus, distances, s

def determine_d_constr(constraint):
  assert constraint.equality
  return 1 if pyo.value(constraint.body) < constraint.upper else -1

def get_constraint_gradient(constraint):
  return differentiate(constraint.body, wrt_list=list(identify_variables(constraint.body)))

def get_constraint_gradient_all_vars(constraint, all_vars):
  return differentiate(constraint.body, wrt_list=all_vars)

def norm_l2(vector):
  return math.sqrt(sum(i**2 for i in vector))

def get_constraint_violation(constraint):
  return abs(pyo.value(constraint.body) - constraint.upper)

def adjust_consensus_value(val, lb, ub):
  if ub == None and lb == None:
    return val
  if lb == None:
    return min(val, ub)
  if ub == None:
    return max(lb, val)
  return max(lb, min(val, ub))

def set_within_bounds(val, lb, ub):
  if ub == None and lb == None:
    return val
  if lb == None:
    return min(val, ub)
  if ub == None:
    return max(lb, val)
  return max(lb, min(val, ub))








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
  

# import numpy as np 
# import scipy as sc
# import matplotlib.pyplot as plt
# import random
# import sys


# def create_matrix(seed):
#     random.seed(seed)
#     size_matrix = 20
#     size_matrix_m = size_matrix
#     size_matrix_n = size_matrix
#     fill_matrix = 60 # maximum possible fill per row before adding diagonal
#     original_matrix = np.zeros((size_matrix, size_matrix))
#     for i in range(size_matrix):
#         # for each row select a number of indices to make nonzero
#         num_non_zeros = random.randint(0,int((size_matrix-1)*fill_matrix/100))
#         indices_used = []
#         for j in range(num_non_zeros):
#             # for each non zero, randomly choose an index (and keep track)
#             if j == 0:
#                 index_to_fill = random.randint(0, size_matrix-1)
#                 original_matrix[i][index_to_fill] = 1
#                 indices_used.append(index_to_fill)
#             else:
#                 index_to_fill = random.randint(0, size_matrix-1)
#                 while index_to_fill in indices_used:
#                     index_to_fill = random.randint(0, size_matrix-1)
#                 original_matrix[i][index_to_fill] = 1
#                 indices_used.append(index_to_fill)
#     return original_matrix


# def reorder_sparse_matrix(m, n, row_order, col_order, target_matrix):
#   target_matrix = sc.sparse.coo_matrix(target_matrix)
#   permutation_matrix = sc.sparse.eye(m).tocoo()
#   permutation_matrix.col = permutation_matrix.col[row_order]
#   permuted_matrix = permutation_matrix.dot(target_matrix)
#   permutation_matrix = sc.sparse.eye(n).tocoo()
#   permutation_matrix.row = permutation_matrix.row[col_order]
#   return permuted_matrix.dot(permutation_matrix)

# def show_matrix_structure(matrix):
#   plt.spy(matrix)
#   plt.show()

# def matrix_to_edges(matrix):
#     edges = []
#     for i in range(len(matrix)):
#         for j in range(len(matrix[0])):
#             if matrix[i][j] == 1:
#                 edges.append((i,j))
#     return edges 

# # original_matrix = np.array([[0,0],[0,0]])
# # print(original_matrix)
# # show_matrix_structure(original_matrix)


# seeds = range(100)
# failed = []
# for i in seeds:
#     print("Instance ", i)
#     original_matrix = create_matrix(i)
#     test = BBBD_algo(matrix_to_edges(original_matrix), len(original_matrix), len(original_matrix[0]), 0.5)
#     # try:
#     col_order, row_order, blocks = test.solve()

#     print(max(test.blocks[i].size for i in test.blocks))
#     print([test.blocks[i].size for i in test.blocks])
# reordered_incidence_matrix = reorder_sparse_matrix(len(row_order),
#     len(col_order), row_order, col_order, original_matrix)
#     except:
#         failed.append(i)

#failed = [8]
# original_matrix = create_matrix(14)
# print(original_matrix)
# # show_matrix_structure(original_matrix)
# test = BBBD_algo(matrix_to_edges(original_matrix), len(original_matrix), len(original_matrix[0]), 1.1)
# col_order, row_order, blocks = test.solve()
# reordered_incidence_matrix = reorder_sparse_matrix(len(row_order),
#     len(col_order), row_order, col_order, original_matrix)
# print(col_order, row_order, blocks)
# show_matrix_structure(reordered_incidence_matrix)
# print(failed)

# edges = [(0,1),(0,3), (1,0), (1,2), (2,2), (3,1)]
# m = 4
# n = 4

# bbbd_algo = BBBD_algo(edges, m, n, 0.5)
# bbbd_algo.iteration()
# bbbd_algo.iteration()
# bbbd_algo.iteration()
# bbbd_algo.iteration()
