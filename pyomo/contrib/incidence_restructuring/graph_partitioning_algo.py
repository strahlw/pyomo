import pymetis as pymetis
import pyomo.environ as pyo
import sys

def partition_graph(num_partitions, adjacency):
  n_cuts, membership = pymetis.part_graph(num_partitions, adjacency=adjacency)
  return n_cuts, membership

def get_column_order_partitions(num_part, adjacency_list, edge_set, membership):
  # initialize the model
  m_border = pyo.ConcreteModel()

  # index set of partitions
  m_border.p = pyo.RangeSet(0, num_part-1)\
  # set of nodes
  m_border.i = pyo.RangeSet(0,len(adjacency_list)-1)
  # set of nodes
  m_border.j = pyo.RangeSet(0, len(adjacency_list)-1)


  m_border.E = pyo.Set(initialize=edge_set)

  m_border.y = pyo.Var(m_border.i, domain=pyo.NonNegativeReals, bounds=(0,1))
  m_border.z = pyo.Var(m_border.E, domain=pyo.NonNegativeReals, bounds=(0,1))

  @m_border.Constraint(m_border.i)
  def separate_partitions(m, i):
    # no indices for node that extend out of partition
    subset_indices = [(i,j) for j in m_border.j if (membership[i] != membership[j] and (i,j) in m_border.E)]
    if len(subset_indices) == 0:
      return pyo.Constraint.Skip
    else:
      return sum(m_border.z[k[0],k[1]] for k in subset_indices) == 0

  @m_border.Constraint(m_border.E)
  def lb_constraint_1(m, i, j):
    return m_border.z[i,j] >= m_border.y[i]

  @m_border.Constraint(m_border.E)
  def lb_constraint_2(m, i, j):
    return m_border.z[i,j] >= m_border.y[j]

  @m_border.Constraint(m_border.E)
  def ub_constraint(m, i, j):
    return m_border.z[i,j] <= m_border.y[i] + m_border.y[j]

  @m_border.Objective(sense=pyo.maximize)
  def obj(m):
    return sum(m_border.y[i] for i in m_border.i)

  solver = pyo.SolverFactory('glpk')
  solution = solver.solve(m_border, tee=True)

  assert all(pyo.value(m_border.y[i]) == 0 or pyo.value(m_border.y[i]) == 1 for i in m_border.i)
  vertex_included = [bool(pyo.value(m_border.y[i])) for i in m_border.i]
  partitions = {i : [] for i in range(max(membership)+1)}
  for i in range(len(membership)):
    if vertex_included[i]:
      partitions[membership[i]].append(i)

  border_indices = [i for i in range(len(vertex_included)) if not vertex_included[i]]
  
  order_columns = []
  for i in partitions:
    for k in partitions[i]:
      order_columns.append(k)
  border_columns = [column for column in range(len(adjacency_list)) if column not in order_columns]
  final_columns = order_columns + border_columns
  return final_columns, partitions, border_indices

def graph_partitioning_algorithm(num_part, adjacency_list):
  edge_set = [[(i,j) for j in adjacency_list[i]] for i in range(len(adjacency_list))]
  edge_set = [elem for array in edge_set for elem in array]
  adjacency_list_pymetis = [[k for k in adjacency_list[i]] for i in adjacency_list]
  n_cuts, membership = partition_graph(num_part, adjacency_list_pymetis)
  return get_column_order_partitions(num_part, adjacency_list, edge_set, membership)

def get_partitions_general(num_part, edge_set, membership, num_vars, num_constr):
  # initialize the model
  m_border = pyo.ConcreteModel()

  # index set of partitions
  m_border.p = pyo.RangeSet(0, num_part-1)
  # set of constraint nodes
  m_border.i = pyo.RangeSet(0,num_vars-1)
  # set of variable nodes
  m_border.j = pyo.RangeSet(0, num_constr-1)


  m_border.E = pyo.Set(initialize=edge_set)

  m_border.y = pyo.Var(m_border.i, domain=pyo.NonNegativeReals, bounds=(0,1))
  m_border.w = pyo.Var(m_border.j, domain=pyo.NonNegativeReals, bounds=(0,1))
  m_border.z = pyo.Var(m_border.E, domain=pyo.NonNegativeReals, bounds=(0,1))

  @m_border.Constraint(m_border.i)
  def separate_partitions(m, i):
    # no indices for node that extend out of partition
    # edit the input data for membership to make it amenable for the same code - split into vars and constr
    # also need to edit the edge set to distinguish between variables and constraints
    subset_indices = [(i,j) for j in m_border.j if (membership[0][i] != membership[1][j] and (i,j) in m_border.E)]
    if len(subset_indices) == 0:
      return pyo.Constraint.Skip
    else:
      return sum(m_border.z[k[0],k[1]] for k in subset_indices) == 0

  @m_border.Constraint(m_border.E)
  def lb_constraint_1(m, i, j):
    return m_border.z[i,j] >= m_border.y[i]

  @m_border.Constraint(m_border.E)
  def lb_constraint_2(m, i, j):
    return m_border.z[i,j] >= m_border.w[j]
  @m_border.Constraint(m_border.E)
  def ub_constraint(m, i, j):
    return m_border.z[i,j] <= m_border.y[i] + m_border.w[j]
  
  @m_border.Constraint(m_border.p)
  def square_systems(m, k):
     variables_in_k = [i for i in m_border.i if membership[0][i] == k]
     constraints_in_k = [j for j in m_border.j if membership[1][j] == k]
     return sum(m_border.y[i] for i in variables_in_k) == sum(m_border.w[j] for j in constraints_in_k)

  @m_border.Objective(sense=pyo.maximize)
  def obj(m):
    return sum(m_border.y[i] for i in m_border.i)

  solver = pyo.SolverFactory('glpk')
  solution = solver.solve(m_border, tee=True)

  assert all(pyo.value(m_border.y[i]) <= 1e-4 or pyo.value(m_border.y[i]) >= 0.999 for i in m_border.i)
  assert all(pyo.value(m_border.w[j]) <= 1e-4 or pyo.value(m_border.w[j]) >= 0.999 for j in m_border.j)

  # post-processing
  variable_included = [bool(int(pyo.value(m_border.y[i]))) for i in m_border.i]
  constraint_included = [bool(int(pyo.value(m_border.w[j]))) for j in m_border.j]
  # partitions include variable vertices and constraint vertices
  # list variables first, then constraints
  partitions = {i : [[],[]] for i in range(max(membership[0])+1)}
  # assign variables to partitions
  
  for i in range(len(membership[0])):
    if variable_included[i]:
      partitions[membership[0][i]][0].append(i)
  # assign constraints to partitions
  for j in range(len(membership[1])):
    if constraint_included[j]:
      partitions[membership[1][j]][1].append(j)
  
  border_vars = [i for i in range(len(variable_included)) if not variable_included[i]]
  border_constr = [j for j in range(len(constraint_included)) if not constraint_included[j]]
  
  order_rows = []
  for i in partitions:
    for k in partitions[i][0]:
      order_rows.append(k)
  order_columns = []
  for i in partitions:
    for k in partitions[i][1]:
      order_columns.append(k)
  final_rows = order_rows + border_vars
  final_columns = order_columns + border_constr
  return final_columns, final_rows, [[partitions[k][1], partitions[k][0]] for k in partitions]

def graph_partitioning_algorithm_general(num_part, edge_set, adjacency_list, num_vars, num_constr):
  n_cuts, membership = partition_graph(num_part, adjacency_list)
  membership = [[membership[i] for i in range(num_vars)], [membership[j] for j in range(num_vars, num_vars + num_constr)]]
  return get_partitions_general(num_part, edge_set, membership, num_vars, num_constr)


##################################################################################################
# def update_outside_edges(i,j,outside_edges, membership):
#     assert str(j) not in outside_edges[str(i)]
#     assert str(i) not in outside_edges[str(j)]
#     if membership[i] != membership[j]:
#         outside_edges[str(i)][str(j)] = ""
#         outside_edges[str(j)][str(i)] = ""
    
# def update_size_data(i,j, size_edges, edges_size):
#     size_i = edges_size[str(i)]
#     size_j = edges_size[str(j)]
#     del size_edges[str(size_i)][str(i)]
#     del size_edges[str(size_j)][str(j)]
#     size_edges[str(size_i+1)][str(i)] = ""
#     size_edges[str(size_j+1)][str(j)] = ""
#     edges_size[str(i)] += 1
#     edges_size[str(j)] += 1
#     return max(size_i+1, size_j+1)
    
# def update_size_data_removal(j, size_edges, edges_size):
#     size_j = edges_size[str(j)]
#     del size_edges[str(size_j)][str(j)]
#     size_edges[str(size_j-1)][str(j)] = ""
#     edges_size[str(j)] -= 1
    
# def remove_vertex(i, size_edges, edges_size, outside_edges, membership, additional_partition):
#     for j in outside_edges[str(i)]:
#         update_size_data_removal(j, size_edges, edges_size)
#     membership[i] = additional_partition
#     size_i = edges_size[str(i)]
#     del size_edges[str(size_i)][str(i)]
#     size_edges[str(0)][str(i)] = ""
#     edges_size[str(i)] = 0
    

# def get_column_order_partitions_algo(num_part, adjacency_list, edge_set, membership):
#     # create data structures 
#     outside_edges = {str(i) : {} for i in adjacency_list}
#     size_edges = {"0" : {str(i) : "" for i in adjacency_list}}
#     edges_size = {str(i) : 0 for i in adjacency_list}
#     max_size = 0
#     additional_partition = max(membership)+1
    
#     # populate data structures
#     for i,j in edge_set:
#         update_outside_edges(i,j, outside_edges, membership)
#         max_num = update_size_data(i,j, size_edges, edges_size)
#         max_size = max(max_size, max_num)
    
#     # remove until all 0
#     done = False
#     while max_size > 0:
#         # check if dictionary is empty
#         while edges_size[str(max_size)] == 0:
#             max_size -= 1
#         # don't go through those that aren't connected
#         if max_size == 0:
#             break        
        
#         # assert there are entries in the dictionary
#         assert size_edges[str(max_size)]
#         vertex_to_remove = next(iter(size_edges[str(max_size)]))
#         remove_vertex(vertex_to_remove, size_edges, edges_size, outside_edges, membership, additional_partition)
        
#     partitions = [[]]*additional_partition
#     for i in membership:
#         column_order[membership[i]].append(i)
#     column_order = [elem for array in partitions for elem in array]
#     return column_order, partitions
        
        
        