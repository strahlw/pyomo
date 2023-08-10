import copy
# small change
# required classes
# create data structures for blocks and for border vertices
class Block(object):
  def __init__(self, vertex, label):
    self.vertices = {}
    self.adj_block_vertices = {}
    self.label = label
    self.size = 0

    self.add_vertex(vertex)

  def add_vertex(self, vertex):
    if vertex not in self.vertices:
      self.vertices[vertex] = ""
      self.size += 1

  def remove_vertex(self, vertex):
    if vertex in self.vertices:
      del self.vertices[vertex]

  def add_border_vertex(self, vertex):
    if vertex not in self.adj_block_vertices:
      self.adj_block_vertices[vertex] = ""

  def remove_border_vertex(self, vertex):
    if vertex in self.adj_block_vertices:
      del self.adj_block_vertices[vertex]


  def merge_block(self, another_block):
    for vertex in another_block.vertices:
      self.add_vertex(vertex)
    # also have to merge the adjacent block_vertices
    for vertex in another_block.adj_block_vertices:
      self.add_border_vertex(vertex)

  def update_block(self, vertex_to_add_data_structure):
    for vertex in vertex_to_add_data_structure.adj_border_vert_contains:
      self.add_border_vertex(vertex)
    self.remove_border_vertex(vertex_to_add_data_structure.label)

  def check_in_block(self, vertex):
    return vertex in self.vertices

  def print_data(self):
    print("Label = ", self.label)
    print("Vertices = ", [i for i in self.vertices])
    print("Size = ", self.size)
    print("Adjacent border vertices = ", list(self.adj_block_vertices.keys()))

class BorderVertexData(object):
  def __lt__(self, other):
    if self.total_num_adj_block == other.total_num_adj_block:
      return self.total_num_adj_border < other.total_num_adj_border
    else:
      return self.total_num_adj_block < other.total_num_adj_block

  def add_border_vertex(self, vertex):
    if vertex not in self.adj_border_vert and vertex != self.label:
      self.adj_border_vert[vertex] = ""
      self.total_num_adj_border += 1

  def add_block(self, block_label, blocks):
    if block_label not in self.adj_block:
      self.adj_block[block_label] = ""
      self.total_num_adj_block += blocks[block_label].size

  def add_block_using_block(self, block):
    if block.label not in self.adj_block:
      self.adj_block[block.label] = ""
    # this is only called when updating the block size given either
    # (i) a merged block system or
    # (ii) an initial block
    self.total_num_adj_block += block.size

  def remove_block(self, block_label, size):
    if block_label in self.adj_block:
      self.total_num_adj_block -= size
      del self.adj_block[block_label]

  def remove_border_vertex(self, vertex):
    if vertex in self.adj_border_vert:
      self.total_num_adj_border -= 1
      del self.adj_border_vert[vertex]

  def add_containing_border_vertex(self, vertex):
    if vertex not in self.adj_border_vert_contains and vertex != self.label:
      self.adj_border_vert_contains[vertex] = ""

  def __init__(self, vertex, y_adjacency, border_vertices, blocks):
    self.label = vertex
    self.total_num_adj_border = 0
    self.total_num_adj_block = 0
    # dictionary containing the adjacent vertices in the border
    self.adj_border_vert = {}
    self.adj_border_vert_contains = {}
    # dictionary containing the adjacent blocks
    self.adj_block = {}

    # create adjacency list for border vertices
    # vertices that are adjacent to the vertex
    for i in y_adjacency:
      if i not in border_vertices:
        continue
      for j in y_adjacency[i]:
        if j == vertex:
          #print("adjacent vertex = ", i)
          self.add_containing_border_vertex(i)

    # # create adjacency list for blocks
    # # blocks that are adjacent to the vertex
    # for i in y_adjacency[vertex]:
    #   if i not in border_vertices:
    #     for block_label in blocks:
    #       # print("block label = ", block_label)
    #       # print("Vertices = ", blocks[block_label].vertices)
    #       if i in blocks[block_label].vertices:
    #         self.add_block(block_label, blocks)

    for i in y_adjacency[vertex]:
      # print("adjacent vertex = ", i)
      if i in border_vertices:
        if i != self.label:
          self.add_border_vertex(i)
      else:
        for block_label in blocks:
          # print("block label = ", block_label)
          # print("Vertices = ", blocks[block_label].vertices)
          if i in blocks[block_label].vertices:
            self.add_block(block_label, blocks)

  def get_total_num_adj_block(self):
    return self.total_num_adj_block

  def update_adj_block(self, merged_blocks, new_block, blocks):
    # include the new label in the merged blocks
    # new_block_in_adj_block = new_block.label in self.adj_block
    for i in merged_blocks:
      self.remove_block(i, blocks[i].size)
    self.add_block_using_block(new_block)
    # # add one to the size of the combined blocks because
    # # the merged block has the vertex already added to it
    # # so we add and subtract the same number, but should be +1 at end
    # if new_block_in_adj_block:
    #  self.total_num_adj_block += 1
    # THIS DOESN'T WORK IF ONE OF THE MERGED BLOCKS IS NOT ADJACENT

  def update_adj_border_vertices(self, vertex_data_structure):
    self.remove_border_vertex(vertex_data_structure.label)
    for vertex in vertex_data_structure.adj_border_vert:
      self.add_border_vertex(vertex)

  def update_data(self, vertex_data_structure, merged_blocks, new_block, blocks):
    assert new_block.label in merged_blocks
    self.update_adj_border_vertices(vertex_data_structure)
    self.update_adj_block(merged_blocks, new_block, blocks)

  def update_data_blocks(self, vertex_data_structure, merged_blocks, new_block, blocks):
    self.remove_border_vertex(vertex_data_structure.label)
    self.update_adj_block(merged_blocks, new_block, blocks)

  def print_data(self):
    print("Label = ", self.label)
    print("q_i = ", self.total_num_adj_border)
    print("S_i = ", self.total_num_adj_block)
    print("Adjacent border vertices = ", list(self.adj_border_vert.keys()))
    print("Adjacent blocks = ", list(self.adj_block.keys()))

def bbbd_algo(x_adjacency, x_size, d_max, n_max, fraction_tol, print_debug=False):
  # for all the vertices that are not already assigned to the block
  border_vertices = {i : "" for i in x_adjacency if x_size[i] >= d_max}
  size_problem = len(x_adjacency)
  if print_debug:
    print(border_vertices)
  # the incidence lists
  T = {i : [j for j in x_adjacency[i] if j not in border_vertices] for i in x_adjacency}
  vertex_to_block_map = {}
  all_blocks = {}
  additional_border_vertices = {}
  # if cut, then not in the map, so no worries
  block_label = 0
  for x_c in [i for i in x_adjacency if i not in border_vertices]:
    # determine the blocks containing the adjacent vertices
    blocks = list(set([vertex_to_block_map[i] for i in T[x_c] if i in vertex_to_block_map]))
    # three cases
    # case 1: the blocks are empty -> then create new block
    if len(blocks) == 0:
      x_c_block = Block(x_c, block_label)
      vertex_to_block_map[x_c] = block_label
      block_label += 1
      all_blocks[x_c_block.label] = x_c_block

    # case 2: there is one other column that has a block
    # in this case, check for cutting and add or cut
    if len(blocks) == 1:
      if all_blocks[blocks[0]].size == n_max:
        # add the vertex to the border
        additional_border_vertices[x_c] = ""
      else:
        # add the vertex to the block and book keeping
        all_blocks[blocks[0]].add_vertex(x_c)
        vertex_to_block_map[x_c] = all_blocks[blocks[0]].label

    # final case: there are multiple blocks that need to be merged
    if len(blocks) > 1:
      # check for cutting, this is likely
      total_size = sum([all_blocks[i].size for i in blocks])
      # here we have to add the vertex to the sum of the merge
      if total_size > n_max - 1:
        additional_border_vertices[x_c] = ""
      else:
        # here we need to merge the blocks and update all data structures
        seed_block = blocks[0]
        for i in range(1, len(blocks)):
          all_blocks[blocks[0]].merge_block(all_blocks[blocks[i]])
          # update data structures
          for vertex in all_blocks[blocks[i]].vertices:
            vertex_to_block_map[vertex] = all_blocks[blocks[0]].label
          del all_blocks[blocks[i]]
          # add the vertex to the merged block
        all_blocks[blocks[0]].add_vertex(x_c)
        vertex_to_block_map[x_c] = all_blocks[blocks[0]].label

    # finish stage one and have all the blocks set up

  # now move on to stage 2
  # in this stage we take the vertices that are now assigned to the border and we add them to the blocks until
  # a termination criteria is satisfied
  # for now, the termination criteria will be the one prescribed in the paper, i.e., once an addition to a block makes the block larger than the number of elements left in the border, we are done

  border_vertices_extended = {i : "" for i in list(border_vertices.keys()) + list(additional_border_vertices.keys())}
  size_border = len(border_vertices_extended)
  # update block data structures with adjacent vertices
  # work through this
  for block in all_blocks:
    for vertex in border_vertices_extended:
      if vertex in all_blocks[block].adj_block_vertices:
        continue
      for block_vertex in all_blocks[block].vertices:
        if block_vertex in x_adjacency[vertex]:
          all_blocks[block].add_border_vertex(vertex)
          continue

  # create the appropriate data structures for the remaining vertices
  border_vertex_structures = {vertex : None for vertex in border_vertices_extended}
  for vertex in border_vertices_extended:
    border_vertex_structures[vertex] = BorderVertexData(vertex, x_adjacency, border_vertices_extended, all_blocks)

  border_vertex_added = {i: False for i in border_vertex_structures}
  # size of the border after moving a vertex in
  border_size = len(border_vertex_added) - 1

  if print_debug:
    print("Before removal and update")
    for i in all_blocks:
      all_blocks[i].print_data()
      print()
    print("Border Vertices")
    for j in border_vertex_structures:
      if not border_vertex_added[j]:
        border_vertex_structures[j].print_data()
        print()
    print()
    print()
  iteration = 0
  while True:
    iteration += 1
    #print(iteration)
    # iteration += 1
    # if iteration == 5:
    #   sys.exit(0)
    vertices_in_blocks = 0
    for i in all_blocks:
      vertices_in_blocks += len(all_blocks[i].vertices)
    border_vertices = border_size + 1
    if vertices_in_blocks + border_vertices != len(x_adjacency):
      print("Not all accounted for!!!")
      print("Vertices in blocks = ", vertices_in_blocks)
      print("Border vertices = ", border_vertices)
      sys.exit(0)

    # implement stage 2 algorithm to determine final border
    # this algorithm goes through the border vertices one at a time (greedily)
    # and adds them to blocks until adding a vertex to a block will achieve a
    # larger block than the border. Then the algorithm terminates

    # vertex_to_add = sorted([border_vertex_structures[i] for i in border_vertex_structures\
    #                         if not border_vertex_added[i]], key=BorderVertexData.get_total_num_adj_block)[0].label
    vertex_to_add = sorted([border_vertex_structures[i] for i in border_vertex_structures\
                            if not border_vertex_added[i]])[0].label
    if print_debug:
      print("Add vertex = ", vertex_to_add)
    # print("vertex to add = ", vertex_to_add)
    # print("S_i = ", border_vertex_structures[vertex_to_add].total_num_adj_block)
    # print("Minimum S_i = ", min([border_vertex_structures[i].total_num_adj_block for i in border_vertex_structures if not border_vertex_added[i]]))

    # check if adding the vertex will create a larger block
    new_size_of_block = border_vertex_structures[vertex_to_add].total_num_adj_block + 1

    if print_debug:
      print("Size of newly formed block = ", new_size_of_block)
      print("Border size = ", border_size)

    # this is the base case -> change this to the size of the border is some fraction of the original size
    fraction_border = border_size / size_problem
    if fraction_border < fraction_tol:
      break
    # if new_size_of_block > border_size:
    #   break
    if print_debug:
      print(new_size_of_block)

    if new_size_of_block == 1:
      block_label += 1
      # this means that the vertex selected is not adjacent to any block
      # in this case, we have to create a new block and add the new block
      new_block = Block(vertex_to_add, block_label)
      vertex_to_block_map[vertex_to_add] = block_label
      new_block.update_block(border_vertex_structures[vertex_to_add])
      all_blocks[block_label] = new_block

      # determine all of the structures to be updated
      vertices_to_update = set()
      vertices_to_add_blocks = set()

      # add vertices in adjacency list of vertex added
      for vertex in border_vertex_structures[vertex_to_add].adj_border_vert_contains:
        vertices_to_update.add(vertex)

      for vertex in vertices_to_update:
        if not border_vertex_added[vertex] or vertex == vertex_to_add:
          #if vertex_to_add in border_vertex_structures[vertex].adj_border_vert:
          border_vertex_structures[vertex].update_data_blocks(border_vertex_structures[vertex_to_add], [block_label], new_block, all_blocks)
      # block_label += 1
      border_vertex_added[vertex_to_add] = True
      border_size -= 1

      if print_debug:
        print("Blocks after removal and update")
        for i in all_blocks:
          all_blocks[i].print_data()
          print()
      if print_debug:
        print("Border Vertices after removal and update")
        for j in border_vertex_structures:
          if not border_vertex_added[j]:
            border_vertex_structures[j].print_data()
            print()
            if border_vertex_structures[j].total_num_adj_block < 0:
              sys.exit(0)
            if sum([all_blocks[i].size for i in border_vertex_structures[j].adj_block]) != \
              border_vertex_structures[j].total_num_adj_block:
              print("Adjacent block number did not update correctly")
              print("Sum of blocks = ", sum([all_blocks[i].size for i in border_vertex_structures[j].adj_block]))
              print("S_i = ", border_vertex_structures[j].total_num_adj_block)
        print()
        print()

      continue


    # determine which blocks will be merged
    merged_blocks = []

    block_to_merge_into = next(iter(border_vertex_structures[vertex_to_add].adj_block))
    for block_label in border_vertex_structures[vertex_to_add].adj_block:
      if block_label != block_to_merge_into:
        merged_blocks.append(block_label)

    # determine all of the structures to be updated
    vertices_to_update = set()
    vertices_to_add_blocks = set()

    # add vertices in adjacency list of vertex added
    for vertex in border_vertex_structures[vertex_to_add].adj_border_vert_contains:
      vertices_to_update.add(vertex)

    # add all of the adjacent vertices from the merged blocks to the set
    for block_label in merged_blocks + [block_to_merge_into]:
      for vertex in all_blocks[block_label].adj_block_vertices:
        if vertex not in vertices_to_update:
          vertices_to_add_blocks.add(vertex)

    if print_debug:
      print("VERTICES TO UPDATE")
      print(vertices_to_update)
      print()

    # now actually merge the blocks
    # gives the actual block
    dominant_block = copy.deepcopy(all_blocks[block_to_merge_into])
    for block in merged_blocks:
      dominant_block.merge_block(all_blocks[block])
      #del all_blocks[block]

    # remove the added vertex from the adjacency list of the border and add the new adjacent vertices
    dominant_block.update_block(border_vertex_structures[vertex_to_add])

    # add the vertex to the new block
    dominant_block.add_vertex(vertex_to_add)

    if print_debug:
      print("DOMINANT BLOCK DATA")
      print(dominant_block.print_data())
      print()



    # add the dominant block to
    merged_blocks.append(block_to_merge_into)

    if print_debug:
      print("Merging blocks ")
      print(merged_blocks)
      print()

    # update the border vertex data structures
    for vertex in vertices_to_update:
      # check if already added
      if border_vertex_added[vertex] or vertex == vertex_to_add:
        # print("SKIPPED")
        continue
      # print("UPDATED ", vertex)
      #border_vertex_structures[vertex].update_data(border_vertex_structures[vertex_to_add], merged_blocks, block_to_merge_into, all_blocks)
      border_vertex_structures[vertex].update_data(border_vertex_structures[vertex_to_add], merged_blocks, dominant_block, all_blocks)
    for vertex in vertices_to_add_blocks:
      # remove vertex from adjacency list and appropriately remove and add blocks
      if border_vertex_added[vertex] or vertex == vertex_to_add:
        # print("SKIPPED")
        continue
      # print("UPDATED ", vertex)
      #border_vertex_structures[vertex].update_data_blocks(border_vertex_structures[vertex_to_add], merged_blocks, block_to_merge_into, all_blocks)
      border_vertex_structures[vertex].update_data_blocks(border_vertex_structures[vertex_to_add], merged_blocks, dominant_block, all_blocks)

    merged_blocks.remove(block_to_merge_into)
    for block in merged_blocks:
      del all_blocks[block]

    all_blocks[block_to_merge_into] = dominant_block

    border_vertex_added[vertex_to_add] = True
    border_size -= 1

    if print_debug:
      print("After removal and update")
      for i in all_blocks:
        all_blocks[i].print_data()
        print()
      print("Border Vertices")
      for j in border_vertex_structures:
        if not border_vertex_added[j]:
          border_vertex_structures[j].print_data()
          # print()
          if border_vertex_structures[j].total_num_adj_block < 0:
              sys.exit(0)
          if sum([all_blocks[i].size for i in border_vertex_structures[j].adj_block]) != \
            border_vertex_structures[j].total_num_adj_block:
            print("Adjacent block number did not update correctly")
            print("Sum of blocks = ", sum([all_blocks[i].size for i in border_vertex_structures[j].adj_block]))
            print("S_i = ", border_vertex_structures[j].total_num_adj_block)
            sys.exit(0)
          for k in border_vertex_structures[j].adj_block:
            print("Block = ", k)
            print("Size = ", all_blocks[k].size)
          print("S_i = ", border_vertex_structures[j].total_num_adj_block)
          print()
      print()
      print()

  border_indices = [border_vertex_structures[i].label for i in border_vertex_structures if not border_vertex_added[i]]
  # get the order of the rows and columns
  order_columns = []
  for i in all_blocks:
    for k in all_blocks[i].vertices:
      order_columns.append(k)
  border_columns = [column for column in range(len(x_adjacency)) if column not in order_columns]
  final_columns = order_columns + border_columns
  return final_columns, all_blocks, border_indices

# import random
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy as sc
# def create_matrix(seed, size, fill):
#     random.seed(seed)
#     size_matrix = size
#     size_matrix_m = size_matrix
#     size_matrix_n = size_matrix
#     fill_matrix = fill # maximum possible fill per row before adding diagonal
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
#     # impose perfectly matched criteria
#     for i in range(size_matrix):
#       original_matrix[i][i] = 1
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

# def create_adj_list_from_matrix(permuted_matrix):
#   # x_adjacency = {i : [] for i in range(permuted_matrix.shape[1])}
#   # y_adjacency = {i : [] for i in range(permuted_matrix.shape[0])}
#   overall_adjacency = {i : [] for i in range(permuted_matrix.shape[0])}
#   for i, j in zip(*permuted_matrix.nonzero()):
#     if i not in overall_adjacency[j]:
#       overall_adjacency[j].append(i)
#     if i != j and j not in overall_adjacency[i]:
#       overall_adjacency[i].append(j)
#   #overall_adjacency = {i : [j for j in range(max(permuted_matrix.shape[1], permuted_matrix.shape[0])) if (j in x_adjacency[i] or j in y_adjacency[i])] for i in x_adjacency} 
#   return overall_adjacency

# def get_adjacency_size(adjacency_list):
#   return {i : len(adjacency_list[i]) for i in adjacency_list}


# original_matrix = np.array([[0,0],[0,0]])
# print(original_matrix)
# show_matrix_structure(original_matrix)
# x_adjacency, x_size, d_max, n_max, fraction_tol

# size_matrix = 100
# fill = 3
# fraction = 0.1
# original_matrix = create_matrix(40, size_matrix, fill)
# adjacency_list = create_adj_list_from_matrix(original_matrix)
# final_cols, all_blocks, border_indices = bbbd_algo(adjacency_list, get_adjacency_size(adjacency_list), 1, 2, fraction)
# print([all_blocks[block].size for block in all_blocks])
# print("fraction = ", fraction)
# print("actual = ", len(border_indices)/size_matrix)
# reordered_incidence_matrix = reorder_sparse_matrix(len(final_cols),
#     len(final_cols), final_cols, final_cols, original_matrix)
# show_matrix_structure(reordered_incidence_matrix)

