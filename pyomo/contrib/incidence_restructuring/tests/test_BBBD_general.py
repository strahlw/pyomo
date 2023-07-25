from pyomo.contrib.incidence_restructuring.BBBD_general import *
import pytest
# @pytest.mark.parametrize(
#         "p1, p2, expected_distance",
#         [
#             (np.array([0.0,0.0,0.0]), 
#              np.array([0.0,0.0,1.0]),
#              1.0),
#              (np.array([0.0,0.0,0.0]), 
#              np.array([0.0,0.0,5.0]),
#              5.0),
#              (np.array([0.0,0.0,0.0]), 
#              np.array([0.0,0.0,25.0]),
#              25.0)
#         ]
# )
# def test_calculate_distance_many(p1, p2, expected_distance):
#     calculated_distance = molecool.calculate_distance(p1, p2)
#     assert np.isclose(calculated_distance, expected_distance, 1e-2)

import pyomo.common.unittest as unittest

       
class TestGenericDataStructure(unittest.TestCase):
    def test_add(self):
        test = GenericDataStructure()
        test.add(4)
        test.add(5)
        test.add(4)
        
        # test add
        assert 4 in test
        assert 5 in test
        assert test.size() == 2

    def test_remove(self):
        test = GenericDataStructure()
        test.add(4)
        test.add(5)

        # test remove
        test.remove(5)

        assert 5 not in test
        assert test.size() == 1

        test.remove(4)

        assert 4 not in test 
        assert test.size() == 0
        
class TestVertex(unittest.TestCase):
    def test_add_adj_block(self):
        test = Vertex(0)
        
        # test add_adj_constr
        test.add_adj_block(1, 2)

        assert 1 in test.adj_blocks
        assert test.adj_blocks.size() == 1
        assert test.size_block == 2

        # test remove adj_constr

    def test_remove_adj_block(self):
        test = Vertex(0)
        test.add_adj_block(1,2)
        test.add_adj_block(2,3)

        test.remove_adj_block(2,3)

        assert 1 in test.adj_blocks
        assert 2 not in test.adj_blocks
        assert test.adj_blocks.size() == 1
        assert test.size_block == 2


class TestVarVertex(unittest.TestCase):
    def test_add_adj_constr(self):
        test = VarVertex(0)

        test.add_adj_constr(0,0)
        test.add_adj_constr(0,0)

        assert 0 in test.adj_constr
        assert test.adj_constr.size() == 1
        assert test.size_constraints == 1
        assert test.constr_size[0] == 0

    def test_remove_adj_constr(self):
        test = VarVertex(0)

        test.add_adj_constr(0,0)
        test.add_adj_constr(1,0)

        test.remove_adj_constr(0)
        test.remove_adj_constr(0)

        assert 0 not in test.adj_constr
        assert 1 in test.adj_constr
        assert test.adj_constr.size() == 1
    
    def test_update_constr_size(self):
        test = VarVertex(0)

        test.add_adj_constr(0,0)
        test.add_adj_constr(1,0)

        test.update_constr_size(0,3)
        
        assert test.constr_size[0] == 3

    def test_set_single_constraint(self):
        test = VarVertex(0)

        assert test.single_constraint == None

        test.set_single_constraint(4)

        assert test.single_constraint == 4

        if test.single_constraint is not None:
            constr_assigned = test.single_constraint

        assert constr_assigned == 4 
    


class TestConstrVertex(unittest.TestCase):
    def test_add_adj_var(self):
        test = ConstrVertex(0)

        test.add_adj_var(0)
        test.add_adj_var(0)

        assert 0 in test.adj_var
        assert test.adj_var.size() == 1
        assert test.size_variables == 1

    def test_remove_adj_var(self):
        test = ConstrVertex(0)

        test.add_adj_var(0)
        test.add_adj_var(1)

        test.remove_adj_var(0)
        assert 0 not in test.adj_var
        assert 1 in test.adj_var

        test.remove_adj_var(1)
        assert 1 not in test.adj_var

        test.remove_adj_var(0)
    


class TestBlock(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def make_block(self):
        self.block = Block(0)
    
    def test_add_var_constr(self):
        self.block.add_var(1)
        self.block.add_constr(1)

        assert 1 in self.block.var
        assert 1 in self.block.constr
        assert self.block.size == 1

    def test_add_var_constr_comb(self):
        self.block.add_var_constr(1,1)
        assert 1 in self.block.var
        assert 1 in self.block.constr
        assert self.block.size == 1
    
    def add_adj_var(self):
        self.block.add_adj_var(1)
        assert 1 in self.block.adj_var
        assert self.block.adj_var.size() == 1
        self.block.add_adj_var(0)
        assert 0 in self.block.adj_var
        assert self.block.adj_var.size() == 2
        self.block.add_adj_var(1)
        assert 0 in self.block.adj_var
        assert self.block.adj_var.size() == 2

    def add_adj_constr(self):       
        self.block.add_adj_constr(1)
        assert 1 in self.block.adj_constr
        assert self.block.adj_constr.size() == 1
        self.block.add_adj_constr(0)
        assert 0 in self.block.adj_constr
        assert self.block.adj_constr.size() == 2
        self.block.add_adj_constr(1)
        assert 0 in self.block.adj_constr
        assert self.block.adj_constr.size() == 2

    def remove_adj_var(self):
        self.block.add_adj_var(1)
        self.block.add_adj_var(0)

        self.block.remove_adj_var(0)
        assert 0 not in self.block.adj_var
        assert self.block.adj_var.size() == 1

        self.block.remove_adj_var(1)
        assert 1 not in self.block.adj_var
        assert self.block.adj_var.size() == 0
    
        self.block.remove_adj_var(0)
        assert 1 not in self.block.adj_var
        assert self.block.adj_var.size() == 0

    def remove_adj_constr(self):
        self.block.add_adj_constr(1)
        self.block.add_adj_constr(0)

        self.block.remove_adj_constr(0)
        assert 0 not in self.block.adj_constr
        assert self.block.adj_constr.size() == 1

        self.block.remove_adj_constr(1)
        assert 1 not in self.block.adj_constr
        assert self.block.adj_constr.size() == 0
    
        self.block.remove_adj_constr(0)
        assert 1 not in self.block.adj_constr
        assert self.block.adj_constr.size() == 0

# is consistent with design, var vertex data structure entry must match label
class TestSortingStructure(object):
    @pytest.fixture(autouse=True)
    def set_up_data(self):
        self.var_vertices = GenericDataStructure()
        self.var_vertices[0] = VarVertex(0)
        self.var_vertices[0].size_constraints = 3
        self.var_vertices[1] = VarVertex(1)
        self.var_vertices[1].size_constraints = 2

        self.sorting_structure = SortingStructure(10,0.9)
    
    @pytest.fixture
    def test_add_data(self):
        self.sorting_structure.add_data(1, self.var_vertices[0].label)
        self.sorting_structure.add_data(1, self.var_vertices[1].label)

        assert self.var_vertices[0].label in self.sorting_structure.data[1]
        assert self.var_vertices[1].label in self.sorting_structure.data[1]
    
    def test_remove_data(self, test_add_data):
        self.sorting_structure.remove_data(1, self.var_vertices[0].label)
        assert self.var_vertices[0].label not in self.sorting_structure.data[1]
    
    @pytest.fixture
    def test_create_current_order(self, test_add_data):
        self.sorting_structure.create_current_order()
        assert 0 in self.sorting_structure.current_order
        assert 1 in self.sorting_structure.current_order
        assert len(self.sorting_structure.current_order) == 2

    def test_sorting_function(self, test_add_data, test_create_current_order):
        assert self.sorting_structure.current_order[0] == 0
        assert self.sorting_structure.current_order[1] == 1

        self.sorting_structure.sorting_function(self.var_vertices)

        assert self.sorting_structure.current_order[0] == 1
        assert self.sorting_structure.current_order[1] == 0

    def test_remove_from_current_order(self, test_add_data, test_create_current_order):
        # if not in current order, will throw an error
        self.sorting_structure.remove_from_current_order(0)
        assert len(self.sorting_structure.current_order) == 2
        assert 1 in self.sorting_structure.current_order
    
    def increment_current_key(self):
        self.sorting_structure.add_data(1, self.var_vertices[0].label)
        self.sorting_structure.add_data(3, self.var_vertices[1].label)
        self.sorting_structure.remove_data(1, self.var_vertices[0].label)

        assert self.current_key == 1
        self.sorting_structure.increment_current_key()
        assert self.current_key == 3

                

import numpy as np 
import scipy as sc
import matplotlib.pyplot as plt
import random
import sys


def create_matrix(seed, size, fraction):
    random.seed(seed)
    size_matrix = size
    size_matrix_m = size_matrix
    size_matrix_n = size_matrix
    fill_matrix = fraction # maximum possible fill per row before adding diagonal
    original_matrix = np.zeros((size_matrix, size_matrix))
    for i in range(size_matrix):
        # for each row select a number of indices to make nonzero
        num_non_zeros = random.randint(0,int((size_matrix-1)*fill_matrix/100))
        indices_used = []
        for j in range(num_non_zeros):
            # for each non zero, randomly choose an index (and keep track)
            if j == 0:
                index_to_fill = random.randint(0, size_matrix-1)
                original_matrix[i][index_to_fill] = 1
                indices_used.append(index_to_fill)
            else:
                index_to_fill = random.randint(0, size_matrix-1)
                while index_to_fill in indices_used:
                    index_to_fill = random.randint(0, size_matrix-1)
                original_matrix[i][index_to_fill] = 1
                indices_used.append(index_to_fill)
    return original_matrix

def matrix_to_edges(matrix):
    edges = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                edges.append((i,j))
    return edges 


def test_random_matrices_10_100_05():
    # make sure that the assertions are enabled (self.check_assertions = True in BBBD_algo)
    seeds = range(100)
    for i in seeds:
        #print("Instance ", i)
        original_matrix = create_matrix(i, 10, 60)
        test = BBBD_algo(matrix_to_edges(original_matrix), len(original_matrix), len(original_matrix[0]), 0.5)
        col_order, row_order, blocks = test.solve()


def test_random_matrices_20_100_07():
    # make sure that the assertions are enabled (self.check_assertions = True in BBBD_algo)
    seeds = range(100)
    for i in seeds:
        #print("Instance ", i)
        original_matrix = create_matrix(i, 20, 60)
        test = BBBD_algo(matrix_to_edges(original_matrix), len(original_matrix), len(original_matrix[0]), 0.7)
        col_order, row_order, blocks = test.solve()
    
    









