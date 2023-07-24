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

### DATA FOR TEST 1
data_vertex_iter_0 = [[[],[],[],[],[],[],[],[]], \
                    [[1,6,7],[0,5,7],[4,7],[0,2,3],[2,4],[0,2],[5],[5,6]],\
                    [{1:0, 6:0, 7:0},{0:0, 5:0, 7:0}, {4:0, 7:0}, {0:0, 2:0, 3:0}, {2:0, 4:0}, {0:0, 2:0}, {5:0}, {5:0, 6:0}],\
                    [1,1,1,1,1,1,1,1],\
                    [3,3,2,3,2,2,1,2],\
                    [1, None, None, 3, None, None, None, None]]




data_constr_iter_0 = [[[],[],[],[],[],[],[],[]], \
                     [0,0,0,0,0,0,0,0], \
                     [[1,3,5], [0], [3,4,5], [3], [2,4], [1,6,7], [0,7], [0,1,2]],\
                     [3,1,3,1,2,3,2,3]]



data_blocks_iter_0 = []
data_blocks_iter_1 = [[0, [6], [5], 1, [1,7], []]]

data_ss_iter_0 = [{1: [0,1,2,3,4,5,6,7]}, 1, 1, [6,2,4,5,7,0,1,3]] 
                


def assert_data_vertices(computed, data_vertex, data_blocks):
    # make objects out of data and then compare
    expected = [VarVertex(i) for i in range(len(data_vertex[0]))]
    for i in range(len(data_vertex[0])):
        for j in data_vertex[0][i]:
            # print(i,j, [k[3] for k in data_blocks if k[0] == j])
            expected[i].add_adj_block(j,[k[3] if k[0] == j else 0 for k in data_blocks][0])
        for j in data_vertex[1][i]:
            expected[i].add_adj_constr(j, 0)
        expected[i].constr_size = data_vertex[2][i]
        expected[i].size_block = data_vertex[3][i]
        expected[i].size_constraints = data_vertex[4][i]
        expected[i].single_constraint = data_vertex[5][i]
    for i in range(len(expected)):
        assert expected[i] == computed[i]
    #assert all(expected[i] == computed[i] for i in range(len(expected)))


def assert_data_constr(computed, data_constr):
    expected = [ConstrVertex(i) for i in range(len(data_constr[0]))]
    for i in range(len(data_constr[0])):
        for j in data_constr[0][i]:
            expected[i].add_adj_block(j, 1)
        expected[i].size_block = data_constr[1][i]
        for j in data_constr[2][i]:
            expected[i].add_adj_var(j)
        expected[i].size_variables = data_constr[3][i]
    for i in range(len(computed)):
        assert expected[i] == computed[i]

def assert_data_blocks(computed, data_blocks):
    blocks = {}
    for i in range(len(data_blocks)):
        block = Block(data_blocks[i][0])
        for j in data_blocks[i][1]:
            block.add_var(j)
        for j in data_blocks[i][2]:
            block.add_constr(j)
        block.size = data_blocks[i][3]
        for j in data_blocks[i][4]:
            block.add_adj_var(j)
        for j in data_blocks[i][5]:
            block.add_adj_constr(j)
        blocks[block.label] = block 
    
    for i in blocks:
        assert blocks[i] == computed[i]
    for i in computed:
        assert blocks[i] == computed[i]
    
    









