from pyomo.contrib.incidence_restructuring.BBBD_decomposition import (
    Block, GenericDataStructure, Vertex, VarVertex, ConstrVertex,
    SortingStructure, BBBD_algo

)
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

        self.sorting_structure = SortingStructure(10)
    
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
        assert len(self.sorting_structure.current_order) == 1
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
data_vertex_iter_1 = [[[],[0],[],[],[],[],[],[0]], \
                    [[1,6,7],[0,7],[4,7],[0,2,3],[2,4],[0,2],[],[6]],\
                    [{1:0, 6:0, 7:0},{0:0, 7:0}, {4:0, 7:0}, {0:0, 2:0, 3:0}, {2:0, 4:0}, {0:0, 2:0}, {}, {6:0}],\
                    [1,2,1,1,1,1,1,2],\
                    [3,2,2,3,2,2,0,1],\
                    [1, None, None, 3, None, None, None, None]]
data_vertex_iter_2 = [[[],[0],[],[],[1],[],[],[0]], \
                    [[1,6,7],[0,7],[7],[0,2,3],[2],[0,2],[],[6]],\
                    [{1:0, 6:0, 7:1},{0:0, 7:1}, {7:0}, {0:0, 2:0, 3:0}, {2:0}, {0:0, 2:0}, {}, {6:0}],\
                    [1,2,1,1,2,1,1,2],\
                    [3,2,1,3,1,2,0,1],\
                    [1, None, None, 3, None, None, None, None]]




data_constr_iter_0 = [[[],[],[],[],[],[],[],[]], \
                     [0,0,0,0,0,0,0,0], \
                     [[1,3,5], [0], [3,4,5], [3], [2,4], [1,6,7], [0,7], [0,1,2]],\
                     [3,1,3,1,2,3,2,3]]
data_constr_iter_1 = [[[],[],[],[],[],[],[],[]], \
                     [0,0,0,0,0,0,0,0], \
                     [[1,3,5], [0], [3,4,5], [3], [2,4], [1,7], [0,7], [0,1,2]],\
                     [3,1,3,1,2,2,2,3]]
data_constr_iter_2 = [[[],[],[],[],[],[],[],[1]], \
                     [0,0,0,0,0,0,0,1], \
                     [[1,3,5], [0], [3,4,5], [3], [4], [1,7], [0,7], [0,1]],\
                     [3,1,3,1,1,2,2,2]]



data_blocks_iter_0 = []
data_blocks_iter_1 = [[0, [6], [5], 1, [1,7], []]]
data_blocks_iter_2 = [[0, [6], [5], 1, [1,7], []], [1, [2], [4], 1, [4], [7]]]

data_ss_iter_0 = [{1: [0,1,2,3,4,5,6,7]}, 1, 1, [6,2,4,5,7,0,1,3]] 
data_ss_iter_1 = [{1: [0,2,3,4,5], 2: [1,7]}, 1, 1, [2,4,5,0,3]]
data_ss_iter_2 = [{1: [0,3,5], 2: [1,4,7]}, 1, 1, [5,0,3]]
                
# iteration 3 data - you do delete the mutual information, but stop updating after that
data_vertex_iter_3 = [[[],[0,2],[],[2],[1],[],[],[0]], \
                    [[1,6,7],[7],[7],[2,3],[2],[2],[],[6]],\
                    [{1:0, 6:0, 7:1},{7:1}, {7:0}, {2:0, 3:0}, {2:1}, {2:0}, {}, {6:0}],\
                    [1,3,1,2,2,1,1,2],\
                    [3,1,1,2,1,1,0,1],\
                    [1, None, None, 3, None, None, None, None]]
data_constr_iter_3 = [[[],[],[2],[],[],[],[],[1]], \
                     [0,0,1,0,0,0,0,1], \
                     [[1,3], [0], [3,4], [3], [4], [1,7], [0,7], [0,1]],\
                     [2,1,2,1,1,2,2,2]]
data_blocks_iter_3 = [[0, [6], [5], 1, [1,7], []], [1, [2], [4], 1, [4], [7]],\
                      [2, [5], [0], 1, [1,3], [2]]]
data_ss_iter_3 = [{1: [0], 2: [3,4,7], 4:[1]}, 1, 1, [0]]


data_vertex_iter_4 = [[[],[0,2],[],[2],[1],[],[],[0]], \
                    [[6,7],[7],[7],[2,3],[2],[2],[],[6]],\
                    [{6:0, 7:1},{7:2}, {7:0}, {2:0, 3:0}, {2:1}, {2:0}, {}, {6:1}],\
                    [1,3,1,2,2,1,1,2],\
                    [2,1,1,2,1,1,0,1],\
                    [1, 7, None, 3, None, None, None, 6]]
data_constr_iter_4 = [[[],[],[2],[],[],[],[3],[1,3]], \
                     [0,0,1,0,0,0,1,2], \
                     [[1,3], [], [3,4], [3], [4], [1,7], [7], [1]],\
                     [2,0,2,1,1,2,1,1]]
data_blocks_iter_4 = [[0, [6], [5], 1, [1,7], []], [1, [2], [4], 1, [4], [7]],\
                      [2, [5], [0], 1, [1,3], [2]], [3, [0], [1], 1, [], [6,7]]]
data_ss_iter_4 = [{1: [], 2: [3,4,7], 4:[1]}, 2, 2, [7,4,3]]


data_vertex_iter_5 = [[[],[0,2],[],[2],[1],[],[],[0]], \
                    [[6,7],[7],[7],[2,3],[2],[2],[],[]],\
                    [{6:0, 7:1},{7:1}, {7:0}, {2:0, 3:0}, {2:1}, {2:0}, {}, {}],\
                    [1,5,1,2,2,1,1,2],\
                    [2,1,1,2,1,1,0,0],\
                    [1, 7, None, 3, None, None, None, 6]]
data_constr_iter_5 = [[[],[],[2],[],[],[],[3],[0,1]], \
                     [0,0,1,0,0,0,1,4], \
                     [[1,3], [], [3,4], [3], [4], [1,7], [], [1]],\
                     [2,0,2,1,1,2,0,1]]
data_blocks_iter_5 = [[0, [6,0,7], [5,1,6], 3, [1], [7]],\
                      [1, [2], [4], 1, [4], [7]],\
                      [2, [5], [0], 1, [1,3], [2]]]
data_ss_iter_5 = [{1: [], 2: [3,4], 4:[], 6:[1]}, 2, 2, [4,3]]


data_vertex_iter_6 = [[[],[0,1],[],[1],[1],[],[],[0]], \
                    [[6,7],[7],[7],[3],[],[2],[],[]],\
                    [{6:0, 7:1},{7:0}, {7:0}, {3:0}, {}, {2:0}, {}, {}],\
                    [1,7,1,4,2,1,1,2],\
                    [2,1,1,1,0,1,0,0],\
                    [1, 7, None, 3, None, None, None, 6]]
data_constr_iter_6 = [[[],[],[2],[],[],[],[3],[0,1]], \
                     [0,0,1,0,0,0,1,6], \
                     [[1,3], [], [3], [3], [4], [1,7], [], [1]],\
                     [2,0,1,1,1,2,0,1]]
data_blocks_iter_6 = [[0, [6,0,7], [6,5,1], 3, [1], [7]], [1, [4,2,5], [2,4,0], 3, [1,3], [7]]]
data_ss_iter_6 = [{1: [], 2: [], 4:[3], 6:[], 7:[1]}, 4, 4, [3]]

data_vertex_iter_7 = [[[],[0,1],[],[1],[1],[],[],[0]], \
                    [[6,7],[7],[7],[],[],[2],[],[]],\
                    [{6:0, 7:1},{7:0}, {7:0}, {}, {}, {2:0}, {}, {}],\
                    [1,8,1,4,2,1,1,2],\
                    [2,1,1,0,0,1,0,0],\
                    [1, 7, None, 3, None, None, None, 6]]
data_constr_iter_7 = [[[],[],[2],[],[],[],[3],[0,1]], \
                     [0,0,1,0,0,0,1,7], \
                     [[1,3], [], [3], [], [4], [1,7], [], [1]],\
                     [2,0,1,0,1,2,0,1]]
data_blocks_iter_7 = [[0, [7,6,0], [6,5,1], 3, [1], [7]],\
                      [1, [3,4,2,5], [3,2,4,0], 4, [1], [7]]]
data_ss_iter_7 = [{1: [], 2: [], 4:[], 6:[], 7:[], 8:[1]}, 8, 8, [1]]

data_vertex_iter_8 = [[[],[0,1],[],[1],[1],[],[],[0]], \
                    [[6,7],[],[7],[],[],[2],[],[]],\
                    [{6:0, 7:1},{}, {7:0}, {}, {}, {2:0}, {}, {}],\
                    [1,8,1,4,2,1,1,2],\
                    [2,0,1,0,0,1,0,0],\
                    [1, 7, None, 3, None, None, None, 6]]
data_constr_iter_8 = [[[],[],[2],[],[],[],[3],[0,1]], \
                     [0,0,1,0,0,0,1,7], \
                     [[1,3], [], [3], [], [4], [1,7], [], []],\
                     [2,0,1,0,1,2,0,0]]
data_blocks_iter_8 = [[1, [1,7,6,0,3,4,2,5], [7,6,5,1,3,2,4,0], 8, [], []]]
data_ss_iter_8 = [{1: [], 2: [], 4:[], 6:[], 7:[], 8:[1]}, 8, 8, []]


#### DATA FOR TEST 2 - selection criteria refinement
data_vertex_iter_0_2 = [[[],[],[],[],[],[],[],[]], \
                    [[1,6,7],[0,5,7],[4,7],[0,2,3],[2,4],[0,2],[5],[5,6]],\
                    [{1:0, 6:0, 7:0},{0:0, 5:0, 7:0}, {4:0, 7:0}, {0:0, 2:0, 3:0}, {2:0, 4:0}, {0:0, 2:0}, {5:0}, {5:0, 6:0}],\
                    [1,1,1,1,1,1,1,1],\
                    [3,3,2,3,2,2,1,2],\
                    [1, None, None, 3, None, None, None, None]]
data_vertex_iter_1_2 = [[[],[0],[],[],[],[],[],[0]], \
                    [[1,6,7],[0,7],[4,7],[0,2,3],[2,4],[0,2],[],[6]],\
                    [{1:0, 6:0, 7:0},{0:0, 7:0}, {4:0, 7:0}, {0:0, 2:0, 3:0}, {2:0, 4:0}, {0:0, 2:0}, {}, {6:0}],\
                    [1,2,1,1,1,1,1,2],\
                    [3,2,2,3,2,2,0,1],\
                    [1, None, None, 3, None, None, None, None]]
data_vertex_iter_2_2 = [[[],[0],[],[],[1],[],[],[0]], \
                    [[1,6,7],[0,7],[7],[0,2,3],[2],[0,2],[],[6]],\
                    [{1:0, 6:0, 7:1},{0:0, 7:1}, {7:0}, {0:0, 2:0, 3:0}, {2:0}, {0:0, 2:0}, {}, {6:0}],\
                    [1,2,1,1,2,1,1,2],\
                    [3,2,1,3,1,2,0,1],\
                    [1, None, None, 3, None, None, None, None]]




data_constr_iter_0_2 = [[[],[],[],[],[],[],[],[]], \
                     [0,0,0,0,0,0,0,0], \
                     [[1,3,5], [0], [3,4,5], [3], [2,4], [1,6,7], [0,7], [0,1,2]],\
                     [3,1,3,1,2,3,2,3]]
data_constr_iter_1_2 = [[[],[],[],[],[],[],[],[]], \
                     [0,0,0,0,0,0,0,0], \
                     [[1,3,5], [0], [3,4,5], [3], [2,4], [1,7], [0,7], [0,1,2]],\
                     [3,1,3,1,2,2,2,3]]
data_constr_iter_2_2 = [[[],[],[],[],[],[],[],[1]], \
                     [0,0,0,0,0,0,0,1], \
                     [[1,3,5], [0], [3,4,5], [3], [4], [1,7], [0,7], [0,1]],\
                     [3,1,3,1,1,2,2,2]]



data_blocks_iter_0_2 = []
data_blocks_iter_1_2 = [[0, [6], [5], 1, [1,7], []]]
data_blocks_iter_2_2 = [[0, [6], [5], 1, [1,7], []], [1, [2], [4], 1, [4], [7]]]

data_ss_iter_0_2 = [{1: [0,1,2,3,4,5,6,7]}, 1, 1, [6,2,4,5,7,0,1,3]] 
data_ss_iter_1_2 = [{1: [0,2,3,4,5], 2: [1,7]}, 1, 1, [2,4,5,0,3]]
data_ss_iter_2_2 = [{1: [0,3,5], 2: [1,4,7]}, 1, 1, [5,0,3]]
                
# iteration 3 data - you do delete the mutual information, but stop updating after that
data_vertex_iter_3_2 = [[[],[0,2],[],[2],[1],[],[],[0]], \
                    [[1,6,7],[7],[7],[2,3],[2],[2],[],[6]],\
                    [{1:0, 6:0, 7:1},{7:1}, {7:0}, {2:0, 3:0}, {2:1}, {2:0}, {}, {6:0}],\
                    [1,3,1,2,2,1,1,2],\
                    [3,1,1,2,1,1,0,1],\
                    [1, None, None, 3, None, None, None, None]]
data_constr_iter_3_2 = [[[],[],[2],[],[],[],[],[1]], \
                     [0,0,1,0,0,0,0,1], \
                     [[1,3], [0], [3,4], [3], [4], [1,7], [0,7], [0,1]],\
                     [2,1,2,1,1,2,2,2]]
data_blocks_iter_3_2 = [[0, [6], [5], 1, [1,7], []], [1, [2], [4], 1, [4], [7]],\
                      [2, [5], [0], 1, [1,3], [2]]]
data_ss_iter_3_2 = [{1: [0], 2: [3,4,7], 4:[1]}, 1, 1, [0]]


data_vertex_iter_4_2 = [[[],[0,2],[],[2],[1],[],[],[0]], \
                    [[6,7],[7],[7],[2,3],[2],[2],[],[6]],\
                    [{6:0, 7:1},{7:2}, {7:0}, {2:0, 3:0}, {2:1}, {2:0}, {}, {6:1}],\
                    [1,3,1,2,2,1,1,2],\
                    [2,1,1,2,1,1,0,1],\
                    [1, 7, None, 3, None, None, None, 6]]
data_constr_iter_4_2 = [[[],[],[2],[],[],[],[3],[1,3]], \
                     [0,0,1,0,0,0,1,2], \
                     [[1,3], [], [3,4], [3], [4], [1,7], [7], [1]],\
                     [2,0,2,1,1,2,1,1]]
data_blocks_iter_4_2 = [[0, [6], [5], 1, [1,7], []], [1, [2], [4], 1, [4], [7]],\
                      [2, [5], [0], 1, [1,3], [2]], [3, [0], [1], 1, [], [6,7]]]
data_ss_iter_4_2 = [{1: [], 2: [3,4,7], 4:[1]}, 2, 2, [7,4,3]]


data_vertex_iter_5_2 = [[[],[2,4],[],[2],[1],[],[],[0]], \
                    [[6,7],[7],[7],[2,3],[2],[2],[],[]],\
                    [{6:0, 7:1},{7:1}, {7:0}, {2:0, 3:0}, {2:1}, {2:0}, {}, {}],\
                    [1,5,1,2,2,1,1,2],\
                    [2,1,1,2,1,1,0,0],\
                    [1, 7, None, 3, None, None, None, 6]]
data_constr_iter_5_2 = [[[],[],[2],[],[],[],[3],[1,4]], \
                     [0,0,1,0,0,0,1,4], \
                     [[1,3], [], [3,4], [3], [4], [1,7], [], [1]],\
                     [2,0,2,1,1,2,0,1]]
data_blocks_iter_5_2 = [[1, [2], [4], 1, [4], [7]],\
                      [2, [5], [0], 1, [1,3], [2]],\
                      [4, [7,6,0], [6,5,1], 3, [1], [7]]]
data_ss_iter_5_2 = [{1: [], 2: [3,4], 4:[], 6:[1]}, 2, 2, [4,3]]


data_vertex_iter_6_2 = [[[],[4,5],[],[5],[1],[],[],[0]], \
                    [[6,7],[7],[7],[3],[],[2],[],[]],\
                    [{6:0, 7:1},{7:0}, {7:0}, {3:0}, {}, {2:0}, {}, {}],\
                    [1,7,1,4,2,1,1,2],\
                    [2,1,1,1,0,1,0,0],\
                    [1, 7, None, 3, None, None, None, 6]]
data_constr_iter_6_2 = [[[],[],[2],[],[],[],[3],[4,5]], \
                     [0,0,1,0,0,0,1,6], \
                     [[1,3], [], [3], [3], [4], [1,7], [], [1]],\
                     [2,0,1,1,1,2,0,1]]
data_blocks_iter_6_2 = [[4, [7,6,0], [6,5,1], 3, [1], [7]], [5, [4,2,5], [2,4,0], 3, [1,3], [7]]]
data_ss_iter_6_2 = [{1: [], 2: [], 4:[3], 6:[], 7:[1]}, 4, 4, [3]]

data_vertex_iter_7_2 = [[[],[4,6],[],[5],[1],[],[],[0]], \
                    [[6,7],[7],[7],[],[],[2],[],[]],\
                    [{6:0, 7:1},{7:0}, {7:0}, {}, {}, {2:0}, {}, {}],\
                    [1,8,1,4,2,1,1,2],\
                    [2,1,1,0,0,1,0,0],\
                    [1, 7, None, 3, None, None, None, 6]]
data_constr_iter_7_2 = [[[],[],[2],[],[],[],[3],[4,6]], \
                     [0,0,1,0,0,0,1,7], \
                     [[1,3], [], [3], [], [4], [1,7], [], [1]],\
                     [2,0,1,0,1,2,0,1]]
data_blocks_iter_7_2 = [[4, [7,6,0], [6,5,1], 3, [1], [7]],\
                      [6, [3,4,2,5], [3,2,4,0], 4, [1], [7]]]
data_ss_iter_7_2 = [{1: [], 2: [], 4:[], 6:[], 7:[], 8:[1]}, 8, 8, [1]]

data_vertex_iter_8_2 = [[[],[4,6],[],[5],[1],[],[],[0]], \
                    [[6,7],[],[7],[],[],[2],[],[]],\
                    [{6:0, 7:1},{}, {7:0}, {}, {}, {2:0}, {}, {}],\
                    [1,8,1,4,2,1,1,2],\
                    [2,0,1,0,0,1,0,0],\
                    [1, 7, None, 3, None, None, None, 6]]
data_constr_iter_8_2 = [[[],[],[2],[],[],[],[3],[4,6]], \
                     [0,0,1,0,0,0,1,7], \
                     [[1,3], [], [3], [], [4], [1,7], [], []],\
                     [2,0,1,0,1,2,0,0]]
data_blocks_iter_8_2 = [[7, [1,7,6,0,3,4,2,5], [7,6,5,1,3,2,4,0], 8, [], []]]
data_ss_iter_8_2 = [{1: [], 2: [], 4:[], 6:[], 7:[], 8:[1]}, 8, 8, []]



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
    
    # assert all(blocks[i] == computed[i] for i in blocks)
    # assert all(computed[i] == blocks[i] for i in computed)

def assert_data_sorting(computed, data_sorting):
    expected = SortingStructure(8)
    for key in data_sorting[0]:
        expected.data[key] = GenericDataStructure()
        for value in data_sorting[0][key]:
            expected.add_data(key, value)
    expected.current_key = data_sorting[1]
    expected.previous_key = data_sorting[2]
    expected.current_order = data_sorting[3]
    assert expected == computed
    

class testProblem(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def set_up_problem(self):
        edges = [(0,1),(0,3), (0,5), (1,0), (2,3), (2,4), (2,5), (3,3), \
                (4,2), (4,4), (5,1), (5,6), (5,7), (6,0),(6,7), (7,0), (7,1), (7,2)]
        m = 8
        n = 8

        self.bbbd_algo = BBBD_algo(edges, m, n, 0.9)

    def test_initial_state(self):
        # check initial state 
        assert_data_vertices(self.bbbd_algo.variables, data_vertex_iter_0, data_blocks_iter_0)  
        assert_data_constr(self.bbbd_algo.constraints, data_constr_iter_0)
        assert_data_sorting(self.bbbd_algo.sorting_structure, data_ss_iter_0)
        assert_data_blocks(self.bbbd_algo.blocks, data_blocks_iter_0)
    
    def test_get_constraint_lowest_val(self):
        self.bbbd_algo.select_variable()
        assert self.bbbd_algo.selected_variable == 6
        self.bbbd_algo.get_constraint_lowest_val()
        assert self.bbbd_algo.selected_constraint == 5
    
    # def test_create_block(self):
    #     create_block(0,1,self.blocks, 0)
    #     assert 0 in self.blocks[0].var
    #     assert 1 in self.blocks[0].constr

    def select_var_constr_iter_0(self):
        self.selected_variable = self.bbbd_algo.select_variable()
        self.bbbd_algo.get_constraint_lowest_val()
        self.selected_constraint = self.bbbd_algo.selected_constraint
        # now we have to update all the data structures based on these two values
        # create a block - will figure out merges later
        self.bbbd_algo.create_block()
        self.bbbd_algo.remove_references()
        assert self.selected_variable == 6
        assert self.selected_constraint == 5

    def test_update_block(self):
        self.bbbd_algo.check_sorting_structure()
        self.bbbd_algo.select_variable()
        self.bbbd_algo.get_constraint_lowest_val()
        self.bbbd_algo.set_seed_block()
        self.bbbd_algo.remove_references()
        self.bbbd_algo.vars_constr_to_update()
        self.bbbd_algo.adjust_vars_sorting_structure()
        self.bbbd_algo.merge_blocks()
        self.bbbd_algo.update_block()
        assert_data_blocks(self.bbbd_algo.blocks, data_blocks_iter_1)
    
    # def test_update_var_vertex(self):
    #     self.bbbd_algo.select_variable()
    #     self.bbbd_algo.get_constraint_lowest_val()
    #     self.bbbd_algo.create_block() # increases the block label by 1
    #     self.bbbd_algo.remove_references()
    #     self.bbbd_algo.update_block(self.bbbd_algo.block_label-1)
    #     self.bbbd_algo.update_var_vertex(self.bbbd_algo.block_label-1)

    def test_update_var_constr_vertex(self):
        self.bbbd_algo.check_sorting_structure()
        self.bbbd_algo.select_variable()
        self.bbbd_algo.get_constraint_lowest_val()
        self.bbbd_algo.set_seed_block()
        self.bbbd_algo.remove_references()
        self.bbbd_algo.vars_constr_to_update()
        self.bbbd_algo.adjust_vars_sorting_structure()
        self.bbbd_algo.merge_blocks()
        self.bbbd_algo.update_block()
        self.bbbd_algo.merge_blocks()
        self.bbbd_algo.update_var_vertices()
        self.bbbd_algo.update_constr_vertices()
        assert_data_constr(self.bbbd_algo.constraints, data_constr_iter_1)
        assert_data_vertices(self.bbbd_algo.variables, data_vertex_iter_1, data_blocks_iter_1)


    def test_update_sorting_structure(self):
        self.bbbd_algo.check_sorting_structure()
        self.bbbd_algo.select_variable()
        self.bbbd_algo.get_constraint_lowest_val()
        self.bbbd_algo.set_seed_block()
        self.bbbd_algo.remove_references()
        self.bbbd_algo.vars_constr_to_update()
        self.bbbd_algo.adjust_vars_sorting_structure()
        self.bbbd_algo.merge_blocks()
        self.bbbd_algo.update_block()
        self.bbbd_algo.update_var_vertices()
        self.bbbd_algo.update_constr_vertices()
        self.bbbd_algo.update_var_sorting_structure()
        self.bbbd_algo.update_sorting_structure()
        assert_data_sorting(self.bbbd_algo.sorting_structure, data_ss_iter_1)

# def test_sort():
#     array = [[1,6],[3,5],[5,4]]
#     print(sorted(array, key = lambda x : (len(x), x[1])))   
#     assert False

class testProblemIteration2(unittest.TestCase):
    def test_algorithm(self):
        edges = [(0,1),(0,3), (0,5), (1,0), (2,3), (2,4), (2,5), (3,3), \
        (4,2), (4,4), (5,1), (5,6), (5,7), (6,0),(6,7), (7,0), (7,1), (7,2)]
        m = 8
        n = 8

        self.bbbd_algo = BBBD_algo(edges, m, n, 0.9)


        self.bbbd_algo.iteration()
        # check initial state 
        assert_data_vertices(self.bbbd_algo.variables, data_vertex_iter_1, data_blocks_iter_1)  
        assert_data_constr(self.bbbd_algo.constraints, data_constr_iter_1)
        assert_data_sorting(self.bbbd_algo.sorting_structure, data_ss_iter_1)
        assert_data_blocks(self.bbbd_algo.blocks, data_blocks_iter_1)

        self.bbbd_algo.iteration()
        assert_data_vertices(self.bbbd_algo.variables, data_vertex_iter_2, data_blocks_iter_2)  
        assert_data_constr(self.bbbd_algo.constraints, data_constr_iter_2)
        assert_data_sorting(self.bbbd_algo.sorting_structure, data_ss_iter_2)
        assert_data_blocks(self.bbbd_algo.blocks, data_blocks_iter_2)

        # iteration 3
        self.bbbd_algo.iteration()
        assert_data_vertices(self.bbbd_algo.variables, data_vertex_iter_3, data_blocks_iter_3)  
        assert_data_constr(self.bbbd_algo.constraints, data_constr_iter_3)
        assert_data_sorting(self.bbbd_algo.sorting_structure, data_ss_iter_3)
        assert_data_blocks(self.bbbd_algo.blocks, data_blocks_iter_3)

        # iteration 4
        self.bbbd_algo.iteration()
        assert_data_vertices(self.bbbd_algo.variables, data_vertex_iter_4, data_blocks_iter_4)  
        assert_data_constr(self.bbbd_algo.constraints, data_constr_iter_4)
        assert_data_sorting(self.bbbd_algo.sorting_structure, data_ss_iter_4)
        assert_data_blocks(self.bbbd_algo.blocks, data_blocks_iter_4)

        # iteration 5
        self.bbbd_algo.iteration()
        assert_data_vertices(self.bbbd_algo.variables, data_vertex_iter_5, data_blocks_iter_5)  
        assert_data_constr(self.bbbd_algo.constraints, data_constr_iter_5)
        assert_data_sorting(self.bbbd_algo.sorting_structure, data_ss_iter_5)
        assert_data_blocks(self.bbbd_algo.blocks, data_blocks_iter_5)

        # iteration 6
        self.bbbd_algo.iteration()
        assert_data_vertices(self.bbbd_algo.variables, data_vertex_iter_6, data_blocks_iter_6)  
        assert_data_constr(self.bbbd_algo.constraints, data_constr_iter_6)
        assert_data_sorting(self.bbbd_algo.sorting_structure, data_ss_iter_6)
        assert_data_blocks(self.bbbd_algo.blocks, data_blocks_iter_6)

        # iteration 7
        self.bbbd_algo.iteration()
        assert_data_vertices(self.bbbd_algo.variables, data_vertex_iter_7, data_blocks_iter_7)  
        assert_data_constr(self.bbbd_algo.constraints, data_constr_iter_7)
        assert_data_sorting(self.bbbd_algo.sorting_structure, data_ss_iter_7)
        assert_data_blocks(self.bbbd_algo.blocks, data_blocks_iter_7)

        # iteration 8
        self.bbbd_algo.iteration()
        # assert self.bbbd_algo.sorting_structure.terminate
        assert_data_vertices(self.bbbd_algo.variables, data_vertex_iter_8, data_blocks_iter_8)  
        assert_data_constr(self.bbbd_algo.constraints, data_constr_iter_8)
        assert_data_sorting(self.bbbd_algo.sorting_structure, data_ss_iter_8)
        assert_data_blocks(self.bbbd_algo.blocks, data_blocks_iter_8)

class testProblemTerminationCriteria(unittest.TestCase):
    def test_solve_termination(self):
        edges = [(0,1),(0,3), (0,5), (1,0), (2,3), (2,4), (2,5), (3,3), \
        (4,2), (4,4), (5,1), (5,6), (5,7), (6,0),(6,7), (7,0), (7,1), (7,2)]
        m = 8
        n = 8

        self.bbbd_algo = BBBD_algo(edges, m, n, 0.5)


        column_order, row_order, blocks = self.bbbd_algo.solve()
        # check initial state 
        assert_data_vertices(self.bbbd_algo.variables, data_vertex_iter_7, data_blocks_iter_7)  
        assert_data_constr(self.bbbd_algo.constraints, data_constr_iter_7)
        assert_data_sorting(self.bbbd_algo.sorting_structure, data_ss_iter_7)
        assert_data_blocks(self.bbbd_algo.blocks, data_blocks_iter_7)

    def test_solve_termination_second_case(self):
        edges = [(0,1),(0,3), (0,5), (1,0), (2,3), (2,4), (2,5), (3,3), \
        (4,2), (4,4), (5,1), (5,6), (5,7), (6,0),(6,7), (7,0), (7,1), (7,2)]
        m = 8
        n = 8

        self.bbbd_algo = BBBD_algo(edges, m, n, 0.3)


        column_order, row_order, blocks = self.bbbd_algo.solve()
        # check initial state 
        assert_data_vertices(self.bbbd_algo.variables, data_vertex_iter_6, data_blocks_iter_6)  
        assert_data_constr(self.bbbd_algo.constraints, data_constr_iter_6)
        assert_data_sorting(self.bbbd_algo.sorting_structure, data_ss_iter_6)
        assert_data_blocks(self.bbbd_algo.blocks, data_blocks_iter_6)

        expected_column_order = [6,7,0,2,4,5,1,3]
        expected_row_order = [5,6,1,4,2,0,3,7]
        expected_blocks = [[[6,7,0],[5,6,1]], [[2,4,5],[4,2,0]]]

        assert expected_column_order == column_order 
        assert expected_row_order == row_order
        assert expected_blocks == blocks


def test_rectangular_row():
    # add dense constraint
    edges = [(0,1),(0,3), (0,5), (1,0), (2,3), (2,4), (2,5), (3,3), \
    (4,2), (4,4), (5,1), (5,6), (5,7), (6,0),(6,7), (7,0), (7,1), (7,2),\
    (8,0), (8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7)]
    m = 9
    n = 8

    bbbd_algo = BBBD_algo(edges, m, n, 0.3)


    column_order, row_order, blocks = bbbd_algo.solve()

    expected_column_order = [6,7,0,2,4,5,1,3]
    expected_row_order = [5,6,1,4,2,0,3,7,8]

    assert expected_column_order == column_order 
    assert expected_row_order == row_order

def test_rectangular_col():
    # add dense variable
    edges = [(0,1),(0,3), (0,5), (1,0), (2,3), (2,4), (2,5), (3,3), \
    (4,2), (4,4), (5,1), (5,6), (5,7), (6,0),(6,7), (7,0), (7,1), (7,2),\
    (0,8), (1,8), (2,8), (3,8), (4,8), (5,8), (6,8), (7,8)]
    m = 8
    n = 9

    bbbd_algo = BBBD_algo(edges, m, n, 0.3)


    column_order, row_order, blocks = bbbd_algo.solve()

    expected_column_order = [6,7,0,2,4,5,1,3,8]
    expected_row_order = [5,6,1,4,2,0,3,7]

    assert expected_column_order == column_order 
    assert expected_row_order == row_order


data2_vertex_iter_0 = [[[],[],[],[]], \
                    [[1],[0,3],[1,2],[0]],\
                    [{1:0},{0:0,3:0}, {1:0, 2:0}, {0:0}],\
                    [1,1,1,1],\
                    [1,2,2,1],\
                    [None, 3, 2, None]]
data2_constr_iter_0 = [[[],[],[],[]], \
                     [0,0,0,0], \
                     [[1,3], [0,2], [2], [1]],\
                     [2,2,1,1]]
data2_blocks_iter_0 = []
data2_ss_iter_0 = [{1: [0,3,1,2]}, 1, 1, [0,3,1,2]] 

data2_vertex_iter_1 = [[[],[],[0],[]], \
                    [[],[0,3],[2],[0]],\
                    [{},{0:0,3:0}, {2:0}, {0:0}],\
                    [1,1,2,1],\
                    [0,2,1,1],\
                    [None, 3, 2, None]]
data2_constr_iter_1 = [[[],[],[],[]], \
                     [0,0,0,0], \
                     [[1,3], [2], [2], [1]],\
                     [2,1,1,1]]
data2_blocks_iter_1 = [[0, [0], [1], 1, [2], []]]
data2_ss_iter_1 = [{1: [3,1], 2:[2]}, 1, 1, [3,1]] 

data2_vertex_iter_2 = [[[],[1],[0],[]], \
                    [[],[3],[2],[]],\
                    [{},{3:0}, {2:0}, {}],\
                    [1,2,2,1],\
                    [0,1,1,0],\
                    [None, 3, 2, None]]
data2_constr_iter_2 = [[[],[],[],[]], \
                     [0,0,0,0], \
                     [[1], [2], [2], [1]],\
                     [1,1,1,1]]
data2_blocks_iter_2 = [[0, [0], [1], 1, [2], []], [1, [3], [0], 1, [1], []]]
data2_ss_iter_2 = [{1: [], 2:[1,2]}, 2, 2, [2,1]] 

data2_vertex_iter_3 = [[[],[1],[0],[]], \
                    [[],[3],[],[]],\
                    [{},{3:0}, {}, {}],\
                    [1,2,2,1],\
                    [0,1,0,0],\
                    [None, 3, 2, None]]
data2_constr_iter_3 = [[[],[],[],[]], \
                     [0,0,0,0], \
                     [[1], [2], [], [1]],\
                     [1,1,0,1]]
data2_blocks_iter_3 = [[0, [0,2], [1,2], 2, [], []], [1, [3], [0], 1, [1], []]]
data2_ss_iter_3 = [{1: [], 2:[1]}, 2, 2, [1]] 

data2_vertex_iter_4 = [[[],[1],[0],[]], \
                    [[],[],[],[]],\
                    [{},{}, {}, {}],\
                    [1,2,2,1],\
                    [0,1,0,0],\
                    [None, 3, 2, None]]
data2_constr_iter_4 = [[[],[],[],[]], \
                     [0,0,0,0], \
                     [[1], [2], [], []],\
                     [1,1,0,0]]
data2_blocks_iter_4 = [[0, [0,2], [1,2], 2, [], []], [1, [3,1], [0,3], 2, [], []]]
data2_ss_iter_4 = [{1: [], 2:[]}, 2, 2, []] 



class testProblemLoop(unittest.TestCase):
    def test_algorithm(self):
        edges = [(0,1),(0,3), (1,0), (1,2), (2,2), (3,1)]
        m = 4
        n = 4

        self.bbbd_algo = BBBD_algo(edges, m, n, 0.5)

        assert_data_vertices(self.bbbd_algo.variables, data2_vertex_iter_0, data2_blocks_iter_0)  
        assert_data_constr(self.bbbd_algo.constraints, data2_constr_iter_0)
        assert_data_sorting(self.bbbd_algo.sorting_structure, data2_ss_iter_0)
        assert_data_blocks(self.bbbd_algo.blocks, data2_blocks_iter_0)

        self.bbbd_algo.iteration()
        assert_data_vertices(self.bbbd_algo.variables, data2_vertex_iter_1, data2_blocks_iter_1)  
        assert_data_constr(self.bbbd_algo.constraints, data2_constr_iter_1)
        assert_data_sorting(self.bbbd_algo.sorting_structure, data2_ss_iter_1)
        assert_data_blocks(self.bbbd_algo.blocks, data2_blocks_iter_1)

        self.bbbd_algo.iteration()
        assert_data_vertices(self.bbbd_algo.variables, data2_vertex_iter_2, data2_blocks_iter_2)  
        assert_data_constr(self.bbbd_algo.constraints, data2_constr_iter_2)
        assert_data_sorting(self.bbbd_algo.sorting_structure, data2_ss_iter_2)
        assert_data_blocks(self.bbbd_algo.blocks, data2_blocks_iter_2)

        self.bbbd_algo.iteration()
        assert_data_vertices(self.bbbd_algo.variables, data2_vertex_iter_3, data2_blocks_iter_3)  
        assert_data_constr(self.bbbd_algo.constraints, data2_constr_iter_3)
        assert_data_sorting(self.bbbd_algo.sorting_structure, data2_ss_iter_3)
        assert_data_blocks(self.bbbd_algo.blocks, data2_blocks_iter_3)

        self.bbbd_algo.iteration()
