"""
This script tests model interface util functions
"""
from pyomo.contrib.incidence_restructuring.model_interface_util_decomposition import *
import pyomo.environ as pyo
import scipy as sc
import pyomo.common.unittest as unittest
import pytest as pytest
import numpy as np

    
def test_reorder_sparse_matrix():
    matrix = sc.sparse.coo_matrix([[1,0,1,0], [0,1,0,0],[0,0,1,0], [0,1,0,1]])
    assert (matrix.getnnz() == 6)

    row_order = [0,2,3,1]
    col_order = [0,2,1,3]

    final_matrix = reorder_sparse_matrix(len(row_order), len(col_order), row_order, col_order, matrix)
    assert (matrix.getnnz() == final_matrix.getnnz())

    expected_final_matrix = sc.sparse.coo_matrix([[1,1,0,0],[0,1,0,0], [0,0,1,1], [0,0,1,0]])

    nz_rows, nz_cols = final_matrix.nonzero()
    nz_rows_matrix, nz_cols_matrix = expected_final_matrix.nonzero()

    for i in range(len(nz_rows)):
        assert nz_rows[i] == nz_rows_matrix[i]
        assert nz_cols[i] == nz_cols_matrix[i]

def test_reorder_sparse_matrix_rectangular_adj_list():
    matrix = sc.sparse.coo_matrix([[1,0,1,0,1], [0,1,0,0,1],[0,0,1,0,1], [0,1,0,1,1]])
    assert (matrix.getnnz() == 10)

    row_order = [0,2,3,1]
    col_order = [4,0,2,1,3]

    final_matrix = reorder_sparse_matrix(len(row_order), len(col_order), row_order, col_order, matrix)
    assert (matrix.getnnz() == final_matrix.getnnz())

    expected_final_matrix = sc.sparse.coo_matrix([[1,1,1,0,0],[1,0,1,0,0], [1,0,0,1,1], [1,0,0,1,0]])

    edges = create_edge_list_from_matrix(final_matrix)
    expected_edges = create_edge_list_from_matrix(expected_final_matrix)

    dense_computed = np.array(final_matrix.todense())
    dense_expected = np.array(expected_final_matrix.todense())

    assert dense_computed.shape == dense_expected.shape
    for i in range(len(dense_computed)):
        for j in range(len(dense_computed[0])):
            assert dense_computed[i][j] == dense_expected[i][j]

    for edge in edges:
        assert edge in expected_edges 
    for edge in expected_edges:
        assert edge in edges

# a very simple model that needs reordering
class TestReorderingVariableMapping(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def set_model(self):
        self.m = pyo.ConcreteModel()
        self.m.x1 = pyo.Var(name="x1")
        self.m.x2 = pyo.Var(name="x2")
        self.m.x3 = pyo.Var(name="x3")
        self.m.x4 = pyo.Var(name="x4")

        self.m.cons1 = pyo.Constraint(expr=self.m.x2 + self.m.x4 == 1)
        self.m.cons2 = pyo.Constraint(expr=self.m.x1 + self.m.x3 == 1)
        self.m.cons3 = pyo.Constraint(expr=self.m.x3 == 0.5)
        self.m.cons4 = pyo.Constraint(expr=self.m.x2 == 0.5)  
        # self.m.cons5 = pyo.Constraint(expr=self.m.x3 == 0.5)
        self.igraph, self.incidence_matrix = get_incidence_matrix(self.m)

    def test_get_restructured_matrix(self):
        # only 2 partitions for simplicity
        self.col_order, self.row_order, self.blocks, self.idx_var_map, self.idx_constr_map = \
            get_restructured_matrix(self.incidence_matrix, self.igraph, self.m, method=1, fraction=1.1)

        assert self.col_order == [1, 0, 2, 3]
        assert self.row_order == [0, 3, 1, 2]
        assert self.blocks == [[[1, 0], [0, 3]], [[2, 3], [1, 2]]]

    def test_show_decomposed_matrix(self):
        self.col_order, self.row_order, self.blocks, self.idx_var_map, self.idx_constr_map = \
            get_restructured_matrix(self.incidence_matrix, self.igraph, self.m, method=1, fraction=1.1)
        show_decomposed_matrix(self.m, method=1, fraction=1.1)
        assert True #visual test

    def test_get_restructured_matrix_gp(self):
        # only 2 partitions for simplicity
        self.col_order, self.row_order, self.blocks, self.idx_var_map, self.idx_constr_map = \
            get_restructured_matrix(self.incidence_matrix, self.igraph, self.m, method=2, num_part=2)
        show_decomposed_matrix(self.m, method=2, num_part=2)

        print(self.col_order)
        print(self.row_order)
        print(self.blocks)
        # assert self.col_order == [1, 0, 2, 3]
        # assert self.row_order == [0, 3, 1, 2]
        # assert self.blocks == [[[1, 0], [0, 3]], [[2, 3], [1, 2]]]
        assert True

#     def test_get_mappings_to_original(self):
#         original_matching_constr_computed, original_matching_var_computed = \
#             get_mappings_to_original(self.order, self.col_map, self.idx_var_map, self.idx_constr_map)

#         original_matching_constr_expected = {0: "cons1", 1: "cons2", 2: "cons3", 3: "cons4"}
#         original_matching_var_expected = {0: "x4", 1: "x1", 2: "x3", 3: "x2"}

#         for key in original_matching_var_computed:
#             assert original_matching_var_computed[key].name == original_matching_var_expected[key]
        
#         for key in original_matching_constr_computed:
#             assert original_matching_constr_computed[key].name == original_matching_constr_expected[key]

class TestBlockDecompositionForSolver():
    # solution is x_i = 0.5
    @pytest.fixture(autouse=True, params=[1,2])
    def set_model(self, request):
        self.m = pyo.ConcreteModel()
        self.m.x1 = pyo.Var(name="x1", bounds=(0,1))
        self.m.x2 = pyo.Var(name="x2", bounds=(0,1))
        self.m.x3 = pyo.Var(name="x3", bounds=(0,1))
        self.m.x4 = pyo.Var(name="x4", bounds=(0,1))
        self.m.x5 = pyo.Var(name="x5", bounds=(0,1))
        self.m.x6 = pyo.Var(name="x6", bounds=(0,1))

        self.m.cons1 = pyo.Constraint(expr=self.m.x2 + self.m.x4 + self.m.x5 == 1.5)
        self.m.cons2 = pyo.Constraint(expr=self.m.x1 + self.m.x3 == 1)
        self.m.cons3 = pyo.Constraint(expr=self.m.x3 == 0.5)
        self.m.cons4 = pyo.Constraint(expr=self.m.x2 == 0.5)
        self.m.cons5 = pyo.Constraint(expr=self.m.x1 + self.m.x3 + self.m.x5 == 1.5)
        self.m.cons6 = pyo.Constraint(expr=self.m.x2 + self.m.x4 + self.m.x6 == 1.5)

        # only 2 partitions for simplicity

        self.igraph, self.incidence_matrix = get_incidence_matrix(self.m)
        self.method = request.param

        self.col_order, self.row_order, self.blocks, self.idx_var_map, \
            self.idx_constr_map = get_restructured_matrix(
            self.incidence_matrix, self.igraph, self.m, method=request.param, num_part=2, fraction=0.6)
#     # @pytest.mark.parametrize(
#     #     'method',
#     #     ([1, 2]),
#     #     indirect=True
#     # )

    def test_create_subsystem_from_constr_list(self):
        constr_list = [1,2]
        subsystem = create_subsystem_from_constr_list(constr_list, self.idx_constr_map)
        
        variables = [var for var in subsystem.component_data_objects(pyo.Var)]
        constraints = [constr for constr in subsystem.component_data_objects(pyo.Constraint)]

        expected_variables = ["x1", "x3"]
        expected_constraints = ["cons2", "cons3"]

        for i in variables:
            assert i.name in expected_variables
        for i in constraints:
            assert i.name in expected_constraints
        
        for i in expected_variables:
            assert i in [j.name for j in variables]
        for i in expected_constraints:
            assert i in [j.name for j in constraints]

    # def test_see_blocks(self):
    #     print(self.blocks)
    #     show_decomposed_matrix(self.m, method=self.method, num_part=2, fraction=0.6)
    #     print({i : self.idx_var_map[i].name for i in self.idx_var_map})
    #     print({i : self.idx_constr_map[i].name for i in self.idx_constr_map})
    #     assert False

#     # @pytest.mark.parametrize(
#     #     'method',
#     #     ([1, 2]),
#     #     indirect=True
#     # )
    def test_create_subsystems(self):
        subsystems = create_subsystems([i[1] for i in self.blocks], self.idx_constr_map)
        assert len(subsystems) == 2
        for subsystem in subsystems:
            variables = [var for var in subsystem.component_data_objects(pyo.Var)]
            constraints = [constr for constr in subsystem.component_data_objects(pyo.Constraint)]

            if self.method == 2:
                if "x6" in [i.name for i in variables]:

                    expected_variables = ["x2", "x4", "x6"]
                    expected_constraints = ["cons4", "cons6"]

                    for i in variables:
                        assert i.name in expected_variables
                    for i in constraints:
                        assert i.name in expected_constraints
                    
                    for i in expected_variables:
                        assert i in [j.name for j in variables]
                    for i in expected_constraints:
                        assert i in [j.name for j in constraints]
                else:
                    expected_variables = ["x1", "x5", "x3"]
                    expected_constraints = ["cons3", "cons5"]

                    for i in variables:
                        assert i.name in expected_variables
                    for i in constraints:
                        assert i.name in expected_constraints
                    
                    for i in expected_variables:
                        assert i in [j.name for j in variables]
                    for i in expected_constraints:
                        assert i in [j.name for j in constraints]
            
            if self.method == 1:
                if "x6" in [i.name for i in variables]:

                    expected_variables = ["x2", "x4", "x5", "x6"]
                    expected_constraints = ["cons1", "cons4", "cons6"]

                    for i in variables:
                        assert i.name in expected_variables
                    for i in constraints:
                        assert i.name in expected_constraints
                    
                    for i in expected_variables:
                        assert i in [j.name for j in variables]
                    for i in expected_constraints:
                        assert i in [j.name for j in constraints]
                else:
                    expected_variables = ["x1", "x3"]
                    expected_constraints = ["cons2", "cons3"]

                    for i in variables:
                        assert i.name in expected_variables
                    for i in constraints:
                        assert i.name in expected_constraints
                    
                    for i in expected_variables:
                        assert i in [j.name for j in variables]
                    for i in expected_constraints:
                        assert i in [j.name for j in constraints]

    def test_list_simple_variables(self):
        simple_variables = get_list_of_simple_variables(self.blocks, self.idx_var_map)

        if self.method == 1:
            expected_variables = ["x1", "x2", "x3", "x5", "x6"]
        if self.method == 2:
            expected_variables = ["x1", "x2", "x3", "x6"]

        for i in simple_variables:
            assert i.name in expected_variables     
        for i in expected_variables:
            assert i in [j.name for j in simple_variables]

    def test_list_complicating_variables(self):
        complicating_variables = get_list_of_complicating_variables(self.blocks, self.idx_var_map)

        if self.method == 1:
            expected_variables = ["x4"]
        if self.method == 2:
            expected_variables = ["x4", "x5"]

        for i in complicating_variables:
            assert i.name in expected_variables     
        for i in expected_variables:
            assert i in [j.name for j in complicating_variables]
    

    
class TestInitialization():
    # solution is x_i = 0.5
    @pytest.fixture(autouse=True)
    def set_model(self):
        self.m = pyo.ConcreteModel()
        self.m.x1 = pyo.Var(name="x1", bounds=(0,1))
        self.m.x2 = pyo.Var(name="x2", bounds=(0,1))
        self.m.x3 = pyo.Var(name="x3", bounds=(0,1))
        self.m.x4 = pyo.Var(name="x4", bounds=(0,1))
        self.m.x5 = pyo.Var(name="x5", bounds=(0,1))
        self.m.x6 = pyo.Var(name="x6", bounds=(0,1))

        self.m.cons1 = pyo.Constraint(expr=self.m.x2 + self.m.x4 + self.m.x5 == 1.5)
        self.m.cons2 = pyo.Constraint(expr=self.m.x1 + self.m.x3 == 1)
        self.m.cons3 = pyo.Constraint(expr=self.m.x3 == 0.5)
        self.m.cons4 = pyo.Constraint(expr=self.m.x2 == 0.5)
        self.m.cons5 = pyo.Constraint(expr=self.m.x1 + self.m.x3 + self.m.x5 == 1.5)
        self.m.cons6 = pyo.Constraint(expr=self.m.x2 + self.m.x4 + self.m.x6 == 1.5)
    
    def test_initialization_strategy(self):
        for var in self.m.component_object(pyo.Var):
            print(var.name, var.value)
        initialization_strategy(self.m, method=2, num_part=2, fraction=0.6)
        for var in self.m.component_object(pyo.Var):
            print(var.name, var.value)
        assert False


