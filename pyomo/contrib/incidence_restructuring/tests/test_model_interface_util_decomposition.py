"""
This script tests model interface util functions
"""
from pyomo.contrib.incidence_restructuring.model_interface_util_decomposition import *
import pyomo.environ as pyo
import scipy as sc
import pyomo.common.unittest as unittest
import pytest as pytest
import numpy as np
import math

    
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
            get_restructured_matrix(self.incidence_matrix, self.igraph, self.m, "test_problem", method=1, fraction=1.1)

        assert self.col_order == [1, 0, 2, 3]
        assert self.row_order == [0, 3, 1, 2]
        assert self.blocks == [[[1, 0], [0, 3]], [[2, 3], [1, 2]]]

    def test_show_decomposed_matrix(self):
        self.col_order, self.row_order, self.blocks, self.idx_var_map, self.idx_constr_map = \
            get_restructured_matrix(self.incidence_matrix, self.igraph, self.m, "test_problem", method=1, fraction=1.1)
        show_decomposed_matrix(self.m, method=1, fraction=1.1)
        assert True #visual test

    def test_get_restructured_matrix_gp(self):
        # only 2 partitions for simplicity
        self.col_order, self.row_order, self.blocks, self.idx_var_map, self.idx_constr_map = \
            get_restructured_matrix(self.incidence_matrix, self.igraph, self.m, "test_problem", method=2, num_part=2)
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
            self.idx_constr_map = get_restructured_matrix_general(
            self.incidence_matrix, self.igraph, self.m, "test_problem", method=request.param, num_part=2, fraction=0.6)
#     # @pytest.mark.parametrize(
#     #     'method',
#     #     ([1, 2]),
#     #     indirect=True
#     # )

    def test_create_subsystem_from_constr_list(self):
        constr_list = [1,2]
        create_subsystem_from_constr_list(self.m, constr_list, self.idx_constr_map, "block_1")
        
        variables = [var for var in self.m.find_component("block_1").component_data_objects(pyo.Var)]
        constraints = [constr for constr in self.m.find_component("block_1").component_data_objects(pyo.Constraint)]

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
        create_subsystems(self.m, [i[1] for i in self.blocks], self.idx_constr_map)
        for idx in range(len(self.blocks)):
            variables = [var for var in self.m.find_component(f"subsystem_{idx}").component_data_objects(pyo.Var)]
            constraints = [constr for constr in self.m.find_component(f"subsystem_{idx}").component_data_objects(pyo.Constraint)]

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
    
    def test_list_complicating_constraints(self):
        complicating_constr = get_list_of_complicating_constraints(self.blocks, self.idx_constr_map)

        if self.method == 1:
            expected_constr = ["cons5"]
        if self.method == 2:
            expected_constr = ["cons1", "cons2"]

        for i in complicating_constr:
            assert i.name in expected_constr     
        for i in expected_constr:
            assert i in [j.name for j in complicating_constr]

class TestBlockDecompositionForSolverMatched():
    @pytest.fixture(autouse=True, params=[1,2])
    def set_model(self, request):
        self.m = pyo.ConcreteModel()
        self.m.x1 = pyo.Var(name="x1")
        self.m.x2 = pyo.Var(name="x2")
        self.m.x3 = pyo.Var(name="x3")
        self.m.x4 = pyo.Var(name="x4")
        self.m.x5 = pyo.Var(name="x5")
        self.m.x6 = pyo.Var(name="x6")

        self.m.cons1 = pyo.Constraint(expr=self.m.x2 + self.m.x4 + self.m.x5 == 1.5)
        self.m.cons2 = pyo.Constraint(expr=self.m.x1 + self.m.x3 == 1)
        self.m.cons3 = pyo.Constraint(expr=self.m.x3 == 0.5)
        self.m.cons4 = pyo.Constraint(expr=self.m.x2 == 0.5)
        self.m.cons5 = pyo.Constraint(expr=self.m.x1 + self.m.x3 + self.m.x5 == 1.5)
        self.m.cons6 = pyo.Constraint(expr=self.m.x2 + self.m.x4 + self.m.x6 == 1.5)

        self.igraph, self.incidence_matrix = get_incidence_matrix(self.m)
        self.method = request.param
        # only 2 partitions for simplicity
        self.order, self.tmp, self.blocks, \
            self.original_mapping_var, self.original_mapping_constr = \
                get_restructured_matrix_general(self.incidence_matrix, self.igraph, 
                self.m, "test_problem", method=request.param, num_part=2, d_max=1, n_max=2, matched=True,
                border_fraction=0.2)

    # @pytest.mark.parametrize(
    #     'method',
    #     ([1, 2]),
    #     indirect=True
    # )
    def test_reformat_blocks(self):
        resulting_blocks = self.blocks
        expected_blocks = [[[1,2],[1,2]],[[3,5],[3,5]]]

        # test equivalency regardless of order
        for i in resulting_blocks:
            assert i in expected_blocks
        for i in expected_blocks:
            assert i in resulting_blocks

    def test_create_subsystem_from_block(self):
        block = [1,2]
        create_subsystem_from_constr_list(self.m, block, self.original_mapping_constr, "block_1")
        
        variables = [var for var in self.m.find_component("block_1").component_data_objects(pyo.Var)]
        constraints = [constr for constr in self.m.find_component("block_1").component_data_objects(pyo.Constraint)]

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

    # @pytest.mark.parametrize(
    #     'method',
    #     ([1, 2]),
    #     indirect=True
    # )
    def test_create_subsystems(self):
        create_subsystems(self.m, [i[1] for i in self.blocks], self.original_mapping_constr)
        for idx in range(len(self.blocks)):
            variables = [var for var in self.m.find_component(f"subsystem_{idx}").component_data_objects(pyo.Var)]
            constraints = [constr for constr in self.m.find_component(f"subsystem_{idx}").component_data_objects(pyo.Constraint)]


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

class TestConstraintConsensus():
    @pytest.fixture(autouse=True)  
    def set_model(self, request):
        self.m = pyo.ConcreteModel()
        self.m.x1 = pyo.Var(name="x1")
        self.m.x2 = pyo.Var(name="x2")
        self.m.x3 = pyo.Var(name="x3")
        self.m.x4 = pyo.Var(name="x4")
        self.m.x5 = pyo.Var(name="x5")
        self.m.x6 = pyo.Var(name="x6")

        self.m.cons1 = pyo.Constraint(expr=self.m.x2 + self.m.x4 + self.m.x5 == 1.5)
        self.m.cons2 = pyo.Constraint(expr=self.m.x1 + self.m.x3 == 1)
        self.m.cons3 = pyo.Constraint(expr=self.m.x3 == 0.5)
        self.m.cons4 = pyo.Constraint(expr=self.m.x2 == 0.5)
        self.m.cons5 = pyo.Constraint(expr=self.m.x1 + self.m.x3 + self.m.x5 == 1.5)
        self.m.cons6 = pyo.Constraint(expr=self.m.x2 + self.m.x4 + self.m.x6 == 1.5)

        self.m.x1.value = 1
        self.m.x2.value = 1
        self.m.x3.value = 1
        self.m.x4.value = 0
        self.m.x5.value = 0
        self.m.x6.value = 0

        self.constraints = [constr for constr in self.m.component_objects(pyo.Constraint)]
        self.variables = [var for var in self.m.component_objects(pyo.Var)]
    
    def test_determine_d_constr(self):
        d_s = [determine_d_constr(constr) for constr in self.constraints]
        expected_d_s = [1, -1, -1, -1, -1, 1]
        assert d_s == expected_d_s
        # assert False
    def test_get_constraint_gradient(self):
        grad = []
        for constraint in self.constraints:
            grad.append(get_constraint_gradient(constraint))
        grad_expected = [[1,1,1], [1,1], [1], [1], [1,1,1], [1,1,1]]
        assert grad == grad_expected

    def test_get_constraint_gradient_all_vars(self):
        # keep track of index to variable for gradient
        vars = list(var for var in self.m.component_objects(pyo.Var))
        grads = list(get_constraint_gradient_all_vars(constr, vars) for constr in self.constraints)
        expected_grads = [[0,1,0,1,1,0], [1,0,1,0,0,0], [0,0,1,0,0,0],\
                          [0,1,0,0,0,0], [1,0,1,0,1,0], [0,1,0,1,0,1]]
        assert grads == expected_grads

    def test_norm_l2(self):
        vector = [1,0,-1]
        norm = norm_l2(vector)
        assert norm == math.sqrt(2)

        vector = [-2, 3, 4]
        norm = norm_l2(vector)
        assert norm == math.sqrt(29)
    
    def test_get_constraint_violation(self):
        violations = [get_constraint_violation(constr) for constr in self.constraints]
        expected_violations = [0.5, 1, 0.5, 0.5, 0.5, 0.5]
        assert violations == expected_violations

    def test_adjust_consensus_value(self):
        lb = -1
        ub = 1
        val = 2
        assert adjust_consensus_value(val, lb, ub) == 1

        val = -2
        assert adjust_consensus_value(val, lb, ub) == -1

        val = 0
        assert adjust_consensus_value(val, lb, ub) == 0

    def test_constraint_consensus(self):
        #d, v, grad_constr, norm_grad_constr, squared_norm_grad_constr, s, \
        fv_consensus, distances, s = constraint_consensus(self.constraints, self.variables, 0.01) 
        # assert all(d == [1, -1, -1, -1, -1, 1])
        # assert all(v == [0.5, 1, 0.5, 0.5, 0.5, 0.5])
        # expected_grad_constr = list(np.array(i) for i in [[0,1,0,1,1,0], [1,0,1,0,0,0], [0,0,1,0,0,0],\
        #                   [0,1,0,0,0,0], [1,0,1,0,1,0], [0,1,0,1,0,1]])
        # assert all(all(grad_constr[i] == expected_grad_constr[i]) for i in range(len(grad_constr)))
        # assert norm_grad_constr == [math.sqrt(3), math.sqrt(2), 1, 1, math.sqrt(3), math.sqrt(3)]
        # exp_sngc = [3,2,1,1,3,3]
        # assert all(list(abs(squared_norm_grad_constr[i] - exp_sngc[i]) < 1e-10 for i in range(len(exp_sngc))))
        # exp_s = [2,3,3,2,2,1]
        # assert all(list(abs(s[i] - exp_s[i]) < 1e-10 for i in range(len(s))))
        expected_fv_consensus = [-1/3, -1/18, -7/18, 1/6, 0, 1/6]
        assert all(list(abs(expected_fv_consensus[i] - fv_consensus[i]) < 1e-10 for i in range(len(fv_consensus))))
        expected_distances = [0.5/math.sqrt(3), 1/math.sqrt(2), 1/2, 1/2, 0.5/math.sqrt(3), 0.5/math.sqrt(3)]
        assert all(list(abs(expected_distances[i] - distances[i]) < 1e-10 for i in range(len(distances))))
       
        fv_consensus, distances, s = constraint_consensus(self.constraints, self.variables, 0.5/math.sqrt(3)+0.01) 
        expected_fv_consensus = [-1/2, -1/2, -1/2, 0, 0, 0]
        assert all(list(abs(expected_fv_consensus[i] - fv_consensus[i]) < 1e-10 for i in range(len(fv_consensus))))
        expected_distances = [0.5/math.sqrt(3), 1/math.sqrt(2), 1/2, 1/2, 0.5/math.sqrt(3), 0.5/math.sqrt(3)]
        assert all(list(abs(expected_distances[i] - distances[i]) < 1e-10 for i in range(len(distances))))
        
        vars = [self.variables[i] for i in range(len(self.variables)) if i%2==0 ]
        fv_consensus, distances, s = constraint_consensus(self.constraints, vars, 0.01) 
        expected_fv_consensus = [-1/3, -7/18,  1/6]
        assert all(list(abs(expected_fv_consensus[i] - fv_consensus[i]) < 1e-10 for i in range(len(fv_consensus))))
        expected_distances = [0.5, 1/math.sqrt(2), 1/2,  0.5/math.sqrt(3)]
        assert all(list(abs(expected_distances[i] - distances[i]) < 1e-10 for i in range(len(distances))))
       
class TestBlockDecompositionForSolverMatchedConfig2():
    @pytest.fixture(autouse=True)  
    def set_model(self, request):
        self.m = pyo.ConcreteModel()
        self.m.x1 = pyo.Var(name="x1")
        self.m.x2 = pyo.Var(name="x2")
        self.m.x3 = pyo.Var(name="x3")
        self.m.x4 = pyo.Var(name="x4")
        self.m.x5 = pyo.Var(name="x5")
        self.m.x6 = pyo.Var(name="x6")
        self.m.x7 = pyo.Var(name="x7")
        self.m.x8 = pyo.Var(name="x8")

        # add some constraints here
        self.m.cons1 = pyo.Constraint(expr=self.m.x2 + self.m.x4 + self.m.x5 == 1.5)
        self.m.cons2 = pyo.Constraint(expr=self.m.x1 + self.m.x3 == 1)
        self.m.cons3 = pyo.Constraint(expr=self.m.x3 + self.m.x7 == 1.0)
        self.m.cons4 = pyo.Constraint(expr=self.m.x2 == self.m.x8 == 1.0)
        self.m.cons5 = pyo.Constraint(expr=self.m.x1 + self.m.x3 + self.m.x5 == 1.5)
        self.m.cons6 = pyo.Constraint(expr=self.m.x2 + self.m.x4 + self.m.x6 == 1.5)
        self.m.cons7 = pyo.Constraint(expr=self.m.x2 + self.m.x3 + self.m.x7 = 1.5)
        self.m.cons8 = pyo.Constraint(expr=self.m.x1 + self.m.x4 + self.m.x7 + self.m.x8 = 2.0)

        self.igraph, self.incidence_matrix = get_incidence_matrix(self.m)
        self.method = 2
        # only 2 partitions for simplicity
        self.order, self.tmp, self.blocks, \
            self.idx_var_map, self.idx_constr_map = \
                get_restructured_matrix_general(self.incidence_matrix, self.igraph, 
                self.m, "test_problem", method=self.method, num_part=2, d_max=1, n_max=2, matched=True,
                border_fraction=0.2)
    
    def test_visual_1(self):
        assert True

    def test_show_decomposed_matrix(self):
        self.col_order, self.row_order, self.blocks, self.idx_var_map, self.idx_constr_map = \
            get_restructured_matrix(self.incidence_matrix, self.igraph, self.m, "test_problem", method=1, fraction=1.1)
        show_decomposed_matrix(self.m, method=1, fraction=1.1)
        assert True #visual test

    def test_get_restructured_matrix_gp(self):
        # only 2 partitions for simplicity
        self.col_order, self.row_order, self.blocks, self.idx_var_map, self.idx_constr_map = \
            get_restructured_matrix(self.incidence_matrix, self.igraph, self.m, "test_problem", method=2, num_part=2)
        show_decomposed_matrix(self.m, method=2, num_part=2)

        print(self.col_order)
        print(self.row_order)
        print(self.blocks)
        # assert self.col_order == [1, 0, 2, 3]
        # assert self.row_order == [0, 3, 1, 2]
        # assert self.blocks == [[[1, 0], [0, 3]], [[2, 3], [1, 2]]]
        assert True



        