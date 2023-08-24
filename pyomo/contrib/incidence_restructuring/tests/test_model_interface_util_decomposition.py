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
        self.m.cons4 = pyo.Constraint(expr=self.m.x2 + self.m.x8 == 1.0)
        self.m.cons5 = pyo.Constraint(expr=self.m.x1 + self.m.x3 + self.m.x5 == 1.5)
        self.m.cons6 = pyo.Constraint(expr=self.m.x2 + self.m.x4 + self.m.x6 == 1.5)
        self.m.cons7 = pyo.Constraint(expr=self.m.x2 + self.m.x3 + self.m.x7 == 1.5)
        self.m.cons8 = pyo.Constraint(expr=self.m.x1 + self.m.x4 + self.m.x8 == 1.5)
        # self.m = pyo.ConcreteModel()
        # self.m.obj = pyo.Objective(expr=0)
        # self.m.x1 = pyo.Var(name="x1")
        # self.m.x2 = pyo.Var(name="x2")
        # self.m.x3 = pyo.Var(name="x3")
        # self.m.x4 = pyo.Var(name="x4")
        # self.m.x5 = pyo.Var(name="x5")
        # self.m.x6 = pyo.Var(name="x6")
        # self.m.x7 = pyo.Var(name="x7")
        # self.m.x8 = pyo.Var(name="x8")

        # # add some constraints here
        # self.m.cons1 = pyo.Constraint(expr=pyo.log(self.m.x2) + self.m.x4**3 + pyo.sqrt(self.m.x5) == pyo.log(0.5) + 0.5**3 + pyo.sqrt(0.5))
        # self.m.cons2 = pyo.Constraint(expr=self.m.x1*self.m.x3 == 0.25)
        # self.m.cons3 = pyo.Constraint(expr=self.m.x3*self.m.x7**2 == 0.5**3)
        # self.m.cons4 = pyo.Constraint(expr=pyo.exp(self.m.x2 + self.m.x8) == pyo.exp(1))
        # self.m.cons5 = pyo.Constraint(expr=self.m.x1*self.m.x3*self.m.x5 == 0.5**3)
        # self.m.cons6 = pyo.Constraint(expr=self.m.x2**2 + self.m.x4**2 + self.m.x6**2 == 3*(0.25))
        # self.m.cons7 = pyo.Constraint(expr=self.m.x2 + self.m.x3**2 + self.m.x7 == 1.25)
        # self.m.cons8 = pyo.Constraint(expr=self.m.x1**4 + self.m.x4 + self.m.x8**4 == 2*(0.5)**4 + 0.5)
       
        self.igraph, self.incidence_matrix = get_incidence_matrix(self.m)
        self.method = 2
        # only 2 partitions for simplicity
        self.order, self.tmp, self.blocks, \
            self.idx_var_map, self.idx_constr_map = \
                get_restructured_matrix_general(self.incidence_matrix, self.igraph, 
                self.m, "test_problem", method=self.method, fraction=0.2, num_part=2, matched=True, d_max=2, n_max=1, 
                border_fraction=0.2)

        self.complicating_constraints = get_list_complicating_constraint_indices(self.blocks, self.idx_constr_map)
        self.complicating_variables = get_list_of_complicating_variables(self.blocks, self.idx_var_map)
        self.constr_list = [i[1] for i in self.blocks]
        self.var_list = [i[0] for i in self.blocks]
        self.subsystem_constr_list = [i + self.complicating_constraints for i in self.constr_list]
        self.subsystems = create_subsystems(self.m, self.subsystem_constr_list, self.idx_constr_map, no_objective=True)
        self.constr_list_all = [self.idx_constr_map[constr] for constr in self.idx_constr_map]
    
    def test_display_reordered_matrix(self):
        get_restructured_matrix_matched(self.incidence_matrix, self.igraph, "test_folder", 
            method=1, num_part=2, d_max=2, n_max=1, 
            border_fraction=0.2, show=True)
        # show_matrix_structure(self.incidence_matrix)
        # show_decomposed_matrix(self.m, method=2, num_part=2)
        #display_reordered_matrix(self.order, self.tmp, self.incidence_matrix)
        print(self.blocks)
        for idx, i in enumerate(self.idx_constr_map):
            print(idx, " : ", self.idx_constr_map[i].name)
        for idx, i in enumerate(self.idx_var_map):
            print(idx, " : ", self.idx_var_map[i].name)

        assert True #visual test
    
    def test_subsystem_partitioning(self):
        print("Blocks : ", self.blocks)
        for constr in self.complicating_constraints:
            print(self.idx_constr_map[constr].name)
        for var in self.complicating_variables:
            print(var.name)
        for idx, constrs in enumerate(self.subsystem_constr_list):
            print("Subsystem {}".format(idx))
            for constr in constrs:
                print(self.idx_constr_map[constr].name)
            print("Variable(s) to report : ")
            for var in self.var_list[idx]:
                print(self.idx_var_map[var].name)
        assert False
    
    def test_first_solve(self):
        starting_value = 10
        global_approximation = ComponentMap()
        for idx in self.idx_var_map:
            if idx%2 == 0:
                global_approximation[self.idx_var_map[idx]] = starting_value  # initialize to 2 
            else:
                global_approximation[self.idx_var_map[idx]] = -starting_value  # initialize to 2 
            self.idx_var_map[idx].value = starting_value
        # for idx in self.idx_var_map:
        #     global_approximation[self.idx_var_map[idx]] = self.idx_var_map[idx].value
        for var in global_approximation:
            print(var.name, var.value)
        
        subsystem_solutions = []
        for subsystem in self.subsystems:
            solutions = ComponentMap()
            for idx in self.idx_var_map:
                solutions[self.idx_var_map[idx]] = global_approximation[self.idx_var_map[idx]]
            subsystem_solutions.append(solutions)

        # solve the system, have to reload the global approximate solution at each step and store
        # the solution for each subsystem at each step
        for i in range(2):
            for idx, subsystem in enumerate(self.subsystems):
                solve_subsystem(self.m.find_component(subsystem), "test_problem", "conopt",  idx)
                print("Solve subsystem {}".format(idx))
                for var in subsystem_solutions[idx]:
                    print(var.name, " : ", var.value)
                    subsystem_solutions[idx][var] = var.value
                    var.value = global_approximation[var]
            
            # update the global approximation from each subsystem
            for i in range(len(self.var_list)):
                print("Updates from subsystem {}".format(i))
                for var_idx in self.var_list[i]:
                    print(self.idx_var_map[var_idx].name, " new value = ", subsystem_solutions[i][self.idx_var_map[var_idx]])
                    self.idx_var_map[var_idx].value = subsystem_solutions[i][self.idx_var_map[var_idx]]
            
            # update the linking constraint variables
            for var in self.complicating_variables:
                var.value = sum([subsystem_solutions[i][var] for i in range(len(subsystem_solutions))])/len(subsystem_solutions)
            
            # update the global approximation
            for var in global_approximation:
                global_approximation[var] = var.value
            
            print("updated global approximation")
            for var in global_approximation:
                print(var.name, " : ", var.value)
            
            print("sum of violation : ", sum(get_constraint_violation(constr) for constr in self.constr_list_all))
        assert False

    def test_initialization_strategy_LC_overlap(self):
        # self.test_first_solve()
        initialization_strategy_LC_overlap(self.m, "test_problem", method=2, num_part=2, fraction=0.5, matched=True,
            d_max=1, n_max=2, solver="ipopt", use_init_heur=1, use_fbbt=1, max_iteration=5, border_fraction=0.5, test=0,
            strip_model_bounds=False, distance=None)
        assert False
    
    
    

class TestStrategyLinearModel():
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
        self.m.cons4 = pyo.Constraint(expr=self.m.x2 + self.m.x8 == 1.0)
        self.m.cons5 = pyo.Constraint(expr=self.m.x1 + self.m.x3 + self.m.x5 == 1.5)
        self.m.cons6 = pyo.Constraint(expr=self.m.x2 + self.m.x4 + self.m.x6 == 1.5)
        self.m.cons7 = pyo.Constraint(expr=self.m.x2 + self.m.x3 + self.m.x7 == 1.5)
        self.m.cons8 = pyo.Constraint(expr=self.m.x1 + self.m.x4 + self.m.x8 == 1.5)
       
        self.igraph, self.incidence_matrix = get_incidence_matrix(self.m)
        self.method = 2
        # only 2 partitions for simplicity
        self.order, self.tmp, self.blocks, \
            self.idx_var_map, self.idx_constr_map = \
                get_restructured_matrix_general(self.incidence_matrix, self.igraph, 
                self.m, "test_problem", method=self.method, fraction=0.2, num_part=2, matched=True, d_max=2, n_max=1, 
                border_fraction=0.2)

        self.complicating_constraints = get_list_complicating_constraint_indices(self.blocks, self.idx_constr_map)
        self.complicating_variables = get_list_of_complicating_variables(self.blocks, self.idx_var_map)
        self.constr_list = [i[1] for i in self.blocks]
        self.var_list = [i[0] for i in self.blocks]
        self.nonlinking_vars = [self.idx_var_map[var] for array in self.var_list for var in array]
        self.subsystem_constr_list = [i + self.complicating_constraints for i in self.constr_list]
        self.subsystems = create_subsystems(self.m, self.subsystem_constr_list, self.idx_constr_map, no_objective=True)
        self.constr_list_all = [self.idx_constr_map[constr] for constr in self.idx_constr_map]
        self.linking_constraint_subsytem = create_subsystem_from_constr_list(self.m, self.complicating_constraints, 
                                                self.idx_constr_map, "linking_constraint_subsystem")
        
        # problem settings:
        self.starting_value = 10
        self.alternate = False
        self.solver = "conopt"
        self.num_iterations = 5
        self.folder = "test_problem_strategy_2"

    
    def test_display_reordered_matrix(self):
        get_restructured_matrix_matched(self.incidence_matrix, self.igraph, "test_folder", 
            method=1, num_part=2, d_max=2, n_max=1, 
            border_fraction=0.2, show=True)
        # show_matrix_structure(self.incidence_matrix)
        # show_decomposed_matrix(self.m, method=2, num_part=2)
        #display_reordered_matrix(self.order, self.tmp, self.incidence_matrix)
        print(self.blocks)
        for idx, i in enumerate(self.idx_constr_map):
            print(idx, " : ", self.idx_constr_map[i].name)
        for idx, i in enumerate(self.idx_var_map):
            print(idx, " : ", self.idx_var_map[i].name)

        assert True #visual test
    
    def test_subsystem_partitioning(self):
        print("Blocks : ", self.blocks)
        for constr in self.complicating_constraints:
            print(self.idx_constr_map[constr].name)
        for var in self.complicating_variables:
            print(var.name)
        for idx, constrs in enumerate(self.subsystem_constr_list):
            print("Subsystem {}".format(idx))
            for constr in constrs:
                print(self.idx_constr_map[constr].name)
            print("Variable(s) to report : ")
            for var in self.var_list[idx]:
                print(self.idx_var_map[var].name)
        print("nonlinking variables")
        for var in self.nonlinking_vars:
            print(var.name)
        print("Linking constraint subsystem")
        for var in self.m.find_component("linking_constraint_subsystem").component_data_objects(pyo.Var):
            print(var.name)
        for constr in self.m.find_component("linking_constraint_subsystem").component_data_objects(pyo.Constraint):
            print(constr.name)
        assert False

    def test_strategy_1(self):
        for constr in [self.idx_constr_map[i] for i in self.complicating_constraints]:
            constr.activate()
        global_approximation = ComponentMap()
        for idx in self.idx_var_map:
            if self.alternate and idx%2 == 0:
                global_approximation[self.idx_var_map[idx]] = -self.starting_value  # initialize to 2 
            else:
                global_approximation[self.idx_var_map[idx]] = self.starting_value  # initialize to 2 
            self.idx_var_map[idx].value = global_approximation[self.idx_var_map[idx]]
        # for idx in self.idx_var_map:
        #     global_approximation[self.idx_var_map[idx]] = self.idx_var_map[idx].value
        for var in global_approximation:
            print(var.name, var.value)
        
        subsystem_solutions = []
        for subsystem in self.subsystems:
            solutions = ComponentMap()
            for idx in self.idx_var_map:
                solutions[self.idx_var_map[idx]] = global_approximation[self.idx_var_map[idx]]
            subsystem_solutions.append(solutions)

        # # solve the system, have to reload the global approximate solution at each step and store
        # # the solution for each subsystem at each step
        for i in range(self.num_iterations):
            for idx, subsystem in enumerate(self.subsystems):
                solve_subsystem(self.m.find_component(subsystem), self.folder, self.solver, idx)
                print("Solve subsystem {}".format(idx))
                for var in subsystem_solutions[idx]:
                    print(var.name, " : ", var.value)
                    # save solution
                    subsystem_solutions[idx][var] = var.value
                    # reset solution
                    var.value = global_approximation[var]
            
            # update the global approximation from each subsystem
            for i in range(len(self.var_list)):
                print("Updates from subsystem {}".format(i))
                for var_idx in self.var_list[i]:
                    print(self.idx_var_map[var_idx].name, " new value = ", subsystem_solutions[i][self.idx_var_map[var_idx]])
                    self.idx_var_map[var_idx].value = subsystem_solutions[i][self.idx_var_map[var_idx]]
            
            # update the linking constraint variables
            for var in self.complicating_variables:
                var.value = sum([subsystem_solutions[i][var] for i in range(len(subsystem_solutions))])/len(subsystem_solutions)
            
            # update the global approximation
            for var in global_approximation:
                global_approximation[var] = var.value
            
            # udpate the variables
            
            print("updated global approximation")
            for var in global_approximation:
                print(var.name, " : ", var.value)
            
            print("sum of violation : ", sum(get_constraint_violation(constr) for constr in self.constr_list_all))
        assert False
    
    def test_strategy_2(self):
        # fix linking variables, all others free
        # solve subsystems
        # average the variable values
        # fix the nonlinking variables, minimize the residual of the linking variables
        global_approximation = ComponentMap()
        for idx in self.idx_var_map:
            if self.alternate and idx%2 == 0:
                global_approximation[self.idx_var_map[idx]] = -self.starting_value  # initialize to 2 
            else:
                global_approximation[self.idx_var_map[idx]] = self.starting_value  # initialize to 2 
            self.idx_var_map[idx].value = global_approximation[self.idx_var_map[idx]]
        # for idx in self.idx_var_map:
        #     global_approximation[self.idx_var_map[idx]] = self.idx_var_map[idx].value
        for var in global_approximation:
            print(var.name, var.value)
        
        subsystem_solutions = []
        for subsystem in self.subsystems:
            solutions = ComponentMap()
            for idx in self.idx_var_map:
                solutions[self.idx_var_map[idx]] = global_approximation[self.idx_var_map[idx]]
            subsystem_solutions.append(solutions)
        
        # determine variables that appear uniquely in subsystems
        # this is the set of variables that do not appear in the linking constraint subsystem
        # these we do not want to average, but update directly
        unique_var_list = [var for var in self.nonlinking_vars if var.name 
            not in [i.name for i in self.m.find_component("linking_constraint_subsystem").component_data_objects(pyo.Var)]]

        unique_var_subsystem_map = ComponentMap()
        for var in unique_var_list:
            for i in range(len(self.subsystems)):
                if var.name in [j.name for j in self.m.find_component(self.subsystems[i]).component_data_objects(pyo.Var)]:
                    unique_var_subsystem_map[var] = i

        for var in unique_var_subsystem_map:
            print(var.name, " : ", unique_var_subsystem_map[var])


        # solve the system, have to reload the global approximate solution at each step and store
        # the solution for each subsystem at each step
        for i in range(self.num_iterations):
            # fix the linking variables
            unfix_variables(self.nonlinking_vars)
            fix_variables(self.complicating_variables)

            # solve the subsystems
            for idx, subsystem in enumerate(self.subsystems):
                solve_subsystem(self.m.find_component(subsystem), self.folder, self.solver, idx)
                print("Solve subsystem {}".format(idx))
                for var in subsystem_solutions[idx]:
                    print(var.name, " : ", var.value)
                    subsystem_solutions[idx][var] = var.value
                    var.value = global_approximation[var]
            
            # update the subsystem variables - averaging them
            for var in self.nonlinking_vars:
                if var in unique_var_subsystem_map:
                    # variables unique to subsystems are not averaged because they do not show up in any of the others
                    global_approximation[var] = subsystem_solutions[unique_var_subsystem_map[var]][var]
                else:
                    global_approximation[var] = sum([subsystem_solutions[i][var] for i in range(len(subsystem_solutions))])/len(subsystem_solutions)

            # update the values
            for var in self.nonlinking_vars:
                var.value = global_approximation[var]
            
            # unfix the linking vars, minimize residuals of linking constraint system
            unfix_variables(self.complicating_variables)
            fix_variables(self.nonlinking_vars)

            for var in self.m.find_component("linking_constraint_subsystem").component_data_objects(pyo.Var):
                print(var.name, " : fixed = ", var.fixed)
                if var.fixed:
                    print(var.value)


            # solve the linking constraint subsystem
            solve_subsystem(self.m.find_component("linking_constraint_subsystem"), self.folder, self.solver, "linking_constraint")
            print("Solve linking constraint subsystem")
            print("Objective value = ", pyo.value(self.m.find_component("linking_constraint_subsystem").obj))
            for var in self.m.find_component("linking_constraint_subsystem").component_data_objects(pyo.Var):
                print(var.name, " : ", var.value)
                global_approximation[var] = var.value
            
            print("updated global approximation")
            for var in global_approximation:
                print(var.name, " : ", var.value)
            
            print("sum of violation : ", sum(get_constraint_violation(constr) for constr in self.constr_list_all))
        assert False 


        