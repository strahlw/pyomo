"""
This script tests model interface util functions
"""
from pyomo.contrib.incidence_restructuring.model_interface_util import (
    invert_order, reorder_sparse_matrix,
    get_restructured_matrix, update_keys, get_restructured_mapping,
    get_perfect_match_mapping, get_mappings_to_original, reformat_blocks,
    create_subsystem_from_block, create_subsystems, solve_subsystems_sequential,
    solve_subsystems_sequential_independent, create_subfolder, 
    filter_small_blocks
)
import pyomo.environ as pyo
import scipy as sc
import pyomo.common.unittest as unittest
import pytest as pytest

def test_invert_order():
    array1 = [2,1,3,5,4,0]
    order = [3,2,5,1,0,4]

    result = invert_order(array1, order)
    expected = [2, 0, 3, 1, 5, 4]

    for i in range(len(result)):
        assert result[i] == expected[i]
    
def test_reorder_sparse_matrix():
    matrix = sc.sparse.coo_matrix([[1,0,1,0], [0,1,0,0],[0,0,1,0], [0,1,0,1]])
    assert (matrix.getnnz() == 6)

    row_order = [0,2,3,1]
    col_order = [0,2,1,3]

    final_matrix = reorder_sparse_matrix(4, row_order, col_order, matrix)
    assert (matrix.getnnz() == final_matrix.getnnz())

    expected_final_matrix = sc.sparse.coo_matrix([[1,1,0,0],[0,1,0,0], [0,0,1,1], [0,0,1,0]])

    nz_rows, nz_cols = final_matrix.nonzero()
    nz_rows_matrix, nz_cols_matrix = expected_final_matrix.nonzero()

    final_matrix_nz = zip(nz_rows, nz_cols)
    expected_matrix_nz = zip(nz_rows_matrix, nz_cols_matrix)
    for i in range(len(nz_rows)):
        assert nz_rows[i] == nz_rows_matrix[i]
        assert nz_cols[i] == nz_cols_matrix[i]

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

        # only 2 partitions for simplicity
        self.order, self.blocks, self.col_map, self.method, self.idx_var_map, \
            self.idx_constr_map, self.block_indices = get_restructured_matrix(self.m, method=2, num_part=2)
    
    # self.order is the reordering of rows and columns for BBBD structure from perfect matching
    # self.col_map is the reordering of columns from original matrix to perfectly matched matrix
    # goal of this test is to design functions that take the final indices assigned to the BBBD
    # structure and back out or extract the original variable and constraint indices of the model

    def test_update_keys(self):
        dictionary = {0: "x1", 1: "x2", 2: "x3", 3: "x4", 4: "x5"}
        new_order = [0,3,2,4,1]

        computed_result = update_keys(dictionary, new_order)
        expected_result= {0: "x1", 1: "x4", 2: "x3", 3: "x5", 4: "x2"}

        for key in computed_result:
            assert computed_result[key] == expected_result[key]
    
    def test_get_perfect_match_mapping(self):
        perfect_matching_map_computed = get_perfect_match_mapping(self.idx_var_map, self.col_map)
        perfect_matching_map_expected = {0: "x4", 1: "x1", 2: "x3", 3: "x2"}
        for key in perfect_matching_map_computed:
            assert perfect_matching_map_expected[key] == perfect_matching_map_computed[key].name
        self.idx_var_map = perfect_matching_map_computed

    def test_get_restructured_mapping(self):
        self.test_get_perfect_match_mapping()
        
        # restructured_matching_var_computed = update_keys(self.idx_var_map, self.order)
        # restructured_matching_constr_computed = update_keys(self.idx_constr_map, self.order)
        restructured_matching_constr_computed, restructured_matching_var_computed = \
            get_restructured_mapping(self.idx_var_map, self.idx_constr_map, self.order)

        restructured_matching_constr_expected = {0: "cons1", 1: "cons4", 2: "cons2", 3: "cons3"}
        restructured_matching_var_expected = {0: "x4", 1: "x2", 2: "x1", 3: "x3"}

        for key in restructured_matching_var_computed:
            assert restructured_matching_var_computed[key].name == restructured_matching_var_expected[key]
        
        for key in restructured_matching_constr_computed:
            assert restructured_matching_constr_computed[key].name == restructured_matching_constr_expected[key]
    
    def test_get_mappings_to_original(self):
        original_matching_constr_computed, original_matching_var_computed = \
            get_mappings_to_original(self.order, self.col_map, self.idx_var_map, self.idx_constr_map)

        original_matching_constr_expected = {0: "cons1", 1: "cons2", 2: "cons3", 3: "cons4"}
        original_matching_var_expected = {0: "x4", 1: "x1", 2: "x3", 3: "x2"}

        for key in original_matching_var_computed:
            assert original_matching_var_computed[key].name == original_matching_var_expected[key]
        
        for key in original_matching_constr_computed:
            assert original_matching_constr_computed[key].name == original_matching_constr_expected[key]

class TestBlockDecompositionForSolver():
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

        # only 2 partitions for simplicity
        self.order, self.blocks, self.col_map, self.method, self.idx_var_map, \
            self.idx_constr_map, self.border_indices = get_restructured_matrix(self.m, method=request.param, num_part=2, d_max=1, n_max=2)

        self.original_mapping_constr, self.original_mapping_var = get_mappings_to_original(self.order, self.col_map, 
                                                                        self.idx_var_map, self.idx_constr_map)
    # @pytest.mark.parametrize(
    #     'method',
    #     ([1, 2]),
    #     indirect=True
    # )
    def test_reformat_blocks(self):
        resulting_blocks = reformat_blocks(self.method, self.blocks)
        expected_blocks = [[1,2],[3,5]]

        # test equivalency regardless of order
        for i in resulting_blocks:
            assert i in expected_blocks
        for i in expected_blocks:
            assert i in resulting_blocks

    def test_create_subsystem_from_block(self):
        block = [1,2]
        subsystem = create_subsystem_from_block(block, self.original_mapping_constr)
        
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

    # @pytest.mark.parametrize(
    #     'method',
    #     ([1, 2]),
    #     indirect=True
    # )
    def test_create_subsystems(self):
        subsystems = create_subsystems(reformat_blocks(self.method, self.blocks), self.original_mapping_constr)
        assert len(subsystems) == 2
        for subsystem in subsystems:
            variables = [var for var in subsystem.component_data_objects(pyo.Var)]
            constraints = [constr for constr in subsystem.component_data_objects(pyo.Constraint)]

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

    # @pytest.mark.parametrize(
    #     'method',
    #     ([1, 2]),
    #     indirect=True
    # )
    def test_solve_subsystems_sequential(self):
        subsystems = create_subsystems(reformat_blocks(self.method, self.blocks), self.original_mapping_constr)
        assert len(subsystems) == 2
        idx_test_case = [idx for idx, i in enumerate(subsystems) if "x2" in [j.name for j in i.component_data_objects(pyo.Var)]][0]
        variables = [var for var in subsystems[idx_test_case].component_data_objects(pyo.Var)]
        constraints = [constr for constr in subsystems[idx_test_case].component_data_objects(pyo.Constraint)]
        # print([i.name for i in variables])
        # print([i.value for i in variables])
        # print([i.name for i in constraints])

        variables[2].value = None

        solved = solve_subsystems_sequential([subsystems[idx_test_case]])
        
        assert solved[0] == True
        assert variables[0].value == 0.5
    
    # @pytest.mark.parametrize(
    #     'method',
    #     ([1, 2]),
    #     indirect=True
    # )
    def test_data_collection(self):
        blocks = [[0]*9, [0]*100, [0]*30]
        # print([len(i) for i in blocks])
        # assert False
        
        computed_blocks = filter_small_blocks(blocks)
        expected_blocks = blocks

        # for i in blocks:
        #     print(len(i))
        # assert False

        for i in computed_blocks:
            assert i in expected_blocks 
        for i in expected_blocks:
            if len(i) >= 10:
                assert i in computed_blocks

    def test_solve_subsystems_sequential_independent(self):
        subsystems = create_subsystems(reformat_blocks(self.method, self.blocks), self.original_mapping_constr)
        assert len(subsystems) == 2
        create_subfolder("test_folder")
        solved = solve_subsystems_sequential_independent(subsystems, self.border_indices, self.idx_var_map, "test_folder")
        
        # print("Border Indices = ", self.border_indices)

        for i in range(len(solved)):
            assert solved[i] == True
        
        # assert variables[0].value == 0.5
        print("Border Variables ", [self.idx_var_map[index].name for index in self.border_indices])
        print([i.name for i in self.m.component_data_objects(pyo.Var)])
        print([i.value for i in self.m.component_data_objects(pyo.Var)])

        assert True

