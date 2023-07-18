"""
This file provides the algorithm for the bordered block diagonal decomposition of 
general M X N matrices
"""

class GenericDataStructure(object):
    # this implementation can change based on data structure choice
    def __init__(self):
        self.data = {}
    
    def __iter__(self):
        return self.data.__iter__()
    
    def __next__(self):
        return self.data.__next__()

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __delitem__(self, key):
        del self.data[key]

    def __eq__(self, other):
        if not isinstance(other, GenericDataStructure):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.data == other.data

    def __str__(self):
        return self.data.__str__()

    def add(self, label):
        self.data[label] = None
    
    def remove(self, label):
        del self.data[label]
        
    def size(self):
        return len(self.data)
    
    

class Vertex(object):
    def __init__(self, id):
        self.label = id
        self.adj_blocks = GenericDataStructure()
        self.size_block = 0
    
    def add_adj_block(self, label, size):
        if label not in self.adj_blocks:
            self.adj_blocks.add(label)
            self.size_block += size
    
    def remove_adj_block(self, label, size):
        if label in self.adj_blocks:
            self.adj_blocks.remove(label)
            self.size_block -= size

class VarVertex(Vertex):
    def __init__(self, id):
        super().__init__(id)
        self.adj_constr = GenericDataStructure()
        self.constr_size = {}
        self.size_block = 1
        self.size_constraints = 0
        self.single_constraint = None
        self.associated_constraint = None
        self.size_block_combined = 0
    
    def __eq__(self, other): 
        if not isinstance(other, VarVertex):
            # don't attempt to compare against unrelated types
            return NotImplemented
        if self.adj_constr != other.adj_constr:
            print("label", self.label, other.label)
            print("Adj_constr")
            print("Self = ", self.adj_constr)
            print("Other = ", other.adj_constr)
            return False 
        if self.size_block != other.size_block:
            print("label variable = ", self.label)
            print("size_block")
            print("Self size_block = ", self.size_block)
            print("other size_block = ", other.size_block)
            print(other.adj_blocks)
            return False 
        if self.size_constraints != other.size_constraints:
            print("label variable = ", self.label)
            print("Size constraints")
            print("Self size constraints = ", self.size_constraints)
            print("other size constraints = ", other.size_constraints)
            return False 
        if self.single_constraint != other.single_constraint:
            print("single constraint")
            print("Self = ", self.single_constraint)
            print("other = ", other.single_constraint)
            return False 
        if self.constr_size != other.constr_size:
            print("Label = ", self.label)
            print("constr_size")
            print("Self constr_size = ", self.constr_size)
            print("other constr_size = ", other.constr_size)
            return False 
        if self.label != other.label:
            print("label")
            return False
        if self.adj_blocks != other.adj_blocks:
            print("adj_blocks")
            return False 
        return True
    
    def set_associated_constraint(self, constraint):
        self.associated_constraint = constraint
        if self.single_constraint == None:
            self.size_block_combined = self.size_block + self.constr_size[self.associated_constraint]
        else:
            self.size_block_combined = self.size_block + self.constr_size[self.single_constraint]

    def add_adj_constr(self, label, constr_size):
        if label not in self.adj_constr:
            self.adj_constr.add(label)
            self.constr_size[label] = constr_size
            self.size_constraints += 1
    
    def remove_adj_constr(self, label):
        if label in self.adj_constr:
            self.adj_constr.remove(label)
            del self.constr_size[label]
            self.size_constraints -= 1

    def update_constr_size(self, constr, size):
        if constr in self.constr_size:
            self.constr_size[constr] = size
    
    def set_single_constraint(self, label):
        self.single_constraint = label

class ConstrVertex(Vertex):
    def __init__(self, id):
        super().__init__(id)
        self.adj_var = GenericDataStructure()
        self.size_variables = 0

    def __eq__(self, other): 
        if not isinstance(other, ConstrVertex):
            # don't attempt to compare against unrelated types
            return NotImplemented
        if self.adj_var != other.adj_var:
            print("label = ", self.label)
            print("adj_var")
            print("Self adj_var = ", self.adj_var)
            print("other adj_var = ", other.adj_var)
            return False 
        if self.size_block != other.size_block:
            print("label = ", self.label)
            print("size_block")
            print("Self size_block = ", self.size_block)
            print("other size_block = ", other.size_block)
            return False 
        if self.label != other.label:
            print("label")
            print("Self label = ", self.label)
            print("other label = ", other.label)
            return False
        if self.adj_blocks != other.adj_blocks:
            print("adj_blocks")
            print("Self adj_blocks = ", self.adj_blocks)
            print("other adj_blocks = ", other.adj_blocks)
            return False 
        if self.size_variables != other.size_variables:
            print("size_variables")
            print("Self size_variables = ", self.size_variables)
            print("other size_variables = ", other.size_variables)
            return False 
        return True

    def add_adj_var(self, label):
        if label not in self.adj_var:
            self.adj_var.add(label)
            self.size_variables += 1
    
    def remove_adj_var(self, label):
        if label in self.adj_var:
            self.adj_var.remove(label)
            self.size_variables -= 1

class Block(object):
    def __init__(self, id):
        self.label = id 
        self.var = GenericDataStructure()
        self.constr = GenericDataStructure()
        self.size = 0
        self.adj_var = GenericDataStructure()
        self.adj_constr = GenericDataStructure()
    

    def __eq__(self, other): 
        if not isinstance(other, Block):
            # don't attempt to compare against unrelated types
            return NotImplemented
        if self.var != other.var:
            print("Self var = ", self.var)
            print("other var = ", other.var)
            return False 
        if self.constr != other.constr:
            print("self constr = ", self.constr)
            print("other constr = ", other.constr)
            return False 
        if self.size != other.size:
            print("label = ", self.label)
            print("self size = ", self.size)
            print("other size = ", other.size)
            return False 
        if self.label != other.label:
            print("self label = ", self.label)
            print("other label = ", other.label)
            return False
        if self.adj_var != other.adj_var:
            print("self adj_var = ", self.adj_var)
            print("other adj_var = ", other.adj_var)
            return False 
        if self.adj_constr != other.adj_constr:
            print("self adj_constr = ", self.adj_constr)
            print("other adj_constr = ", other.adj_constr)
            return False 
        return True      

    def add_var(self, label):
        if label not in self.var:
            self.var.add(label)
            self.size += 1
    
    def add_constr(self, label):
        if label not in self.constr:
            self.constr.add(label)

    def add_var_constr(self, var_label, constr_label):
        # requires that variables and constraints added in pairs
        self.add_var(var_label)
        self.add_constr(constr_label)
    
    def add_adj_var(self, label):
        if label not in self.adj_var:
            self.adj_var.add(label)

    def add_adj_constr(self, label):
        if label not in self.adj_constr:
            self.adj_constr.add(label)

    def remove_adj_var(self, label):
        if label in self.adj_var:
            self.adj_var.remove(label)
    
    def remove_adj_constr(self, label):
        if label in self.adj_constr:
            self.adj_constr.remove(label)

class SortingStructure(object):
    def __init__(self,n):
        self.data = {1 : GenericDataStructure()}
        self.current_key = 1
        self.previous_key = 1
        self.current_order = []
        self.n = n
        self.terminate = False
    
    def __eq__(self, other): 
        if not isinstance(other, SortingStructure):
            # don't attempt to compare against unrelated types
            return NotImplemented
        if self.data != other.data:
            print("Self data = ", self.data)
            for key in self.data:
                print(key, self.data[key])
            print("other data = ", other.data)
            for key in other.data:
                print(key, other.data[key])
            return False 
        if self.current_key != other.current_key:
            print("Self current_key = ", self.current_key)
            print("other current_key = ", other.current_key)
            return False 
        if self.previous_key != other.previous_key:
            print("Self previous_key = ", self.previous_key)
            print("other previous_key = ", other.previous_key)
            return False 
        if self.current_order != other.current_order:
            print("Self current_order = ", self.current_order)
            print("other current_order = ", other.current_order)
            return False 
        return True

    def sorting_function(self, var_vertices):
        self.current_order.sort(key=lambda id: var_vertices[id].size_constraints)

    def add_data(self, key, data):
        if key not in self.data:
            self.data[key] = GenericDataStructure()
            self.data[key].add(data)
            return
        if data not in self.data[key]:
            self.data[key].add(data)
    
    def remove_data(self, key, data):
        if key not in self.data:
            raise ValueError("The predicted size does not exist in data")
        if data not in self.data[key]:
            raise ValueError("The variable does not exist in predicted size")
        del self.data[key][data]
    
    def remove_data_no_check(self, key, data):
        if key not in self.data:
            return
        if data not in self.data[key]:
            raise ValueError("The variable does not exist in predicted size")
        del self.data[key][data]


    def remove_from_current_order(self, data):
        self.current_order.remove(data)
    
    def increment_current_key(self):
        # run algorithm to exhaustion - all the variables are in the block system
        if self.current_key == self.n:
            self.terminate = True
            return
        while self.current_key not in self.data or self.data[self.current_key].size() == 0:
            self.current_key += 1
        if self.current_key == self.previous_key:
            return True
        if self.current_key != self.previous_key:
            self.previous_key = self.current_key 
            return False
    
    def create_current_order(self):
        self.current_order = [i for i in self.data[self.current_key]]
        
    def select_variable(self):
        return self.current_order.pop(0)

    def get_border_variables(self):
        border_vars = []
        for key in self.data:
            for value in self.data[key]:
                border_vars.append(value)
        return border_vars

class BBBD_algo(object):
    def create_initial_data_structures(self, edge_list, m, n, fraction):
        # edge list is a list of tuples (constr -> var)
        # m is size of rows
        # n is size of constraints
        self.m = m
        self.n = n
        self.edge_list = edge_list
        self.fraction = fraction
        self.variables = [VarVertex(i) for i in range(n)]
        self.constraints = [ConstrVertex(j) for j in range(m)]
        self.sorting_structure = SortingStructure(self.n)
        for i in range(n):
            self.sorting_structure.add_data(1, i)

        for constr, var in edge_list:
            self.variables[var].add_adj_constr(constr, 0)
            self.constraints[constr].add_adj_var(var)
        
        for var in self.variables:
            self.set_associated_constraint_var(var.label)
            # self.sorting_structure.add_data(self.variables[var.label].size_block_combined, var.label)

        for constr in self.constraints:
            if constr.size_variables == 1:
                self.variables[next(iter(constr.adj_var))].single_constraint = constr.label
        
        self.sorting_structure.create_current_order()
        self.sorting_structure.sorting_function(self.variables)
        self.blocks = GenericDataStructure()

        # for merging
        self.to_add_merged_block_to_var = set()    
        self.to_add_merged_block_to_constr = set()
        self.blocks_to_remove = set()
        self.border_vars_no_constr = []

        # for termination
        self.border_size = n

        # for border constraints
        self.constraint_in_system = [False]*m
    
    def __init__(self, edge_list, m, n, fraction):
        self.create_initial_data_structures(edge_list, m ,n, fraction)
        self.selected_variable = -1
        self.selected_constraint = -1
        self.block_label = 0

    def select_variable(self):
        self.selected_variable = self.sorting_structure.select_variable()
        # if a variable has no associated constraints available send to the border
        while self.variables[self.selected_variable].adj_constr.size() == 0:
            self.border_vars_no_constr.append(self.selected_variable)
            self.selected_variable = self.sorting_structure.select_variable()

    def get_constraint_lowest_val(self):
        if self.variables[self.selected_variable].single_constraint != None:
            self.selected_constraint = self.variables[self.selected_variable].single_constraint
            return
        self.selected_constraint = self.variables[self.selected_variable].associated_constraint

    def remove_references(self):
        # remove them from adjacency list so that they are not updated
        self.variables[self.selected_variable].remove_adj_constr(self.selected_constraint)
        self.constraints[self.selected_constraint].remove_adj_var(self.selected_variable)

    def create_block(self):
        self.blocks[self.block_label] = Block(self.block_label)
        self.blocks[self.block_label].add_var_constr(self.selected_variable, self.selected_constraint)
        self.block_label += 1
        self.constraint_in_system[self.selected_constraint] = True

    def update_block(self):
        # don't add the variables and constraints that form the block
        for adj_constr in self.variables[self.selected_variable].adj_constr:
            if adj_constr != self.selected_constraint:
                self.blocks[self.block_label-1].add_adj_constr(adj_constr)
        for adj_var in self.constraints[self.selected_constraint].adj_var:
            if adj_var != self.selected_variable:
                self.blocks[self.block_label-1].add_adj_var(adj_var)
    
    # def update_variable_sizes(self):
    #     for var in self.to_add_merged_block_to_var:
    #         self.set_associated_constraint_var(var)


    def update_sorting_structure(self):
        self.sorting_structure.remove_data(self.sorting_structure.current_key, self.selected_variable)
        # update the variables
        
        # self.sorting_structure.remove_from_current_order(self.selected_variable)
        if not self.sorting_structure.increment_current_key():
            self.sorting_structure.create_current_order()
            self.sorting_structure.sorting_function(self.variables)

    # now assumes that the constraints have been updated first
    def update_var_vertices(self):
        for var in self.to_add_merged_block_to_var:
            self.variables[var].remove_adj_constr(self.selected_constraint)
            self.variables[var].add_adj_block(self.block_label-1, self.blocks[self.block_label-1].size)

    def update_var_sorting_structure(self):
        for var in self.to_add_merged_block_to_var:
            self.set_associated_constraint_var(var)
            self.sorting_structure.add_data(self.variables[var].size_block_combined, var)

    def update_constr_vertices(self):
        for constr in self.to_add_merged_block_to_constr:
            self.constraints[constr].remove_adj_var(self.selected_variable)
            self.constraints[constr].add_adj_block(self.block_label-1, self.blocks[self.block_label-1].size)
            for var in self.constraints[constr].adj_var:
                self.variables[var].constr_size[constr] = self.constraints[constr].size_block
                for adj_block in self.constraints[constr].adj_blocks:
                    if adj_block in self.variables[var].adj_blocks:
                        self.variables[var].constr_size[constr] -= self.blocks[adj_block].size
            if self.constraints[constr].adj_var.size() == 1:
                self.variables[next(iter(self.constraints[constr].adj_var))].single_constraint = constr

    def adjust_vars_sorting_structure(self):
        for var in self.to_add_merged_block_to_var:
            # remove from data
            self.sorting_structure.remove_data(self.variables[var].size_block_combined, var)
            # remove from current_order
            if var in self.sorting_structure.current_order:
                self.sorting_structure.remove_from_current_order(var)
    
    def remove_merged_blocks(self):
        for block in self.blocks_to_remove:
            del self.blocks[block]
        
    def vars_constr_to_update(self):
        # will update everything except the selected vertex and selected constraint themselves
        # will need to update the sorting structure with the selection after merge
        self.to_add_merged_block_to_var.clear()
        self.to_add_merged_block_to_constr.clear()
        self.blocks_to_remove.clear()
        for constr in self.variables[self.selected_variable].adj_constr:
            self.to_add_merged_block_to_constr.add(constr)
        for var in self.constraints[self.selected_constraint].adj_var:
            self.to_add_merged_block_to_var.add(var)
        
        for key in self.variables[self.selected_variable].adj_blocks:
            self.blocks_to_remove.add(key)
            for adj_variable in self.blocks[key].adj_var:
                if adj_variable != self.selected_variable:
                    self.to_add_merged_block_to_var.add(adj_variable)
            for adj_constr in self.blocks[key].adj_constr:
                if adj_constr != self.selected_constraint:
                    self.to_add_merged_block_to_constr.add(adj_constr)
        for key in self.constraints[self.selected_constraint].adj_blocks:
            self.blocks_to_remove.add(key)
            if key not in self.variables[self.selected_variable].adj_blocks:
                for adj_constr in self.blocks[key].adj_constr:
                    if adj_constr != self.selected_constraint:
                        self.to_add_merged_block_to_constr.add(adj_constr)
                for adj_var in self.blocks[key].adj_var:
                    if adj_var != self.selected_variable:
                        self.to_add_merged_block_to_var.add(adj_var)


    def merge_blocks(self):        
        if self.variables[self.selected_variable].adj_blocks.size() == 0:
            return

        # merge the blocks   
        for key in self.variables[self.selected_variable].adj_blocks:
            self.merge_block(key)
        for key in self.constraints[self.selected_constraint].adj_blocks:
            if key not in self.variables[self.selected_variable].adj_blocks:
                self.merge_block(key)
        
        # update data structures
        for var in self.to_add_merged_block_to_var:
            self.variables[var].add_adj_block(self.block_label-1, self.blocks[self.block_label-1].size)
        for constr in self.to_add_merged_block_to_constr:
            self.constraints[constr].add_adj_block(self.block_label-1, self.blocks[self.block_label-1].size)

    
    def merge_block(self, block):
        for variable in self.blocks[block].var:
            if variable != self.selected_variable:
                self.blocks[self.block_label-1].add_var(variable)
        for constr in self.blocks[block].constr:
            if constr != self.selected_constraint:
                self.blocks[self.block_label-1].add_constr(constr)
        for adj_variable in self.blocks[block].adj_var:
            if adj_variable != self.selected_variable:
                self.blocks[self.block_label-1].add_adj_var(adj_variable)
                self.variables[adj_variable].remove_adj_block(block, self.blocks[block].size)
        for adj_constr in self.blocks[block].adj_constr:
            if adj_constr != self.selected_constraint:
                self.blocks[self.block_label-1].add_adj_constr(adj_constr)
                self.constraints[adj_constr].remove_adj_block(block, self.blocks[block].size)

    def iteration(self):
        self.select_variable()
        self.get_constraint_lowest_val()
        self.create_block() # increases the block label by 1
        self.remove_references()
        self.vars_constr_to_update()
        self.adjust_vars_sorting_structure()
        self.update_block()
        self.merge_blocks()
        self.update_var_vertices()
        self.update_constr_vertices()
        self.update_var_sorting_structure()
        self.update_sorting_structure()
        self.remove_merged_blocks()
        self.border_size -= 1
  
    def set_associated_constraint_var(self, var_index):
        self.variables[var_index].set_associated_constraint(sorted(list(self.variables[var_index].constr_size.keys()),
                key=lambda id: (self.variables[var_index].constr_size[id], self.constraints[id].size_variables)
            )[0])

    def solve(self):
        while not self.termination_criteria():
            self.iteration()
        return self.get_column_row_order()

    def termination_criteria(self):
        # a block created will be greater than some fraction of size
        # of the original matrix
        return self.sorting_structure.current_key > self.fraction*self.n
        # other termination criteria could be a limit on the border size
        # return self.border_size <= self.border_threshold
    
    def get_column_row_order(self):
        # get the reordering of the columns and rows
        column_order = []
        row_order = []
        for block in self.blocks:
            column_order += [item for item in self.blocks[block].var]
            row_order += [item for item in self.blocks[block].constr]
        # get remaining border variables and columns
        border_vars = self.sorting_structure.get_border_variables()
        border_constr = [i for i in range(len(self.constraint_in_system)) if not self.constraint_in_system[i]]
        
        column_order += border_vars 
        row_order += border_constr
        column_order += self.border_vars_no_constr

        return column_order, row_order


    




