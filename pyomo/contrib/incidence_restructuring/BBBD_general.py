"""
This file provides the algorithm for the bordered block diagonal decomposition of 
general M X N matrices
"""

class GenericDataStructure(object):
    # this implementation can change based on data structure choice
    def __init__(self):
        self.data = {}
        #self.in_data = [False]*size
    
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
    
    def check_in(self, key):
        if self.data.get(key, -1) == -1:
            return False
        return True
    
class Vertex(object):
    def __init__(self, id):
        self.label = id
        self.adj_blocks = GenericDataStructure()
        self.size_block = 0
        self.largest_adj_block = None
        self.size_largest_block = 0
    
    def add_adj_block(self, label, size):
        if not self.adj_blocks.check_in(label):
        # if label not in self.adj_blocks:
            self.adj_blocks.add(label)
            self.size_block += size
            if size >= self.size_largest_block:
                # if merged, the resulting block will always be larger
                # so it monotonically increases, and there will always be a label
                self.largest_adj_block = label
                self.size_largest_block = size
    
    def remove_adj_block(self, label, size):
        if self.adj_blocks.check_in(label):
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
            print("Label = ", self.label)
            print("adj_blocks")
            print("Self adj_blocks = ", self.adj_blocks)
            print("other adj_blocks = ", other.adj_blocks)
            return False 
        return True
    
    def set_associated_constraint(self, constraint, single_constraint_in_system):
        # print("variable = ", self.label)
        # print(constraint, single_constraint_in_system)
        if constraint == None:
            self.size_block_combined = self.size_block
            return
        self.associated_constraint = constraint
        if self.single_constraint == None or single_constraint_in_system:
            self.size_block_combined = self.size_block + self.constr_size[self.associated_constraint]
        else:
            self.size_block_combined = self.size_block + self.constr_size[self.single_constraint]

    def add_adj_constr(self, label, constr_size):
        if not self.adj_constr.check_in(label):
            self.adj_constr.add(label)
            self.constr_size[label] = constr_size
            self.size_constraints += 1
    
    def add_adj_constr_no_size(self, label, constr_size):
        if not self.adj_constr.check_in(label):
            self.adj_constr.add(label)
            self.constr_size[label] = constr_size
            # self.size_constraints += 1

    def remove_adj_constr(self, label):
        if self.adj_constr.check_in(label):
            self.adj_constr.remove(label)
            del self.constr_size[label]
            self.size_constraints -= 1

    def update_constr_size(self, constr, size):
        if constr in self.constr_size:
            self.constr_size[constr] = size
    
    def set_single_constraint(self, label):
        self.single_constraint = label

    def print_data(self):
        print("Label = ", self.label)
        print("adj_constr = ", [i for i in self.adj_constr])
        print("adj blocks = ", [i for i in self.adj_blocks])
        print("size variables = ", self.size_constraints)
        print("size block = ", self.size_block)
        print("constraint size = ", self.constr_size)
        print("size combined = ", self.size_block_combined)
        print("single constraint = ", self.single_constraint)
        print("largest adj_block = ", self.largest_adj_block)
        print("associated constraint = ", self.associated_constraint)


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
            print("label = ", self.label)
            print("adj_blocks")
            print("Self adj_blocks = ", self.adj_blocks)
            print("other adj_blocks = ", other.adj_blocks)
            return False 
        if self.size_variables != other.size_variables:
            print("label = ", self.label)
            print("size_variables")
            print("Self size_variables = ", self.size_variables)
            print("other size_variables = ", other.size_variables)
            return False 
        return True

    def add_adj_var(self, label):
        if not self.adj_var.check_in(label):
            self.adj_var.add(label)
            self.size_variables += 1
    
    def remove_adj_var(self, label):
        if self.adj_var.check_in(label):
            self.adj_var.remove(label)
            self.size_variables -= 1

    
    def print_data(self):
        print("Label = ", self.label)
        print("adj_var = ", [i for i in self.adj_var])
        print("adj blocks = ", [i for i in self.adj_blocks])
        print("size variables = ", self.size_variables)
        print("size block = ", self.size_block)
        print("largest adj_block = ", self.largest_adj_block)

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
        if not self.var.check_in(label):
            self.var.add(label)
            self.size += 1
    
    def add_constr(self, label):
        if not self.constr.check_in(label):
            self.constr.add(label)

    def add_var_constr(self, var_label, constr_label):
        # requires that variables and constraints added in pairs
        self.add_var(var_label)
        self.add_constr(constr_label)
    
    def add_adj_var(self, label):
        if not self.adj_var.check_in(label):
            self.adj_var.add(label)

    def add_adj_constr(self, label):
        if not self.adj_constr.check_in(label):
            self.adj_constr.add(label)

    def remove_adj_var(self, label):
        if self.adj_var.check_in(label):
            self.adj_var.remove(label)
    
    def remove_adj_constr(self, label):
        if self.adj_constr.check_in(label):
            self.adj_constr.remove(label)
    
    def print_data(self):
        print("Block label = ", self.label)
        print("Variables = ", [i for i in self.var])
        print("Constr = ", [i for i in self.constr])
        print("Adj Var = ", [i for i in self.adj_var])
        print("Adj constr = ", [i for i in self.adj_constr])
        print("size = ", self.size)

class SortingStructure(object):
    def __init__(self,n, fraction):
        self.data = {1 : GenericDataStructure()}
        self.current_key = 1
        self.previous_key = 1
        self.current_order = []
        self.n = n
        self.terminate = False
        self.in_current_order = [False]*n
        self.fraction = fraction
        self.print_sorting_data = False

        #self.set_printing_data()
    
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

    def set_printing_data(self):
        self.print_sorting_data = True
        self.print_data()

    def sorting_function(self, var_vertices):
        self.current_order.sort(key=lambda id: var_vertices[id].size_constraints)

    def add_data(self, key, data):
        if key not in self.data:
            self.data[key] = GenericDataStructure()
            self.data[key].add(data)
            return
        if not self.data[key].check_in(data):
            self.data[key].add(data)
    
    def remove_data(self, key, data):
        if key not in self.data:
            raise ValueError("The predicted size does not exist in data")
        if not self.data[key].check_in(data):
            self.print_data()
            raise ValueError("The variable {} does not exist in predicted size {}".format(data, key))
        del self.data[key][data]
    
    def remove_data_no_check(self, key, data):
        if key not in self.data:
            return
        if not self.data[key].check_in(data):
            return
            #raise ValueError("The variable does not exist in predicted size")
        del self.data[key][data]


    def remove_from_current_order(self, data):
        self.in_current_order[data] = False
        #self.current_order.remove(data)
    
    def increment_current_key(self):
        # run algorithm to exhaustion - all the variables are in the block system
        if self.current_key >= self.n:
            if self.print_sorting_data:
                print("Terminate on current key")
            self.terminate = True
            return True
        if self.current_key >= self.fraction*self.n:
            if self.print_sorting_data:
                print("Terminate on fraction increment current key 1")
            self.terminate = True 
            return True
        while self.current_key not in self.data or self.data[self.current_key].size() == 0:
            # print("loop 358")
            if self.current_key >= self.n:
                if self.print_sorting_data:
                    print("Terminate on current key in while loop")
                self.terminate = True
                return True
            if self.current_key >= self.fraction*self.n:
                if self.print_sorting_data:
                    print("Terminate on fraction increment current key 2")
                self.terminate = True 
                return True
            self.current_key += 1
        # no update
        if self.current_key == self.previous_key:
            return True
        # updates
        if self.current_key != self.previous_key:
            self.previous_key = self.current_key 
            return False
    
    def create_current_order(self):
        self.current_order = [i for i in self.data[self.current_key]]
        self.in_current_order = [False]*self.n
        for i in self.data[self.current_key]:
            self.in_current_order[i] = True

    def get_in_current_order(self, var):
        return self.in_current_order[var]
            
    def select_variable(self):
        variable = self.current_order.pop(0)
        while not (self.in_current_order[variable]):
            if len(self.current_order) == 0:
                self.increment_current_key()
                self.create_current_order()
                if self.terminate:
                    return -5
            variable = self.current_order.pop(0)
        return variable

    def get_border_variables(self):
        border_vars = []
        for key in self.data:
            for value in self.data[key]:
                border_vars.append(value)
        return border_vars

    def print_data(self):
        print("size order")
        for key in self.data:
            print("key = ", key, " data = ", self.data[key])
        print("current order = ", self.current_order)
        print("current key = ", self.current_key)
        print("previous key = ", self.previous_key)
        print("in current order = ", self.in_current_order)
        print("fraction = ", self.fraction)
        print("num vars = ", self.n)
        print()

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
        self.sorting_structure = SortingStructure(self.n, self.fraction)
        # for border constraints
        self.constraint_in_system = [False]*m
        self.variable_in_system = [False]*n
        #self.variable_in_border = [False]*n


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
        self.border_vars_no_constr = {}

        # for termination
        self.border_size = n
        self.num_border_vars = 0
        self.num_sys_vars = 0
        self.terminate = False


        # for debugging
        self.iteration_counter = 0

        self.print_data = True
        if self.print_data:
            self.sorting_structure.set_printing_data()
        self.check_assertions = True
    
    def __init__(self, edge_list, m, n, fraction):
        self.create_initial_data_structures(edge_list, m ,n, fraction)
        self.selected_variable = -1
        self.selected_constraint = -1
        self.block_label = 0
        if self.print_data:
            print("Initial states")
            self.print_data_objects()
    
    def set_seed_block(self):
        if self.variables[self.selected_variable].largest_adj_block == None:
            size_var = -1
        else:
            size_var = self.blocks[self.variables[self.selected_variable].largest_adj_block].size

        if self.constraints[self.selected_constraint].largest_adj_block == None:
            size_constr = -1
        else:
            size_constr = self.blocks[self.constraints[self.selected_constraint].largest_adj_block].size
        
        if size_var == -1 and size_constr == -1:
            self.seed_block = self.block_label
            return

        if size_var == size_constr:
            self.seed_block = min(self.variables[self.selected_variable].largest_adj_block, 
                                  self.constraints[self.selected_constraint].largest_adj_block)
            return
        if size_var > size_constr:
            self.seed_block = self.variables[self.selected_variable].largest_adj_block
        else:
            self.seed_block = self.constraints[self.selected_constraint].largest_adj_block
    
    def check_sorting_structure(self):
        if len(self.sorting_structure.current_order) == 0:
            self.sorting_structure.increment_current_key()
            if self.sorting_structure.terminate:
                self.terminate = True
                return
            self.sorting_structure.create_current_order()
            self.sorting_structure.sorting_function(self.variables)

    def select_variable(self):
        self.selected_variable = self.sorting_structure.select_variable()
        # if a variable has no associated constraints available send to the border
        while self.variables[self.selected_variable].adj_constr.size() == 0:

            # self.num_border_vars += 1
            # if self.check_assertions:
            #     assert self.selected_variable not in self.border_vars_no_constr
            self.border_vars_no_constr[self.selected_variable] = None
            # # remove the reference to the border variable so that it is not updated in merges
            # for block in self.variables[self.selected_variable].adj_blocks:
            #     self.blocks[block].remove_adj_var(self.selected_variable)
            # should only be in current order will not be re-added to the variable selection queue
            self.sorting_structure.remove_data(self.sorting_structure.current_key, self.selected_variable)
            self.check_sorting_structure()
            if self.sorting_structure.terminate:
                self.terminate = True
                break
            self.selected_variable = self.sorting_structure.select_variable()

    def get_constraint_lowest_val(self):
        # if self.variables[self.selected_variable].single_constraint != None and not self.constraint_in_system[self.variables[self.selected_variable].single_constraint]:
        #     self.selected_constraint = self.variables[self.selected_variable].single_constraint
        #     return
        self.selected_constraint = self.variables[self.selected_variable].associated_constraint

    def remove_references(self):
        # remove them from adjacency list so that they are not updated
        self.variables[self.selected_variable].remove_adj_constr(self.selected_constraint)
        self.constraints[self.selected_constraint].remove_adj_var(self.selected_variable)

    def create_block(self):
        self.blocks[self.block_label] = Block(self.block_label)
        #self.blocks[self.block_label].add_var_constr(self.selected_variable, self.selected_constraint)
        self.block_label += 1

    def update_block(self):
        # don't add the variables and constraints that form the block
        for adj_constr in self.variables[self.selected_variable].adj_constr:
            # may be redundant
            if adj_constr != self.selected_constraint and not self.blocks[self.seed_block].constr.check_in(adj_constr):
                self.blocks[self.seed_block].add_adj_constr(adj_constr)
                for var in self.blocks[self.seed_block].adj_var:
                    # if var == 3:
                    #     print("Size of var 3 remove 1 = ", self.variables[var].size_block_combined)
                    #     self.sorting_structure.print_data()
                    if not self.variables[var].adj_constr.check_in(adj_constr) and var != self.selected_variable:
                        self.variables[var].add_adj_constr(adj_constr, self.constraints[adj_constr].size_block)
                        self.constraints[adj_constr].add_adj_var(var)    
                        for adj_block in self.constraints[adj_constr].adj_blocks:
                            if self.variables[var].adj_blocks.check_in(adj_block):
                                self.variables[var].constr_size[adj_constr] -= self.blocks[adj_block].size
        
        for var in self.blocks[self.seed_block].adj_var:
            if var != self.selected_variable:
                self.set_associated_constraint_var(var)
            # if var == 3:
            #     print("Size of var 3 add = ", self.variables[var].size_block_combined)
            #     self.sorting_structure.print_data()
            # self.sorting_structure.add_data(self.variables[var].size_block_combined, var)

        # update the variables       
        for adj_var in self.constraints[self.selected_constraint].adj_var:
            if adj_var != self.selected_variable and not self.blocks[self.seed_block].var.check_in(adj_var):
                self.blocks[self.seed_block].add_adj_var(adj_var)
                for adj_constr in self.blocks[self.seed_block].adj_constr:
                    if not self.variables[adj_var].adj_constr.check_in(adj_constr):  
                        self.variables[adj_var].add_adj_constr(adj_constr, self.constraints[adj_constr].size_block)
                        self.constraints[adj_constr].add_adj_var(adj_var)    
                        for adj_block in self.constraints[adj_constr].adj_blocks:
                            if self.variables[adj_var].adj_blocks.check_in(adj_block):
                                self.variables[adj_var].constr_size[adj_constr] -= self.blocks[adj_block].size
                self.set_associated_constraint_var(adj_var)


    def update_sorting_structure(self):
        self.sorting_structure.remove_data(self.sorting_structure.current_key, self.selected_variable)
        # update the variables
        
        # self.sorting_structure.remove_from_current_order(self.selected_variable)
        if not self.sorting_structure.increment_current_key():
            self.sorting_structure.create_current_order()
            self.sorting_structure.sorting_function(self.variables)

    # now assumes that the constraints have been updated first
    # all variables have been added to the seed block
    def update_var_vertices(self):
        for var in self.to_add_merged_block_to_var:
            self.variables[var].remove_adj_constr(self.selected_constraint)
            self.variables[var].add_adj_block(self.seed_block, self.blocks[self.seed_block].size)

    def update_var_sorting_structure(self):
        for var in self.to_add_merged_block_to_var:
            if var != self.selected_variable:
                self.set_associated_constraint_var(var)
            self.sorting_structure.add_data(self.variables[var].size_block_combined, var)

    # all variables have to be added to the seed block first
    def update_constr_vertices(self):
        for constr in self.to_add_merged_block_to_constr:
            if constr != self.selected_constraint:
                self.constraints[constr].remove_adj_var(self.selected_variable)
                self.constraints[constr].add_adj_block(self.seed_block, self.blocks[self.seed_block].size)
            # if self.constraints[constr].adj_var.size() == 1:
            #     self.variables[next(iter(self.constraints[constr].adj_var))].single_constraint = constr
        
        for constr in self.to_add_merged_block_to_constr:
            if constr != self.selected_constraint:
                for var in self.constraints[constr].adj_var:
                    if var not in self.to_add_merged_block_to_var and var != self.selected_variable:
                        tmp = self.variables[var].size_block_combined
                        self.sorting_structure.remove_data(self.variables[var].size_block_combined, var)
                    self.variables[var].constr_size[constr] = self.constraints[constr].size_block
                    for adj_block in self.constraints[constr].adj_blocks:
                        if self.variables[var].adj_blocks.check_in(adj_block):
                            self.variables[var].constr_size[constr] -= self.blocks[adj_block].size
                    # if self.check_assertions:
                    #     assert self.selected_variable in self.to_add_merged_block_to_var
                    if var not in self.to_add_merged_block_to_var and var != self.selected_variable:
                        self.set_associated_constraint_var(var)
                        self.sorting_structure.add_data(self.variables[var].size_block_combined, var)
                        if tmp < self.variables[var].size_block_combined:
                            self.sorting_structure.remove_from_current_order(var)

    def adjust_vars_sorting_structure(self):
        for var in self.to_add_merged_block_to_var:
            # check if the border variables have adjacent constraints
            if var in self.border_vars_no_constr:
                if self.variables[var].adj_constr.size() != 0:
                    del self.border_vars_no_constr[var]
            # remove from data if not already removed
            if var not in self.border_vars_no_constr and var != self.selected_variable:
                self.sorting_structure.remove_data(self.variables[var].size_block_combined, var)
            # remove from current_order
            if self.sorting_structure.get_in_current_order(var):
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
        # means that we aren't creating a block in this iteration
        if self.seed_block != self.block_label:
            for adj_var in self.blocks[self.seed_block].adj_var:
                if adj_var != self.selected_variable:
                    self.to_add_merged_block_to_var.add(adj_var)
            for adj_constr in self.blocks[self.seed_block].adj_constr:
                if adj_constr != self.selected_constraint:
                    self.to_add_merged_block_to_constr.add(adj_constr)

        for key in self.variables[self.selected_variable].adj_blocks:
            if key != self.seed_block:
                self.blocks_to_remove.add(key)
            for adj_variable in self.blocks[key].adj_var:
                if adj_variable != self.selected_variable:
                    self.to_add_merged_block_to_var.add(adj_variable)
            for adj_constr in self.blocks[key].adj_constr:
                if adj_constr != self.selected_constraint:
                    self.to_add_merged_block_to_constr.add(adj_constr)
        for key in self.constraints[self.selected_constraint].adj_blocks:
            if key != self.seed_block:
                self.blocks_to_remove.add(key)
            if not self.variables[self.selected_variable].adj_blocks.check_in(key):
                for adj_constr in self.blocks[key].adj_constr:
                    if adj_constr != self.selected_constraint:
                        self.to_add_merged_block_to_constr.add(adj_constr)
                for adj_var in self.blocks[key].adj_var:
                    if adj_var != self.selected_variable:
                        self.to_add_merged_block_to_var.add(adj_var)
        
        if self.check_assertions:
            assert self.selected_variable not in self.to_add_merged_block_to_var

    def add_variable_constr_to_block(self):
        self.constraint_in_system[self.selected_constraint] = True
        self.variable_in_system[self.selected_variable] = True
        self.blocks[self.seed_block].add_var(self.selected_variable)
        self.blocks[self.seed_block].add_constr(self.selected_constraint)
        self.blocks[self.seed_block].remove_adj_var(self.selected_variable)
        self.blocks[self.seed_block].remove_adj_constr(self.selected_constraint)

    def merge_blocks(self):
        # variable and constraint added to system
        if (self.variables[self.selected_variable].adj_blocks.size() == 0 and \
            self.constraints[self.selected_constraint].adj_blocks.size() == 0) or \
                self.seed_block == self.block_label:
            # here is where we create a block
            self.create_block()
            self.add_variable_constr_to_block()
            return
       
        
        self.remove_adj_block_seed()
        self.add_variable_constr_to_block()
        # merge the blocks
        for key in self.blocks_to_remove:   
            self.merge_block(key)
        # for key in self.variables[self.selected_variable].adj_blocks:
        #     self.merge_block(key)
        # for key in self.constraints[self.selected_constraint].adj_blocks:
        #     if key not in self.variables[self.selected_variable].adj_blocks:
        #         self.merge_block(key)
        
        # update data structures
        for var in self.to_add_merged_block_to_var:
            self.variables[var].add_adj_block(self.seed_block, self.blocks[self.seed_block].size)
        for constr in self.to_add_merged_block_to_constr:
            self.constraints[constr].add_adj_block(self.seed_block, self.blocks[self.seed_block].size)
    
    def merge_block(self, block):
        for variable in self.blocks[block].var:
            if variable != self.selected_variable:
                self.blocks[self.seed_block].add_var(variable)
                # if variable in self.blocks[self.seed_block].adj_var:
                #     self.blocks[self.seed_block].remove_adj_var(variable)
        for constr in self.blocks[block].constr:
            if constr != self.selected_constraint:
                self.blocks[self.seed_block].add_constr(constr)
                # if constr in self.blocks[self.seed_block].adj_constr:
                #     self.blocks[self.seed_block].remove_adj_constr(constr)
                
        for adj_constr in self.blocks[block].adj_constr:
            if adj_constr != self.selected_constraint and not self.blocks[self.seed_block].constr.check_in(adj_constr):
                self.blocks[self.seed_block].add_adj_constr(adj_constr)
                self.constraints[adj_constr].remove_adj_block(block, self.blocks[block].size)
                # now we update the adjacencies 
                for var in self.blocks[self.seed_block].adj_var:
                    if not self.variables[var].adj_constr.check_in(adj_constr) and var != self.selected_variable:
                        self.variables[var].add_adj_constr(adj_constr, self.constraints[adj_constr].size_block)    
                        self.constraints[adj_constr].add_adj_var(var)    
                        for adj_block in self.constraints[adj_constr].adj_blocks:
                            if self.variables[var].adj_blocks.check_in(adj_block):
                                self.variables[var].constr_size[adj_constr] -= self.blocks[adj_block].size
        
        for var in self.blocks[self.seed_block].adj_var:
            if var != self.selected_variable:
                self.set_associated_constraint_var(var)
            #ASSERT there are is only one of each variable index in the sorting structure
                
        for adj_variable in self.blocks[block].adj_var:
            if adj_variable != self.selected_variable and not self.blocks[self.seed_block].var.check_in(adj_variable):
                self.blocks[self.seed_block].add_adj_var(adj_variable)
                self.variables[adj_variable].remove_adj_block(block, self.blocks[block].size)
                for adj_constr in self.blocks[self.seed_block].adj_constr:
                    if not self.variables[adj_variable].adj_constr.check_in(adj_constr):  
                        self.variables[adj_variable].add_adj_constr(adj_constr, self.constraints[adj_constr].size_block)    
                        self.constraints[adj_constr].add_adj_var(adj_variable)    
                        for adj_block in self.constraints[adj_constr].adj_blocks:
                            if self.variables[adj_variable].adj_blocks.check_in(adj_block):
                                self.variables[adj_variable].constr_size[adj_constr] -= self.blocks[adj_block].size
                self.set_associated_constraint_var(adj_variable)

    def remove_adj_block_seed(self):
        for adj_variable in self.blocks[self.seed_block].adj_var:
            if adj_variable != self.selected_variable:
                self.variables[adj_variable].remove_adj_block(self.seed_block, self.blocks[self.seed_block].size)
        for adj_constr in self.blocks[self.seed_block].adj_constr:
            if adj_constr != self.selected_constraint:
                self.constraints[adj_constr].remove_adj_block(self.seed_block, self.blocks[self.seed_block].size)


    def print_data_objects(self):
        print("\nVariable data\n")
        for var in self.variables:
            var.print_data()
        print("\nConstraint Data\n")
        for constr in self.constraints:
            constr.print_data()
        print("\nBlock Data\n")
        for block in self.blocks:
            self.blocks[block].print_data()
        print("\nSorting structure data\n")
        self.sorting_structure.print_data()

    def iteration(self):
        # print("num_sys vars = ", self.num_sys_vars)
        # print("num_border_vars = ", self.num_border_vars)
        if self.num_sys_vars + self.num_border_vars == self.n:
            if self.print_data:
                print("Terminate on number of variables")
            self.terminate = True 
            return
        self.iteration_counter += 1
        if self.print_data:
            print("Iteration {}".format(self.iteration_counter))
        self.check_sorting_structure()
        self.terminate |= self.sorting_structure.terminate
        if self.terminate:
            return 
        self.select_variable()
        self.terminate |= self.sorting_structure.terminate
        if self.terminate:
            return
        if self.sorting_structure.current_key >= self.fraction*self.n:
            self.terminate = True
            return
        #print("selected variable = ", self.selected_variable)
    
        if self.print_data:
            print("\n Variable and Constraint Selection\n")
            print("selected variable = ", self.selected_variable)
        self.get_constraint_lowest_val()
        if self.print_data:
            print("selected constraint = ", self.selected_constraint)

        if self.variables[self.selected_variable].size_block_combined != self.sorting_structure.current_key:
            print("ASSERTION ERROR")
            print("size block combined = ", self.variables[self.selected_variable].size_block_combined)
            print("Sorting structure_key = ", self.sorting_structure.current_key)
            
        assert self.variables[self.selected_variable].size_block_combined == self.sorting_structure.current_key
        #self.create_block()
        self.num_sys_vars += 1
        # increases the block label by 1
        self.set_seed_block()
        self.remove_references()
        self.vars_constr_to_update()
        self.adjust_vars_sorting_structure()
        self.merge_blocks()
        self.update_block()
        self.update_var_vertices()
        self.update_constr_vertices()
        self.update_var_sorting_structure()
        self.remove_merged_blocks()
        if self.num_sys_vars + self.num_border_vars != self.n:
            self.update_sorting_structure()
        self.border_size -= 1
        if self.print_data:
            self.print_data_objects()
        
        # if self.blocks[self.seed_block].size != self.variables[self.selected_variable].size_block_combined:
        #     print("BLOCK SIZE AND PREDICTED MISMATCH")
        #     print("selected variable = ", self.selected_variable)
        #     print("seed block = ", self.seed_block)
        #     print("size predicted  = ", self.variables[self.selected_variable].size_block_combined)
        #     print("size actual = ", self.blocks[self.seed_block].size)
        #size_largest_block = max(self.blocks[i].size for i in self.blocks)
        if self.check_assertions:
            # print(self.selected_variable)
            assert self.blocks[self.seed_block].size == self.variables[self.selected_variable].size_block_combined
        #assert size_largest_block <= self.sorting_structure.current_key
  
    def set_associated_constraint_var(self, var_index):
        if len(self.variables[var_index].constr_size.keys()) == 0:
            self.variables[var_index].set_associated_constraint(None, False)
            return
        # if len(self.variables[var_index].constr_size.keys()):
        #     raise ValueError ("{} has no associated constraints".format(var_index))
        single_constraint_in_system = False 
        # if self.variables[var_index].single_constraint != None:
        #     single_constraint_in_system = self.constraint_in_system[self.variables[var_index].single_constraint]
        self.variables[var_index].set_associated_constraint(sorted(list(self.variables[var_index].constr_size.keys()),
                key=lambda id: (self.variables[var_index].constr_size[id], self.constraints[id].size_variables)
            )[0], single_constraint_in_system)

    def solve(self):
        while not self.termination_criteria():
            # print("loop 661")
            self.iteration()
            if self.terminate:
                break
        if self.print_data:
            self.sorting_structure.print_data()
            print("terminate on fractional size", self.sorting_structure.current_key, self.fraction*self.n)                
        # self.print_data_objects()
        # sum = 0 
        # for block in self.blocks:
        #     sum += self.blocks[block].var.size()
        # print("Number of variables in blocks = ", sum)
        return self.get_column_row_order_blocks()

    def termination_criteria(self):
        if self.border_size == 0:
            if self.print_data:
                print("Terminate on border size")
            # terminate if border goes to 0
            return True
        # a block created will be greater than some fraction of size
        # of the original matrix
        return self.sorting_structure.current_key > self.fraction*self.n
        # other termination criteria could be a limit on the border size
        # return self.border_size <= self.border_threshold
    
    def get_blocks(self):
        return [[[i for i in self.blocks[block].var],\
                 [i for i in self.blocks[block].constr]] for block in self.blocks]
    
    def get_column_row_order_blocks(self):
        # get the reordering of the columns and rows
        column_order = []
        row_order = []
        for block in self.blocks:
            column_order += [item for item in self.blocks[block].var]
            row_order += [item for item in self.blocks[block].constr]
        # get remaining border variables and columns
        border_vars = [i for i in range(len(self.variable_in_system)) if not self.variable_in_system[i]]
        border_constr = [i for i in range(len(self.constraint_in_system)) if not self.constraint_in_system[i]]
        
        column_order += border_vars 
        row_order += border_constr
        #column_order += self.border_vars_no_constr


        return column_order, row_order, self.get_blocks()


import numpy as np 
import scipy as sc
import matplotlib.pyplot as plt
import random
import sys


def create_matrix(seed):
    random.seed(seed)
    size_matrix = 20
    size_matrix_m = size_matrix
    size_matrix_n = size_matrix
    fill_matrix = 60 # maximum possible fill per row before adding diagonal
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


def reorder_sparse_matrix(m, n, row_order, col_order, target_matrix):
  target_matrix = sc.sparse.coo_matrix(target_matrix)
  permutation_matrix = sc.sparse.eye(m).tocoo()
  permutation_matrix.col = permutation_matrix.col[row_order]
  permuted_matrix = permutation_matrix.dot(target_matrix)
  permutation_matrix = sc.sparse.eye(n).tocoo()
  permutation_matrix.row = permutation_matrix.row[col_order]
  return permuted_matrix.dot(permutation_matrix)

def show_matrix_structure(matrix):
  plt.spy(matrix)
  plt.show()

def matrix_to_edges(matrix):
    edges = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                edges.append((i,j))
    return edges 

# # original_matrix = np.array([[0,0],[0,0]])
# # print(original_matrix)
# # show_matrix_structure(original_matrix)


# seeds = range(100)
# failed = []
# for i in seeds:
#     print("Instance ", i)
#     original_matrix = create_matrix(i)
#     test = BBBD_algo(matrix_to_edges(original_matrix), len(original_matrix), len(original_matrix[0]), 0.3)
#     # try:
#     col_order, row_order, blocks = test.solve()

#         print(max(test.blocks[i].size for i in test.blocks))
#         print([test.blocks[i].size for i in test.blocks])
# reordered_incidence_matrix = reorder_sparse_matrix(len(row_order),
#     len(col_order), row_order, col_order, original_matrix)
#     except:
#         failed.append(i)

#failed = [8]
original_matrix = create_matrix(14)
print(original_matrix)
# show_matrix_structure(original_matrix)
test = BBBD_algo(matrix_to_edges(original_matrix), len(original_matrix), len(original_matrix[0]), 1.1)
col_order, row_order, blocks = test.solve()
reordered_incidence_matrix = reorder_sparse_matrix(len(row_order),
    len(col_order), row_order, col_order, original_matrix)
print(col_order, row_order, blocks)
show_matrix_structure(reordered_incidence_matrix)
# print(failed)

# edges = [(0,1),(0,3), (1,0), (1,2), (2,2), (3,1)]
# m = 4
# n = 4

# bbbd_algo = BBBD_algo(edges, m, n, 0.5)
# bbbd_algo.iteration()
# bbbd_algo.iteration()
# bbbd_algo.iteration()
# bbbd_algo.iteration()

