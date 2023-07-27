nested_list = [[['a', 'b'], ['c','d']], [['e','f'], ['g','h']]]
print([elem for array in nested_list for elem in array[1]])