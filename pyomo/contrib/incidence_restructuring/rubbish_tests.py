nested_list = [[['a', 'b'], ['c','d']], [['e','f'], ['g','h']]]
print([elem for array in nested_list for elem in array[1]])


# import pyomo.environ as pyo
# m = pyo.ConcreteModel()
# m.x1 = pyo.Var(name="x1", initialize=10)
# for i in range(2, 3):
#     setattr(m, f"x{i}", pyo.Var(name=f"x{i}", initialize=10))
#     getattr(m, f"x{i}").value = 10
# m.cons1 = pyo.Constraint(expr=m.x1 + m.x2 == 4)

from pyomo.environ import *
# create the model
model = ConcreteModel()
model.x = Var(bounds=(-5, 5), initialize=1.0)
model.y = Var(bounds=(0, 1), initialize=1.0)
model.obj = Objective(expr=1e8*model.x + 1e6*model.y)
model.con = Constraint(expr=model.x + model.y == 1.0)
# create the scaling factors
model.scaling_factor = Suffix(direction=Suffix.EXPORT)
model.scaling_factor[model.obj] = 1e-6 # scale the objective
model.scaling_factor[model.con] = 2.0  # scale the constraint
model.scaling_factor[model.x] = 0.2    # scale the x variable
# transform the model
scaled_model = TransformationFactory('core.scale_model').create_using(model)
# print the value of the objective function to show scaling has occurred
model.pprint()
scaled_model.pprint()
