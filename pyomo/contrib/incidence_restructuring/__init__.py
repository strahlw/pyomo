"""
Python utilities to re-order the incidence matrix of IDAES process flowsheets to enable automatic and/or efficient initialization strategies.
"""

from pyomo.contrib.incidence_restructuring.model_interface_util import (
    get_restructured_matrix
)
# from incidence_reordering.graph_partitioning_algo import (
#     get_column_order_partitioning_algo
# )