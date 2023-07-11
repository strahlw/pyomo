from pyomo.contrib.incidence_restructuring.BBBD_algorithm import Block

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

class TestBlockDataStructure(unittest.TestCase):
    def test_example(self):
        block = Block("3", "1")
        assert True
    
    def test_example2(self):
        self.assertEqual(1, 1)