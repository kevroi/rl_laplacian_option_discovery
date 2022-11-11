import unittest
from plot_utils import to_env_list

class TestToEnvList(unittest.TestCase):
    def test_gridenv(self):
        # test list read in when we pass in grid environment as txt file
        self.assertEqual(to_env_list('environments/gridenv.txt'),
                        [
                        '_________G',
                        '__________',
                        '__________',
                        '__________',
                        '__________',
                        '__________',
                        '__________',
                        '__________',
                        '__________',
                        'S_________',
                        ])