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

    def test_roomenv(self):
        self.assertEqual(to_env_list('environments/room.txt'),
                        [
                        '_____X____G',
                        '_____X_____',
                        '___________',
                        '_____X_____',
                        '_____X_____',
                        'X_XXXX_____',
                        '_____XXX_XX',
                        '_____X_____',
                        '_____X_____',
                        '___________',
                        'S____X_____'
                        ])

    def test_imazeenv(self):
        self.assertEqual(to_env_list('environments/imaze.txt'),
                        [
                        '_XXXXXXXXXXXXXG',
                        '_______________',
                        'SXXXXXXXXXXXXX_'
                        ])