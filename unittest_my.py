import sys

import unittest


class TestSomeFunctions(unittest.TestCase):

    # test_ is for naming convention
    def test_(self):
        self.assertTrue(palindrome('madam'))
        self.assertFalse(palindrome('text'))

    @unittest.skipIf(sys.version_info < (3, 3), 'not supported by the version')
    def test_list_clear(self):
        empty_list = list(range(10))
        empty_list.clear()
        self.assertEqual(empty_list, [])

    def test_swap(self):
        s = 'hellO WoRld'
        self.assertEqual(s.swapcase(), 'HELLo wOrLD')


def palindrome(word):
    return word == word[::-1]


def run_my_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSomeFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    # unittest.main() # Ran 3 tests in 0.001s OK
    run_my_tests()
