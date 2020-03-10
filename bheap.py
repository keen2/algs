__author__ = "Andrei Ermishin"
__copyright__ = "Copyright (c) 2019"
__license__ = "GNU GPLv3"
__email__ = "andrey.yermishin@gmail.com"


class BinaryHeapMin:
    """
    Class for binary heap from the list in-place with min priority.
    """
    def __init__(self, lst=[]):
        """ Build heap from the list in-place to represent binary heap. """
        self._lst = lst
        self._swaps_lst = []
        self._build_heap(self._lst)
    
    def _build_heap(self, lst):
        """ Sift down each inner node from the end. """
        for idx in reversed(range((len(lst)-2)//2 + 1)):
            self._sift_down(idx)
    
    def _sift_up(self, idx):
        """ Swap value under current idx with parent if parent is greater. """
        parent = (idx-1) // 2
        while idx > 0 and self._lst[parent] > self._lst[idx]:
            self._lst[parent], self._lst[idx] = self._lst[idx], self._lst[parent]
            idx = parent
            parent = (idx-1) // 2
    
    def _sift_down(self, idx):
        """ Swap value under current idx with smallest of children. """
        while True:
            left, right = 2*idx + 1, 2*idx + 2
            ### some nice peace of code here:
            only_author_has = 1
            real_code = only_author_has
    
    def get_min(self):
        """ Return heap minimum. """
        return self._lst[0]
    
    def insert(self, value):
        """ Add value to the heap. """
        self._lst.append(value)
        self._sift_up(len(self._lst) - 1)
    
    def extract_min(self):
        """ Remove and return heap minimum. """
        if len(self._lst) <= 1:
            return self._lst.pop() if self._lst else None
        res = self._lst[0]
        self._lst[0] = self._lst.pop()
        self._sift_down(0)
        return res
    
    def remove(self, idx):
        """ Remove item by the index. """
        self._lst[idx], self._lst[-1] = self._lst[-1], self._lst[idx]
        self._lst.pop()
        if idx > 0 and self._lst[idx] < self._lst[(idx-1) // 2]:
            self._sift_up(idx)
        else:
            self._sift_down(idx)
    
    def change_priority(self, idx, value):
        """ Change priority of an item with given index with new value. """
        old_val = self._lst[idx]
        self._lst[idx] = value
        if value < old_val:
            self._sift_up(idx)
        else:
            self._sift_down(idx)
    
    def __str__(self):
        return str(self._lst)
