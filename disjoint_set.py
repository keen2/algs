class DisjointSet:
    """
    Class for disjoint sets with path compression and rank/size heuristics.
    """
    def __init__(self, lst=[]):
        """ Make singletons for each item and initialize ranks and sizes. """
        self._parents = [-1] * len(lst)
        self._rank_size = [()] * len(lst)
        rank = 0
        self._max_size = 0
        for idx, size in enumerate(lst):
            self.make_set(idx, rank, size)
            self._max_size = max(self._max_size, size)
    
    def make_set(self, item, rank, size):
        """ Create singleton from given item (root for item itself). """
        self._parents[item] = item
        self._rank_size[item] = (rank, size)
    
    def find(self, item):
        """
        Return the root element of set containing this item.
        Make every parent a point to the root (path compression).
        """
        if item != self._parents[item]:
            self._parents[item] = self.find(self._parents[item])
        return self._parents[item]
    
    def union_by_size(self, source, dest):
        """
        Make union of two sets containing source and dest.
        Add size of source to size of dest.
        """
        source = self.find(source)
        dest = self.find(dest)

        if source == dest: return

        rank, old_size = self._rank_size[dest]
        self._rank_size[dest] = (rank, old_size + self._rank_size[source][1])
        self._parents[source] = dest
        self._max_size = max(self._max_size, self._rank_size[dest][1])
    
    def union_by_rank(self, item_1, item_2):
        """ Make union of two sets containing item_1 and item_2. """
        item_1 = self.find(item_1)
        item_2 = self.find(item_2)

        if item_1 == item_2: return

        if self._rank_size[item_1] > self._rank_size[item_2]:
            self._parents[item_2] = item_1
        else:
            self._parents[item_1] = item_2
            if self._rank_size[item_1] == self._rank_size[item_2]:
                old_rank, size = self._rank_size[item_2]
                self._rank_size[item_2] = (old_rank + 1, size)
    
    def get_max_size(self):
        """ Return maximum size among all roots. """
        return self._max_size
    
    def __str__(self):
        return str(self._parents) + '\n' + str(self._rank_size)
