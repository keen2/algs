__author__ = "Andrey Ermishin"
__copyright__ = "Copyright (c) 2019"
__license__ = "GNU GPLv3"
__email__ = "andrey.yermishin@gmail.com"


class Node:
    """ Node of binary tree. """
    def __init__(self, key, left=None, right=None, parent=None,
                 height=0, size=1, data=''):
        self.key = key
        self.left = left
        self.right = right
        self.parent = parent
        self.height = height
        self.size = size
        self.sum = key
        self.data = data
    
    def simple_str(self):
        return '[{}: {} {} p{} h{} s{} sum{}]'.format(self.key,
                                       self.left.key if self.left else '-',
                                       self.right.key if self.right else '-',
                                       self.parent.key if self.parent else '-',
                                       self.height, self.size, self.sum)
    
    def _build_tree_str(self, root, curr_index=0, index=False, delimiter='-'):
        """
		Recursively walk down the tree and build a pretty-print string.
		(The inspiration of code is from binarytree repository.)
		"""
        if root is None:
            return [], 0, 0, 0

        nodes = []
        links = []
        if index:
            node_repr = f'{curr_index}{delimiter}{root.key}'
        else:
            node_repr = str(root.key)

        new_root_width = gap_size = len(node_repr)

        # Get the left and right subtrees, their widths, root repr positions
        l_box, l_box_width, l_root_start, l_root_end = \
            self._build_tree_str(root.left, 2*curr_index +1, index, delimiter)
        r_box, r_box_width, r_root_start, r_root_end = \
            self._build_tree_str(root.right, 2*curr_index +2, index, delimiter)

        # Draw the branch connecting the current root node to the left subtree
        # Pad the line with whitespaces where necessary
        if l_box_width > 0:
            l_root = (l_root_start + l_root_end)//2 + 1
            nodes.append(' ' * (l_root + 1))
            nodes.append('_' * (l_box_width - l_root))
            links.append(' ' * l_root + '/')
            links.append(' ' * (l_box_width - l_root))
            new_root_start = l_box_width + 1
            gap_size += 1
        else:
            new_root_start = 0

        # Draw the representation of the current root node
        nodes.append(node_repr)
        links.append(' ' * new_root_width)

        # Draw the branch connecting the current root node to the right subtree
        # Pad the line with whitespaces where necessary
        if r_box_width > 0:
            r_root = (r_root_start + r_root_end) // 2
            nodes.append('_' * r_root)
            nodes.append(' ' * (r_box_width - r_root + 1))
            links.append(' ' * r_root + '\\')
            links.append(' ' * (r_box_width - r_root))
            gap_size += 1
        new_root_end = new_root_start + new_root_width - 1

        # Combine the left and right subtree with the branches drawn above
        gap = ' ' * gap_size
        new_box = [''.join(nodes), ''.join(links)]
        for i in range(max(len(l_box), len(r_box))):
            l_line = l_box[i] if i < len(l_box) else ' ' * l_box_width
            r_line = r_box[i] if i < len(r_box) else ' ' * r_box_width
            new_box.append(l_line + gap + r_line)

        # Return the new box, its width and its root repr positions
        return new_box, len(new_box[0]), new_root_start, new_root_end
    
    def __str__(self):
        """
        Pretty-print the binary tree.
        link = github.com/joowani/binarytree/blob/master/binarytree/__init__.py
        """
        lines = self._build_tree_str(self)[0]
        return '\n' + '\n'.join(line.rstrip() for line in lines)


class SplayTree:
    """ Self-balancing binary search tree with O(log(n)) for operations. """

    def from_keys(self, keys):
        """ Return splay tree (root Node) obtained from sequence of keys. """
        root = None
        if keys:
            for key in keys:
                root = self.insert(key, root)
        return root
    
    def from_data(self, data_lst=None):
        """ Return splay tree (root Node) obtained from a string data_lst. """
        root = None
        count = 0
        if data_lst:
            for char in data_lst:
                root = self.insert(count, root, data=char)
                count += 1
        return root

    def get_min(self, node):
        """ Return node with minimum key in given subtree. """
        while node and node.left is not None:
            node = node.left
        return node
    
    def get_max(self, node):
        """ Return node with maximum key in given subtree. """
        while node and node.right is not None:
            node = node.right
        return node
    
    def get_root(self, node):
        """ Return root of the splay tree as a node. """
        while node and node.parent is not None:
            node = node.parent
        return node
    
    def _update_size_sum(self, node):
        """ Update size and sum starting from node to the root. """
        while node is not None:
            left_size = node.left.size if node.left else 0
            right_size = node.right.size if node.right else 0
            node.size = left_size + 1 + right_size
            left_sum = node.left.sum if node.left else 0
            right_sum = node.right.sum if node.right else 0
            node.sum = left_sum + node.key + right_sum
            node = node.parent
    
    def in_order_traverse(self, node):
        """ Yield nodes of subtree given by root in in-order manner. """
        stack = []
        while stack or node is not None:
            if node is not None:
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                yield node
                node = node.right
    
    def _cut_node_from_parent(self, node):
        """
        Return side the node was connected or None if node has no parent.
        Disconnect parent from child and vice versa.
        """
        side = None
        if node.parent:
            if node.parent.left == node:
                node.parent.left = None
                side = 'left'
            else:
                node.parent.right = None
                side = 'right'
        node.parent = None
        return side

    def _zig(self, node, to_side):
        """
        Rotate the node towards to_side so that 
        it becomes parent of node's parent and return the node.
        """
        parent = node.parent
        if to_side == 'right':
            # node is left child:
            parent.left = node.right
            if node.right:
                node.right.parent = parent
            node.right = parent
        else:
            # node is right child:
            parent.right = node.left
            if node.left:
                node.left.parent = parent
            node.left = parent
        parent.parent = node
        node.parent = None
        self._update_size_sum(parent)
        return node
    
    def _zig_zig(self, node, to_side):
        """
        Rotate parent of the node, then rotate the node, 
        so it becomes new parent of subtree and return the node.
        """
        self._zig(node.parent, to_side)
        return self._zig(node, to_side)
    
    def _zig_zag(self, node, to_side):
        """
        Rotate the node in respect of parent, 
        then rotate the node in respect of grandparent, 
        so it becomes new parent of subtree and return the node.
        """
        grandparent = node.parent.parent
        self._zig(node, 'right' if to_side == 'left' else 'left')
        # Connect grandparent to node:
        if to_side == 'right':
            grandparent.left = node
        else:
            grandparent.right = node
        node.parent = grandparent
        return self._zig(node, to_side)
    
    def splay(self, node):
        """ Move node up to root's place by rotations and return root. """
        while node is not None and node.parent is not None:
            parent = node.parent
            great_grandpa = parent.parent.parent if parent.parent else None
            side = None
            if parent.parent is None:
                node = self._zig(node,
                                 'right' if node == parent.left else 'left')
                break
            
            elif (node == parent.left and parent == parent.parent.left
              or node == parent.right and parent == parent.parent.right):
                # Disconnect grandparent from great-grandparent:
                side = self._cut_node_from_parent(parent.parent)
                # Rotate node to grandparent place:
                node = self._zig_zig(node,
                                    'right' if node == parent.left else 'left')
            
            else:
                # Disconnect grandparent from great-grandparent:
                side = self._cut_node_from_parent(parent.parent)
                # Rotate node to grandparent place:
                node = self._zig_zag(node,
                                   'right' if node == parent.right else 'left')
            
            # Connect node to great-grandparent:
            if great_grandpa is not None:
                if side == 'right':
                    great_grandpa.right = node
                else:
                    great_grandpa.left = node
            node.parent = great_grandpa
        
        return node
    
    def find(self, key, node):
        """
        Return iteratively Node with given key or None.
        Do splay for last found node. So root can change after splay, 
        good practise is to use: root = self.get_root(root) after find().
        """
        parent = None
        while node is not None:
            parent = node
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                # key == node.key
                return self.splay(node)
        self.splay(parent)
        return node
    
    def insert(self, key, node, parent=None, data=''):
        """ Add node to splay tree by key, splay node and return root. """
        if node is None:
            node = Node(key, parent=parent, data=data)
            if parent:
                if key < parent.key:
                    parent.left = node
                else:
                    parent.right = node
            return self.splay(node)
        elif key == node.key:
            return self.splay(node)
        elif key < node.key:
            return self.insert(key, node.left, node, data)
        elif key > node.key:
            return self.insert(key, node.right, node, data)

    def split(self, root, key):
        """
        Return two roots of subtrees splitted by the key.
        Splay node with given key or previous node and split the tree.
        """
        if root is None:
            return None, None
        
        node = self.find(key, root)
        root = node if node is not None else self.get_root(root)
        
        if key < root.key:
            # Disconnect left child from root (parent was greater):
            node_1 = root.left
            if node_1:
                node_1.parent = None
            root.left = None
            self._update_size_sum(root)
            return node_1, root
        else:
            # Disconnect right child from root:
            node_2 = root.right
            if node_2:
                node_2.parent = None
            root.right = None
            self._update_size_sum(root)
            return root, node_2
    
    def rope_split(self, root, index):
        """
        Return two roots of subtrees splitted by the index.
        Splay node with given index and split the tree.
        """
        if root is None:
            return None, None
        
        node = self._get_item_by_index(root, index)
        if node is None:
            if index < 1:
                return None, root
            else:
                return root, None
        root = self.splay(node)

        # Disconnect right child from root:
        node_2 = root.right
        if node_2:
            node_2.parent = None
        root.right = None
        self._update_size_sum(root)
        return root, node_2

    def merge(self, node_1, node_2):
        """ Return a merger of two splay trees: items in node_1 < node_2. """
        if node_1 is None or node_2 is None:
            return node_1 if node_1 else node_2
        
        root = self.splay(self.get_max(node_1))
        root.right = node_2
        node_2.parent = root
        self._update_size_sum(root)
        return root
    
    def delete(self, key, root, node=None):
        """
        Remove node from splay tree by key and return root.
        Or remove node from splay tree if node is given.
        """
        if root is None:
            return None
        
        node = node if node is not None else self.find(key, root)
        if node is not None:
            root = self.splay(node)
            if root.left:
                root.left.parent = None
            if root.right:
                root.right.parent = None
            root = self.merge(root.left, root.right)
        
        return self.get_root(root)

    def sum_from_to(self, root, left, right):
        """
        Return sum of keys of splay tree between left and right keys.
        Return new root as 2nd element (it could change after split/merge).
        """
        less_left, equal_greater_left = self.split(root, left - 1)
        left_right, greater_right = self.split(equal_greater_left, right)
        sum_left_right = left_right.sum if left_right else 0
        root = self.merge(less_left, left_right)
        root = self.merge(root, greater_right)
        return sum_left_right, root
    
    def _get_item_by_index(self, node, index):
        """ Return node of splay tree by given index [1...node.size]. """
        left_size = node.left.size if node and node.left else 0
        while node is not None and index != left_size + 1:
            if index < left_size + 1:
                node = node.left
            else:
                node = node.right
                index -= left_size + 1
            left_size = node.left.size if node and node.left else 0
        return node
    
    def rope_reorder(self, root, index):
        """
        Place left subtree with nodes with indices smaller than index+1 
        to the right side of remaining subtree. Index of first element is 1.
        """
        if index < 1 or root is None:
            return root
        
        left, right = self.rope_split(root, index)
        return self.merge(right, left)
    
    def cut_and_paste(self, root, left, right, paste_after):
        """
        Cut subtree by nodes with indices (0,...) from left to right 
        and paste it after given index into remaining tree.
        Return new root of the resulting tree.
        """
        num_items = root.size
        less_left, root = self.rope_split(root, left)
        root = self.rope_reorder(root, right - left + 1)
        root = self.merge(less_left, root)
        less_insertion, root = self.rope_split(root, paste_after)
        after_idx = num_items - (right - left + 1) - paste_after
        root = self.rope_reorder(root, after_idx)
        return self.merge(less_insertion, root)


def sum_between_first_and_second_numbers():
    """
    Print result of executing a number of queries (add/del/find/sum).
    Data structure should do all operations in O(log(n)) time (splay tree).
    """
    tree = SplayTree()
    root = None
    funcs = {
        '+': tree.insert,
        '-': tree.delete,
        '?': tree.find,
        's': tree.sum_from_to
    }
    last_sum = 0
    calc = lambda x: (x + last_sum) % 1000000001
    for _ in range(int(input())):
        cmd, *args = input().split()
        if cmd == 's':
            left, right = map(int, args)
            _sum, root = funcs[cmd](root, calc(left), calc(right))
            print(_sum)
            last_sum = _sum
        elif cmd in funcs:
            if cmd == '?':
                node = funcs[cmd](calc(int(args[0])), root)
                root = tree.get_root(root)  # can change after splay in find()
                print('Found' if node else 'Not found')
            else:
                root = funcs[cmd](calc(int(args[0])), root)
        else:
            continue
    # Input:                            Output:
    # 5
    # + 491572259
    # ? 491572259                       Found
    # ? 899375874                       Not found
    # s 310971296 877523306             491572259
    # + 352411209


def cut_and_paste_substrings():
    """
    Print result of executing a number of queries (cut/paste substring).
    Data structure should do all operations in O(log(n)) time (splay tree).
    """
    tree = SplayTree()
    line = input()
    root = tree.from_data(line)
    for _ in range(int(input())):
        left_cut, right_cut, paste_after = map(int, input().split())
        root = tree.cut_and_paste(root, left_cut, right_cut, paste_after)
    print(''.join(n.data for n in tree.in_order_traverse(root)))
    # Input:                            Output:
    # abcdef                            efcabd
    # 2                                 (abcdef) -> cabdef -> efcabd
    # 0 1 1
    # 4 5 0


def test_splay_tree():
    """ Test splay tree for splay/find/insert/split/merge/delete. """
    tree = SplayTree()

    # it will cause zig(), zig_zag(), zig_zig()
    root = tree.from_keys([78, 11, 47, 45, 51, 43, 62, 72, 87, 46, 41, 92, 67])
    # print(root)
    #               __________67_________
    #              /                     \
    #      _______46______            ____92
    #     /               \          /
    #   _41               _62      _78
    #  /   \             /        /   \
    # 11    43         _51       72    87
    #         \       /
    #          45    47
    assert tree.get_root(root.left.right.left).key == 67    # from (51)
    root = tree.find(51, root)
    assert tree.get_root(root).key == 51
    assert tree.find(80, root) == None
    root = tree.get_root(root)  # root can change after splay in find()
    assert root.key == 87
    root = tree.delete(47, root)    # it will cause zig_zag(), zig_zig()
    assert root.key == 46
    root = tree.insert(9, root) # it will cause zig_zig() and zig()
    assert root.key == 9
    root = tree.delete(46, root)
    assert root.key == 45
    root = tree.delete(67, root)
    root = tree.delete(51, root)
    root = tree.delete(87, root)
    root = tree.delete(41, root)
    root = tree.delete(72, root)
    root = tree.delete(46, root)  # there is no 46 in the tree
    assert root.key == 45
    #     ____45
    #    /      \
    #   11       62
    #  /  \        \
    # 9    43       78
    #                 \
    #                  92
    assert [n.key for n in tree.in_order_traverse(root)]==[9,11,43,45,62,78,92]
    assert tree._get_item_by_index(root, 1).key == 9
    assert tree._get_item_by_index(root, 5).key == 62
    _node = tree.find(78, root)
    root = tree.get_root(root)  # 78 is now the root
    assert tree._get_item_by_index(root, 4).key == 45
    assert [tree.get_min(root).key, tree.get_max(root.left).key] == [9, 62]
    
    keys = [78, 11, 47, 45, 51, 43, 62, 72, 87, 46, 41, 92, 67]
    root = tree.from_keys(keys)
    assert root.key, root.sum == (67, 742)
    #               __________67_________
    #              /                     \
    #      _______46______            ____92
    #     /               \          /
    #   _41               _62      _78
    #  /   \             /        /   \
    # 11    43         _51       72    87
    #         \       /
    #          45    47
    left, right = tree.split(root, 70)
    #               __________67         72
    #              /                       \
    #      _______46______                  78___
    #     /               \                      \
    #   _41               _62                    _92
    #  /   \             /                      /
    # 11    43         _51                     87
    #         \       /
    #          45    47
    root = tree.merge(left, right)
    #               __________67
    #              /            \
    #      _______46______       72
    #     /               \        \
    #   _41               _62       78___
    #  /   \             /               \
    # 11    43         _51               _92
    #         \       /                 /
    #          45    47                87
    assert [n.key for n in tree.in_order_traverse(root)] == sorted(keys)
    
    # Split random splay tree into two subtrees by random key and merge them:
    import random as rnd
    for _ in range(100):
        keys = rnd.sample(range(100), rnd.randrange(30, 100))
        root = tree.from_keys(keys)
        rnd_key = rnd.randrange(tree.get_min(root).key, tree.get_max(root).key)
        left, right = tree.split(root, rnd_key)
        root = tree.merge(left, right)
        assert sum(n.key for n in tree.in_order_traverse(root)) == sum(keys)
        assert [n.key for n in tree.in_order_traverse(root)] == sorted(keys)
    
    root = tree.from_keys([3, 8, 0, 6, 10, 2, 4, 1])
    assert tree.sum_from_to(root, 5, 9)[0] == 14
    root = tree.from_keys([4, 1, 6, 0, 2, 5])
    assert tree.sum_from_to(root, 6, 7)[0] == 6
    root = tree.from_keys([4, 0, 8, 9])
    assert tree.sum_from_to(root, 5, 7)[0] == 0
    root = tree.from_keys([7, 4, 9, 2, 5])
    assert tree.sum_from_to(root, 5, 9)[0] == 21
    root = tree.from_keys([7, 4, 9, 2, 5])
    assert tree.sum_from_to(root, 4, 4)[0] == 4
    root = tree.from_keys([7, 4, 9, 2, 5])
    assert tree.sum_from_to(root, 6, 6)[0] == 0
    keys = [72,54,69,58,12,52,77,2,16,9,8,60,36,17,
            46,95,37,75,23,45,80,6,10,15,64,31]
    root = tree.from_keys(keys)
    assert tree.find(40, root) == None
    root = tree.get_root(root)
    #                             ____37__________________
    #                            /                        \
    #                       ____31         ________________64____________
    #                      /      \       /                              \
    #                 ____17       36    45                           ____80
    #                /      \              \                         /      \
    #           ____15       23             46_________         ____75       95
    #          /      \                                \       /      \
    #     ____10       16                        _______60    69       77
    #    /      \                               /               \
    #   6__      12                            52___             72
    #  /   \                                        \
    # 2     9                                       _58
    #      /                                       /
    #     8                                       54
    assert tree.sum_from_to(root, 11, 70)[0] == 635

    for _ in range(100):
        keys = rnd.sample(range(100), rnd.randrange(20, 100))
        root = tree.from_keys(keys)
        left = rnd.randrange(tree.get_min(root).key, tree.get_max(root).key)
        right = rnd.randrange(left, tree.get_max(root).key)
        keys_in_range = filter(lambda x: left <= x <= right, keys)
        assert tree.sum_from_to(root, left, right)[0] == sum(keys_in_range)
    
    print('Ok\n')


if __name__ == "__main__":
    test_splay_tree()
    # sum_between_first_and_second_numbers()
    # cut_and_paste_substrings()
