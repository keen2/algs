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
		The inspiration of code is from binarytree repository.
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


class AVLtree:
    """ Self-balancing binary search tree with O(log(n)) for operations. """
    
    def from_keys(self, keys):
        """ Return AVL tree (root Node) obtained from sequence of keys. """
        root = None
        if keys:
            for key in keys:
                root = self.insert(key, root)
        return root
    
    def from_data(self, data_lst=None):
        """ Return AVL tree (root Node) obtained from a string data_lst. """
        root = None
        count = 0
        if data_lst:
            for char in data_lst:
                root = self.insert(count, root, data=char)
                count += 1
        return root
    
    def get_root(self, node):
        """ Return root of the AVL tree as a node. """
        while node and node.parent is not None:
            node = node.parent
        return node
    
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
    
    def _update_height_size_sum(self, node):
        """ Update height, size, sum starting from node to the root. """
        while node is not None:
            left_height = node.left.height if node.left else -1
            right_height = node.right.height if node.right else -1
            node.height = max(left_height, right_height) + 1
            left_size = node.left.size if node.left else 0
            right_size = node.right.size if node.right else 0
            node.size = left_size + right_size + 1
            left_sum = node.left.sum if node.left else 0
            right_sum = node.right.sum if node.right else 0
            node.sum = left_sum + node.key + right_sum
            node = node.parent
    
    def _change_link_from_parent(self, node, to_node):
        """ Change parent's reference to child from node to to_node. """
        if node.parent:
            if node.parent.left == node:
                node.parent.left = to_node
            else:
                node.parent.right = to_node
    
    def _single_rotation(self, node, to_side='left'):
        """
        Rotate node towards to_side so that 
        it becomes child of right/left node's child.
        """
        node_side = node.right if to_side == 'left' else node.left
        opp_side = node_side.left if to_side == 'left' else node_side.right
        self._change_link_from_parent(node, node_side)
        node_side.parent = node.parent
        node.parent = node_side
        if opp_side:
            opp_side.parent = node
        if to_side == 'left':
            node.right = node.right.left
            node.parent.left = node
        else:
            node.left = node.left.right
            node.parent.right = node
        self._update_height_size_sum(node)
    
    def _rebalance(self, node):
        """
        Return node or new balanced node as root of this subtree.
        Balance AVL tree starting from node with height difference 
        between two child subtrees to the root until difference = 0.
        """
        if node is None:
            return
        
        left_height = node.left.height if node.left else -1
        right_height = node.right.height if node.right else -1
        if abs(right_height - left_height) == 1:
            self._rebalance(node.parent)
        elif abs(right_height - left_height) == 2:
            if left_height < right_height:
                # do left rotation:
                rl_height = node.right.left.height if node.right.left else -1
                rr_height = node.right.right.height if node.right.right else -1
                if rr_height >= rl_height:
                    # single left rotation:
                    self._single_rotation(node, 'left')
                else:
                    # double rotation (right-left):
                    self._single_rotation(node.right, 'right')
                    self._single_rotation(node, 'left')
            else:
                # do right rotation:
                ll_height = node.left.left.height if node.left.left else -1
                lr_height = node.left.right.height if node.left.right else -1
                if ll_height >= lr_height:
                    # single right rotation:
                    self._single_rotation(node, 'right')
                else:
                    # double rotation (left-right):
                    self._single_rotation(node.left, 'left')
                    self._single_rotation(node, 'right')
            return node.parent
        return node
    
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
    
    def find(self, key, node):
        """ Return iteratively Node with given key or None. """
        while node is not None and key != node.key:
            if key < node.key:
                node = node.left
            else:
                node = node.right
        return node
    
    def insert(self, key, node, parent=None, data=''):
        """ Add node to AVL tree by key, balance tree and return root. """
        if node is None:
            node = Node(key, parent=parent, data=data)
            if parent:
                if key < parent.key:
                    parent.left = node
                else:
                    parent.right = node
                self._update_height_size_sum(parent)
                self._rebalance(parent)
            return self.get_root(node)
        elif key == node.key:
            return self.get_root(node)
        elif key < node.key:
            return self.insert(key, node.left, node, data)
        elif key > node.key:
            return self.insert(key, node.right, node, data)
    
    def delete(self, key, root, node=None):
        """
        Remove node from AVL tree by key, balance tree and return root.
        Or remove node from AVL tree if it is given by 'node' argument.
        """
        parent = None
        node = self.find(key, root) if node is None else node
        if node is not None:
            left_max = None
            # node has no children - remove it:
            if not node.left and not node.right:
                self._change_link_from_parent(node, None)
                parent = node.parent
            # two children - change node with maximum of left subtree:
            elif node.left and node.right:
                left_max = self.get_max(node.left)
                node.key = left_max.key
                node.data = left_max.data
                # left_max has only left child or None
                self._change_link_from_parent(left_max, left_max.left)
                if left_max.left:
                    left_max.left.parent = left_max.parent
                parent = left_max.parent if left_max.parent!=node else node
            # one child - replace node with its child:
            else:
                child_node = node.left if node.left else node.right
                child_node.parent = node.parent
                self._change_link_from_parent(node, child_node)
                parent = child_node.parent if node.parent else child_node
            
            if not left_max:
                # Clear node's connections:
                node.parent = None
                node.left = None
                node.right = None
            # Update changed AVL tree:
            self._update_height_size_sum(parent)
            self._rebalance(parent)
        
        return self.get_root(parent) if node else root

    def _merge_with_root(self, node_1, node_2, root):
        """
        Return a merger of node_1 subtree and node_2 subtree with root.
        Items in node_1 < root item < items in node_2.
        """
        root.left = node_1
        root.right = node_2
        if node_1:
            node_1.parent = root
        if node_2:
            node_2.parent = root
        self._update_height_size_sum(root)
        return root
    
    def _merge(self, node_1, node_2, avl_merge=False):
        """ Return a merger of two trees: items in node_1 < node_2. """
        if node_1 is None or node_2 is None:
            return node_1 if node_1 else node_2
        
        root = self.get_max(node_1)
        node_1 = self.delete(root.key, node_1, root)
        
        if avl_merge:
            return self._avl_merge_with_root(node_1, node_2, root)
        return self._merge_with_root(node_1, node_2, root)
    
    def _avl_merge_with_root(self, node_1, node_2, root):
        """
        Return new root after merging node_1 and node_2 AVL subtrees with root.
        Recursively finds subtree with +/-1 height in right/left branch 
        and merges it with smaller (node_1/node_2).
        Items in node_1 < root item < items in node_2.
        """
        balanced_node = None
        node_1_height = node_1.height if node_1 else -1
        node_2_height = node_2.height if node_2 else -1
        if abs(node_2_height - node_1_height) <= 1:
            return self._merge_with_root(node_1, node_2, root)
        elif node_1_height > node_2_height:
            child = self._avl_merge_with_root(node_1.right, node_2, root)
            # Disconnect parent to not let _rebalance() change nodes.
            node_1.parent = None
            # Connect merged child to previous node after merging:
            child.parent = node_1
            node_1.right = child
            self._update_height_size_sum(node_1)
            balanced_node = self._rebalance(node_1)
        else:
            child = self._avl_merge_with_root(node_1, node_2.left, root)
            # Disconnect parent to not let _rebalance() change nodes.
            node_2.parent = None
            # Connect merged child to previous node after merging:
            child.parent = node_2
            node_2.left = child
            self._update_height_size_sum(node_2)
            balanced_node = self._rebalance(node_2)
        
        return balanced_node

    def avl_merge(self, node_1, node_2):
        """ Return a merger of two AVL trees: items in node_1 < node_2. """
        return self._merge(node_1, node_2, avl_merge=True)
    
    def avl_split(self, root, key):
        """
        Return two roots of subtrees splitted by the key.
        Recursively split left/right subtree and merge obtained parts.
        """
        if root is None:
            return None, None
        
        if key < root.key:
            # Disconnect child's link to parent to split left child from root.
            if root.left:
                root.left.parent = None
            # Split left subtree:
            node_1, node_2 = self.avl_split(root.left, key)
            merged_right = self._avl_merge_with_root(node_2, root.right, root)
            return node_1, merged_right
        
        # key >= root.key:
        else:
            # Disconnect child's link to parent to split right child from root.
            if root.right:
                root.right.parent = None
            # Split right subtree:
            node_1, node_2 = self.avl_split(root.right, key)
            merged_left = self._avl_merge_with_root(root.left, node_1, root)
            return merged_left, node_2
    
    def rope_split(self, root, index):
        """
        Return two roots of subtrees splitted by the index.
        Recursively split left/right subtree and merge obtained parts.
        """
        if root is None:
            return None, None
        
        left_size = root.left.size if root.left else 0
        if index < left_size + 1:
            # Disconnect child's link to parent to split left child from root.
            if root.left:
                root.left.parent = None
            # Split left subtree:
            node_1, node_2 = self.rope_split(root.left, index)
            merged_right = self._merge_with_root(node_2, root.right, root)
            return node_1, merged_right
        
        # index >= left_size + 1:
        else:
            # Disconnect child's link to parent to split right child from root.
            if root.right:
                root.right.parent = None
            # Split right subtree:
            node_1, node_2 = self.rope_split(root.right, index - left_size - 1)
            merged_left = self._merge_with_root(root.left, node_1, root)
            return merged_left, node_2
    
    def sum_from_to(self, root, left, right):
        """
        Return sum of keys of AVL tree between left and right keys.
        Return new root as 2nd element (it could change after split/merge).
        """
        less_left, equal_greater_left = self.avl_split(root, left - 1)
        left_right, greater_right = self.avl_split(equal_greater_left, right)
        sum_left_right = left_right.sum if left_right else 0
        root = self.avl_merge(less_left, left_right)
        root = self.avl_merge(root, greater_right)
        return sum_left_right, root
    
    def _get_item_by_index(self, node, index):
        """ Return node of AVL tree by given index [1...node.size]. """
        if node is None:
            return None
        
        left_size = node.left.size if node.left else 0
        if index == left_size + 1:
            return node
        elif index < left_size + 1:
            return self._get_item_by_index(node.left, index)
        else:
            return self._get_item_by_index(node.right, index - left_size - 1)
    
    def rope_reorder(self, root, index):
        """
        Place left subtree with nodes with indices smaller than index+1 
        to the right side of remaining subtree. Index of first element is 1.
        """
        if index < 1 or root is None:
            return root
        
        left, right = self.rope_split(root, index)
        return self._merge(right, left)
    
    def cut_and_paste(self, root, left, right, paste_after):
        """
        Cut subtree by nodes with indices (0,...) from left to right 
        and paste it after given index into remaining tree.
        Return new root of the resulting tree.
        """
        num_items = root.size
        less_left, root = self.rope_split(root, left)
        root = self.rope_reorder(root, right - left + 1)
        root = self._merge(less_left, root)
        less_insertion, root = self.rope_split(root, paste_after)
        after_idx = num_items - (right - left + 1) - paste_after
        root = self.rope_reorder(root, after_idx)
        return self._merge(less_insertion, root)


def sum_between_first_and_second_numbers():
    """
    Print result of executing a number of queries (add/del/find/sum).
    Data structure should do all operations in O(log(n)) time (AVL tree).
    """
    tree = AVLtree()
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
    Data structure should do all operations in O(log(n)) time (AVL tree).
    """
    tree = AVLtree()
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


def test_avl_tree():
    """ Test AVL tree for single/double rotations/splitting/merging. """
    tree = AVLtree()
    
    root = tree.from_keys(range(1, 20, 2))
    # print(root)
    #     __7________
    #    /           \
    #   3         ____15
    #  / \       /      \
    # 1   5     11       17
    #          /  \        \
    #         9    13       19
    assert root.key == tree.get_root(tree.find(19, root)).key == 7
    assert tree.find(17, tree.find(15, root)).key == 17
    assert tree.find(17, tree.find(11, root)) == None
    root = tree.insert(10, root)    # it will cause Double rotate left (r-l)
    root = tree.delete(13, root)    # it will cause Single rotate left
    assert tree.find(11, root).key == 11
    root = tree.delete(7, root)
    root = tree.insert(6, root)
    root = tree.insert(8, root)     # it will cause Double rotate right (l-r)
    root = tree.delete(19, root)
    root = tree.delete(10, root)    # it will cause Double rotate left (r-l)
    root = tree.delete(1, root)
    #     ____9___
    #    /        \
    #   5         _15
    #  / \       /   \
    # 3   6     11    17
    #      \
    #       8
    assert [n.key for n in tree.in_order_traverse(root)]==[3,5,6,8,9,11,15,17]
    assert tree._get_item_by_index(root, 1).key == 3
    assert tree._get_item_by_index(root, 6).key == 11
    assert tree._get_item_by_index(tree.find(15, root), 3).key == 17
    assert [tree.get_min(root).key, tree.get_max(tree.find(5,root)).key]==[3,8]
    
    left, right = tree.avl_split(root, 7)
    assert [n.key for n in tree.in_order_traverse(left)] == [3,5,6]
    assert [n.key for n in tree.in_order_traverse(right)] == [8,9,11,15,17]
    #   5                      9___
    #  / \                    /    \
    # 3   6                  8     _15
    #                             /   \
    #                            11    17
    l_right, r_right = tree.avl_split(right, 12)
    #   9                    15
    #  / \                     \
    # 8   11                    17
    root = tree.avl_merge(left, l_right)
    #     6__
    #    /   \
    #   5     9
    #  /     / \
    # 3     8   11
    root = tree.avl_merge(root, r_right)    # must be AVL tree with same order
    #       ____11
    #      /      \
    #     6__      15
    #    /   \       \
    #   5     9       17
    #  /     /
    # 3     8
    assert [n.key for n in tree.in_order_traverse(root)]==[3,5,6,8,9,11,15,17]
    
    keys = [15,59,54,22,99,5,19,45,17,47,26,83,28,13,40,56,62,79,58,80,48,6,87]
    root = tree.from_keys(keys)
    #                _____________45_______________
    #               /                              \
    #          ____19___                     _______59_________
    #         /         \                   /                  \
    #     ___15         _26            ____54               ____83___
    #    /     \       /   \          /      \             /         \
    #   6       17    22    28       47       56         _79         _99
    #  / \                    \        \        \       /   \       /
    # 5   13                   40       48       58    62    80    87
    left, right = tree.avl_split(root, 26)
    #          ____19___              ________________59_________
    #         /         \            /                           \
    #     ___15         _26     ____45______                  ____83___
    #    /     \       /       /            \                /         \
    #   6       17    22      28         ____54            _79         _99
    #  / \                      \       /      \          /   \       /
    # 5   13                     40    47       56       62    80    87
    #                                    \        \
    #                                     48       58
    root = tree.avl_merge(left, right)
    #                      _______45_______________
    #                     /                        \
    #          __________26                  _______59_________
    #         /            \                /                  \
    #     ___15___          28         ____54               ____83___
    #    /        \           \       /      \             /         \
    #   6         _19          40    47       56         _79         _99
    #  / \       /   \                 \        \       /   \       /
    # 5   13    17    22                48       58    62    80    87
    assert [n.key for n in tree.in_order_traverse(root)] == sorted(keys)
    
    # Split random AVL tree into two subtrees by random key and merge them:
    import random as rnd
    for _ in range(100):
        keys = rnd.sample(range(100), rnd.randrange(30, 100))
        root = tree.from_keys(keys)
        rnd_key = rnd.randrange(tree.get_min(root).key, tree.get_max(root).key)
        left, right = tree.avl_split(root, rnd_key)
        root = tree.avl_merge(left, right)
        assert [n.key for n in tree.in_order_traverse(root)] == sorted(keys)
    
    root = tree.from_keys([3, 8, 0, 6, 10, 2, 4, 1])
    assert tree.sum_from_to(root, 6, 9)[0] == 14
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
    keys = [76,72,54,69,58,97,12,52,77,2,16,9,96,8,60,36,17,46,95,37,75,23,45,
        80,6,10,15,64,31,24,94,18,32,42,55,49,7,57,1,70,83,38,50,34,66,47,84]
    root = tree.from_keys(keys)
    assert tree.sum_from_to(root, 32, 67)[0] == 922
    print('Ok\n')


if __name__ == "__main__":
    test_avl_tree()
    # sum_between_first_and_second_numbers()
    # cut_and_paste_substrings()
