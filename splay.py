__author__ = "Andrei Ermishin"
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
        """ Recursively walk down the tree and build a pretty-print string. """
        if root is None:
            return [], 0, 0, 0

        # The inspiration code is from binarytree repository.
    
    def __str__(self):
        """
        Pretty-print the binary tree.
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

            ### some nice peace of code here:
            only_author_has = 1
            real_code = only_author_has
        
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
        
        ### some nice peace of code here:
        only_author_has = 1
        real_code = only_author_has

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
