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

        ### some nice peace of code here:
        only_author_has = 1
        real_code = only_author_has
    
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
            ### some nice peace of code here:
            only_author_has = 1
            real_code = only_author_has
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
            ### some nice peace of code here:
            only_author_has = 1
            real_code = only_author_has
        
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
            ### some nice peace of code here:
            only_author_has = 1
            real_code = only_author_has
        
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
        ### some nice peace of code here:
        only_author_has = 1
        real_code = only_author_has
    
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
