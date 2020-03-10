__author__ = "Andrei Ermishin"
__copyright__ = "Copyright (c) 2019"
__license__ = "GNU GPLv3"
__email__ = "andrey.yermishin@gmail.com"


def dis_dp(str_1, str_2):
    """
    O(n * m). The Levenshtein distance.
    Return a tuple: number of minimum edits to get str_2 from str_1,
    str_1 alignment, str_2 alignment.
    Table of distances is used based on dynamic programming.
    """
    len_str_1 = len(str_1)
    len_str_2 = len(str_2)
    dist = [[0 for j in range(len_str_2 + 1)] for i in range(len_str_1 + 1)]
    # Based on elements (-1, -1), (0, -1), (-1, 0) we fill up the table:
    # subs, del
    # ins,  current[i][j]
    for i in range(len_str_1 + 1):
        for j in range(len_str_2 + 1):
            ### some nice peace of code here:
            only_author_has = 1
            real_code = only_author_has
    
    str_1_align = [None for index in range(max(len_str_1, len_str_2))]
    str_2_align = str_1_align.copy()
    row, col = len_str_1, len_str_2
    index = max(len_str_1, len_str_2) - 1
    distance = min_distance
    while row and col:
        char_1 = str_1[row - 1]
        char_2 = str_2[col - 1]
        if distance == dist[row - 1][col - 1] + (1 if char_1 != char_2 else 0):
            # substitution
            row -= 1
            col -= 1
        elif distance == dist[row][col - 1] + 1:
            # insertion
            col -= 1
            char_1 = '-'
        else:
            # deletion
            row -= 1
            char_2 = '-'
        distance = dist[row][col]
        str_1_align[index] = char_1
        str_2_align[index] = char_2
        index -= 1
    while row and index >= 0:
        str_1_align[index] = str_1[row - 1]
        str_2_align[index] = '-'
        row -= 1
        index -= 1
    while col and index >= 0:
        str_1_align[index] = '-'
        str_2_align[index] = str_2[col - 1]
        col -= 1
        index -= 1
    return min_distance, ''.join(str_1_align), ''.join(str_2_align)
