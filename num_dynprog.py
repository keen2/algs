__author__ = "Andrei Ermishin"
__copyright__ = "Copyright (c) 2020"
__license__ = "GNU GPLv3"
__email__ = "andrey.yermishin@gmail.com"


def choosing(lst):
    """
    Choose an item on either end of the list. Return the score 
    as a tuple if each of two choose optimally for max.
    """
    len_lst = len(lst)
    matrix = [[0 for dummy_col in range(len_lst)] for dummy_row in range(len_lst)]

    for offset in range(len_lst):
        j = offset
        for i in range(len_lst - offset):
            choice1_1 = matrix[i + 2][j] if i + 2 <= j else 0
            choice1_2 = matrix[i + 1][j - 1] if i + 1 <= j - 1 else 0
            case_1 = lst[i] + min(choice1_1, choice1_2)
            choice1_3 = matrix[i][j - 2] if i <= j - 2 else 0
            case_2 = lst[j] + min(choice1_2, choice1_3)

            matrix[i][j] = max(case_1, case_2)

            j += 1
    score1 = matrix[0][len_lst - 1]
    score2 = min(matrix[1][len_lst - 1], matrix[0][len_lst - 2])
    
    return (score1, score2)


print(choosing([8, 15, 3, 7]))
