__author__ = "Andrei Ermishin"
__copyright__ = "Copyright (c) 2019"
__license__ = "GNU GPLv3"
__email__ = "andrey.yermishin@gmail.com"


def dp(capacity, weights):
    """
    O(n * 2**log(capacity)).
    Return a tuple: maximum weight that one can put into volume,
    list of weight indicies.
    Table of maximum weights is used based on dynamic programming.
    """
    num_items = len(weights)
    mass = [[0 for _w in range(capacity + 1)] for _idx in range(num_items + 1)]
    for i in range(1, num_items + 1):
        for cur_weight in range(1, capacity + 1):
            mass[i][cur_weight] = mass[i - 1][cur_weight]
            if weights[i - 1] <= cur_weight:
                mass[i][cur_weight] = max(mass[i][cur_weight],
                                          mass[i-1][cur_weight-weights[i-1]] + weights[i-1])
    indicies = []
    index = num_items
    cur_weight = capacity
    while index > 0:
        if mass[index][cur_weight] != mass[index-1][cur_weight]:
            # Item with index-1 was picked
            indicies.insert(0, index - 1)
            cur_weight -= weights[index - 1]
        index -= 1
    return mass[num_items][capacity], indicies
