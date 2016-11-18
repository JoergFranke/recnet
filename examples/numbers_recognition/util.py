
"""
This file contains util methods for the numbers recognition example.
"""

import numpy as np


"""
Edit distance using dynamic programming
Credits: Ben Langmead
http://www.cs.jhu.edu/~langmea/resources/lecture_notes/dp_and_edit_dist.pdf
"""

def edit_distance(x, y):
    D = np.zeros((len(x)+1, len(y)+1), dtype=int)
    D[0, 1:] = range(1, len(y)+1)
    D[1:, 0] = range(1, len(x)+1)
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            delt = 1 if x[i-1] != y[j-1] else 0
            D[i, j] = min(D[i-1, j-1]+delt, D[i-1, j]+1, D[i, j-1]+1)
    return D[len(x), len(y)]
