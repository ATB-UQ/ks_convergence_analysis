import numpy as np

def value_to_closest_index(x, x1):
    '''Return the index of x which is closest to the value x1'''
    if len(x) < 2 or x1 < (x[1] - x[0]):
        return 0
    # return the index of the first value that is >= x1
    return np.argmax(x>=x1)