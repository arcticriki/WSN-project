import multiprocessing
import time as time
import numpy as np
from Node import *

n=5
node_list = np.empty(5, dtype=object)
print type(node_list)
node_list[0] = Storage(1, 1, 1, 1, 1, 1)

node_list2 = []
print type(node_list2)