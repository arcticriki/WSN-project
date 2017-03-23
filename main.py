import numpy as np                              # import of package numpy for mathematical tools
import random as rnd                            # import of package random for homonym tools
import matplotlib.pyplot as plt                 # import of package matplotlib.pyplot for plottools
import time as time                             # import of package time for monitoring computational time
from Node import *                              # * means we import both Storage() and Sensor() classes

t = time.time()                                 # initial timestamp

# PARAMETER INITIALIZATION SECTION

n = 1000                                        # number of nodes
k = 10                                         # number of sensors
L = 25                                         # square dimension (messo piccolo xk se ho nodi scollegati crasho e va tt a puttane)

positions = np.zeros((n, 2))                    # matrix containing info on all node positions
node_list = []                                  # list of references to node objects
dmax = 3                                        # maximum distance for communication
dmax2 = dmax * dmax                             # square of maximum distance for communication
sensors_indexes = rnd.sample(range(0, n), k)   # generation of random indices for sensors

# DEGREE INITIALIZATION
d = np.ones(n)*5                                #degree provvisorio per fare prove
#degree_generator = PRNG()                     CAPIRE COME FAR FUNZIONARE STAMMERDA DI FUNZIONE GENERATRICE DI D
#d = [ for i in xrange(n)]

# NETWORK INITIALIZATION

# Generation of storage nodes
for i in xrange(n):                             # for on 0 to n indices
    x = rnd.uniform(0.0, L)                     # generation of random coordinate x
    y = rnd.uniform(0.0, L)                     # generation of random coordinate y
    node_list.append(Storage(i + 1, x, y, d[i], n, k))      # creation of storage node, function Storage()
    positions[i, :] = [x, y]

# Generation of sensor nodes
for i in sensors_indexes:                      # for on sensors position indices
    x = rnd.uniform(0.0, L)                     # generation of random coordinate x
    y = rnd.uniform(0.0, L)                     # generation of random coordinate y
    node_list[i] = Sensor(i + 1, x, y, d[i], n, k)          # creation of sensor node, function Sensor(), extend Storage class
    positions[i, :] = [x, y]                    # support variable for positions info, used for comp. optim. reasons

# Find nearest neighbours using euclidean distance
ok = True
for i in xrange(n):                             # cycle on all nodes
    checker = True
    for j in xrange(n):                         # compare each node with all the others
        x = positions[i, 0] - positions[j, 0]   # compute x distance between node i and node j
        y = positions[i, 1] - positions[j, 1]   # compute y distance between node i and node j
        dist2 = x * x + y * y                   # compute distance square, avoid comp. of sqrt for comp. optim. reasons
        if dist2 <= dmax2:                      # check on distance square
            if dist2 != 0:                      # avoid considering self node as neighbor
                node_list[i].neighbor_write(node_list[j])   # append operation on node's neighbor list
                checker = None
    if checker:
        print 'Node %d has no neighbors' % i
        ok = None

if ok:
    print 'Al nodes are connected'

elapsed = time.time() - t                       # computation of elapsed time
print elapsed



# plt.title("Graphical representation of sensors' positions")
# plt.xlabel('X')
# plt.ylabel('Y')
# #plt.grid()
# plt.xticks([5 * k for k in xrange(L / 5 + 1)])
# plt.yticks([5 * k for k in xrange(L / 5 + 1)])
# plt.axis([-1, L + 1, -1, L + 1])
# plt.plot(positions[:, 0], positions[:, 1], linestyle='', marker='o', label='Storage')
# plt.plot(positions[sensors_indexes, 0], positions[sensors_indexes, 1], color='red', linestyle='', marker='o', label='Sensor')
# plt.legend(loc='upper left')
# plt.show()

[node_list[sensors_indexes[i]].pkt_gen() for i in xrange(k)]
for i in xrange(n):
   node_list[i].send_pkt(0)