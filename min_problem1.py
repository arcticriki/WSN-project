from Node import *
from RSD import *
import numpy as np
import csv

C1 = 2
payload = 10
n = 2000                                # number of nodes
k = 1000                                  # number of sensors
L = 10                                     # square dimension
c0 = 0.2                             # parameter for RSD
delta = 0.05                              # Prob['we're not able to recover the K pkts']<=delta

positions = np.zeros((n, 2))             # matrix containing info on all node positions
node_list = []                           # list of references to node objects
dmax = 5                                 # maximum distance for communication
dmax2 = dmax * dmax                      # square of maximum distance for communication
sensors_indexes = rnd.sample(range(0, n), k)  # generation of random indices for sensors

# -- DEGREE INITIALIZATION --

d, pdf, R = Robust_Soliton_Distribution(n, k, c0, delta)  # See RSD doc
to_be_encoded = np.sum(d)                              # use to check how many pkts we should encode ***OPEN PROBLEM***

# -- NETWORK INITIALIZATION --
# -- Generation of storage nodes --
for i in xrange(n):                     # for on 0 to n indices
    x = rnd.uniform(0.0, L)             # generation of random coordinate x
    y = rnd.uniform(0.0, L)             # generation of random coordinate y
    node_list.append(Storage(i + 1, x, y, d[i], n, k,C1))  # creation of Storage node
    positions[i, :] = [x, y]

# -- Generation of sensor nodes --
for i in sensors_indexes:               # for on sensors position indices
    x = rnd.uniform(0.0, L)             # generation of random coordinate x
    y = rnd.uniform(0.0, L)             # generation of random coordinate y
    node_list[i] = Sensor(i + 1, x, y, d[i], n,k, C1)  # creation of sensor node, function Sensor(), extend Storage class
    positions[i, :] = [x, y]            # support variable for positions info, used for comp. optim. reasons


# -- Find nearest neighbours using euclidean distance --
nearest_neighbor = []                   # simplifying assumption, if no neighbors exist withing the range
# we consider the nearest neighbor
nn_distance = 2 * L * L                 # maximum distance square equal the diagonal of the square [L,L]
for i in xrange(n):                     # cycle on all nodes
    checker = False                     # boolean variable used to check if neighbors are found (false if not)
    for j in xrange(n):                 # compare each node with all the others
        x = positions[i, 0] - positions[j, 0]  # compute x distance between node i and node j
        y = positions[i, 1] - positions[j, 1]  # compute y distance between node i and node j
        dist2 = x * x + y * y           # compute distance square, avoid comp. of sqrt for comp. optim. reasons
        if dist2 <= dmax2:              # check if distance square is less or equal the max coverage dist
            if dist2 != 0:              # avoid considering self node as neighbor
                node_list[i].neighbor_write(node_list[j])  # append operation on node's neighbor list
                checker = True          # at least one neighbor has been founded
        if not checker and dist2 <= nn_distance and dist2 != 0:  # in order to be sure that the graph is connected
            # we determine the nearest neighbor
            # even if its distance is greater than the max distance
            nn_distance = dist2         # if distance of new NN is less than distance of previous NN, update it
            nearest_neighbor = node_list[i]  # save NN reference, to use only if no neighbors are found

    if not checker:  # if no neighbors are found withing max dist, use NN
        print 'Node %d has no neighbors within the range, the nearest neighbor is chosen.' % i
        node_list[i].neighbor_write(nearest_neighbor)  # Connect node with NN

# ----------------------------------------------------------------------------------------
# OPTIMIZATION PROBLEM
# We need to minimize the following cost function : sum from i to K { x_d * d * mu(d) }, where:
#   - x_d : redundancy coefficient for node i with degree d
#   - d : degree of node i
#   - mu(d) : fraction of nodes in the network with code degree = d -> equals to RSD pdf

mu_d = pdf

print 'mu:',(mu_d)
print 'R:',R
# mu_d can be computed empirically, uncommenting the following lines --> it holds asymptotically
#_, counts = np.unique(d, return_counts=True)
#empiric_mu = [float(i)/n for i in counts]

#for MATLAB usage
with open('Dati/mu', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(mu_d)

# with open('R', 'wb') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(R)






