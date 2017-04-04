import numpy as np                              # import of package numpy for mathematical tools
import random as rnd                            # import of package random for homonym tools
import matplotlib.pyplot as plt                 # import of package matplotlib.pyplot for plottools
import time as time
from Node import *
import cProfile
from RSD import *



t1 = time.time()                                 # initial timestamp

# -- PARAMETER INITIALIZATION SECTION --------------------------------------------------------------

n = 10                                      # number of nodes
k = 2                                        # number of sensors
L = 10                                           # square dimension
c0= 0.2                                         # parameter for RSD
delta = 0.05                                    # Prob['we're not able to recover the K pkts']<=delta

positions = np.zeros((n, 2))                    # matrix containing info on all node positions
node_list = []                                  # list of references to node objects
dmax = 5                                        # maximum distance for communication
dmax2 = dmax * dmax                             # square of maximum distance for communication
sensors_indexes = rnd.sample(range(0, n), k)    # generation of random indices for sensors

# -- DEGREE INITIALIZATION --

d = Robust_Soliton_Distribution(n, k, c0, delta) #See RSD doc
to_be_encoded = np.sum(d)                        #use to check how many pkts we should encode ***OPEN PROBLEM***

# -- NETWORK INITIALIZATION --
# Generation of storage nodes
for i in xrange(n):                             # for on 0 to n indices
    x = rnd.uniform(0.0, L)                     # generation of random coordinate x
    y = rnd.uniform(0.0, L)                     # generation of random coordinate y
    node_list.append(Storage(i + 1, x, y, d[i], n, k))      # creation of Storage node
    positions[i, :] = [x, y]

# Generation of sensor nodes
for i in sensors_indexes:                       # for on sensors position indices
    x = rnd.uniform(0.0, L)                     # generation of random coordinate x
    y = rnd.uniform(0.0, L)                     # generation of random coordinate y
    node_list[i] = Sensor(i + 1, x, y, d[i], n, k)    # creation of sensor node, function Sensor(), extend Storage class
    positions[i, :] = [x, y]                    # support variable for positions info, used for comp. optim. reasons

t = time.time()
# Find nearest neighbours using euclidean distance
nearest_neighbor = []                           #simplifying assumption, if no neighbors exist withing the range
# we consider the nearest neighbor
nn_distance = 2*L*L                             # maximum distance square equal the diagonal of the square [L,L]
for i in xrange(n):                             # cycle on all nodes
    checker = False                             # boolean variable used to check if neighbors are found (false if not)
    for j in xrange(n):                         # compare each node with all the others
        x = positions[i, 0] - positions[j, 0]   # compute x distance between node i and node j
        y = positions[i, 1] - positions[j, 1]   # compute y distance between node i and node j
        dist2 = x * x + y * y                   # compute distance square, avoid comp. of sqrt for comp. optim. reasons
        if dist2 <= dmax2:                      # check if distance square is less or equal the max coverage dist
            if dist2 != 0:                      # avoid considering self node as neighbor
                node_list[i].neighbor_write(node_list[j])   # append operation on node's neighbor list
                checker = True                              # at least one neighbor has been founded
        if not checker and dist2 <= nn_distance and dist2 != 0: # in order to be sure that the graph is connected
            # we determine the nearest neighbor
            # even if its distance is greater than the max distance
            nn_distance = dist2                 # if distance of new NN is less than distance of previous NN, update it
            nearest_neighbor = node_list[i]     # save NN reference, to use only if no neighbors are found

    if not checker:                             # if no neighbors are found withing max dist, use NN
        print 'Node %d has no neighbors within the range, the nearest neighbor is chosen.' % i
        node_list[i].neighbor_write(nearest_neighbor)   # Connect node with NN

elapsed = time.time() - t
print 'Tempo di determinazione dei vicini:', elapsed

# -- PKT GENERATION AND DISSEMINATION -----------------------------------------------------------
[node_list[sensors_indexes[i]].pkt_gen() for i in xrange(k)]        #generate data pkt, only sensor nodes are allowed

#USE storage_info() here to get the source pkts

# print [node_list[sensors_indexes[i]].storage_info() for i in xrange(k)]

j = 0
while j < k:
    for i in xrange(n):
        if node_list[i].dim_buffer != 0:
            j += node_list[i].send_pkt(0)
        if j == k:
            break

#-- DECODING PHASE -----------------------------------------------------------------------
#-- Decoding parameters extraction --

payload=10

epsilon=2          #we need h=(k+epsilon) over n nodes to succefully decode with high probability
h=k+epsilon

decoding_indices=rnd.sample(range(0, n), h)   #selecting h random nodes in the graph

degrees= []             #list of integers
ID_list=[]        #empty list of XORed pkt IDs
XOR_list=[]     #empty list of XOR payload stored

hashmap=np.zeros((n,2))  #vector nx2 that maps IDs to integer keys for the matrix of the coded pkts
num_hashmap=0            #key counter

decodificati=np.zeros((k,payload))

for i in xrange(h):     #filling of the variables, through method storage_info()
    degree,ID,XOR=node_list[decoding_indices[i]].storage_info()  #get the useful info
    a=degree            #temp variables
    b=ID
    c=XOR
    if degree<=1:       #se il pkt ha grado 0 non faremo niente
        if degree==1:   #se il pkt ha grado 1, risolviamo il pkt
            decodificati(hashmap[ID-1,1])
    else:



# Up to now, we have:
# - Integer vector of the degrees -> MP works looking for degree 1, etc...
# - List of ID -> when we simplify the system through MP we will keep track of the IDs involved
# - XOR bits

print degrees
print ID_list
print XOR_list

#SOLVE THE PROBLEM: we are still working with list.. note that the "+" between elements of the list do concatenation
#but we want the "XOR" operation... we have to transform the lists into integers,

#POSSIBLE SOLUTIONS FOR THE PROBLEM list -> integers
#1) results = [int(i) for i in results] apply this to the needed vector
#2) use function map: results = map(int, results)