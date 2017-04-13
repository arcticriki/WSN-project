import numpy as np                              # import of package numpy for mathematical tools
import random as rnd                            # import of package random for homonym tools
import matplotlib.pyplot as plt                 # import of package matplotlib.pyplot for plottools
import time as time
from Node import *
import cProfile
from RSD import *



t1 = time.time()                                 # initial timestamp

# -- PARAMETER INITIALIZATION SECTION --------------------------------------------------------------


payload=10

n = 10                                          # number of nodes
k = 4                                           # number of sensors
L = 10                                          # square dimension
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

# -- PKT GENERATION  --------------------------------------------------------------------------------------------------
source_pkt = np.zeros((k,payload), dtype=np.int64)
for i in xrange(k):
    source_pkt[i, :] = node_list[sensors_indexes[i]].pkt_gen()

#print '\nPacchetti generati \n', source_pkt


# -- PKT  DISSEMINATION -----------------------------------------------------------------------------------------------
j = 0
while j < k:
    for i in xrange(n):
        if node_list[i].dim_buffer != 0:
            j += node_list[i].send_pkt(0)
        if j == k:
            break
tot = 0
for i in xrange(n):
    tot += node_list[i].num_encoded
print '\nNumero di pacchetti codificati:', tot, 'su un totale di:', to_be_encoded, '\n'

# -- DECODING PHASE ---------------------------------------------------------------------------------------------------
# -- Initialization -------------------------

epsilon=2                                       #we need h=(k+epsilon) over n nodes to succefully decode with high probability
h=k+epsilon
decoding_indices = rnd.sample(range(0, n), h)   #selecting h random nodes in the graph

degrees=[0]*h
IDs=[0]*h
XORs=[0]*h

for node in range(h):
    degree, ID, XOR = node_list[decoding_indices[node]].storage_info()
    degrees[node]=degree
    IDs[node]=ID
    XORs[node]=XOR

print 'INIZIO'
print 'Degrees vector', degrees, len(degrees)
print 'IDs vector', IDs, len(IDs)
print 'XORs vector', XORs, len(XORs)

#-- MP. Naive approach --------------------------------

ripple_payload=[]  #auxialiary vectors
ripple_IDs=[]

empty_ripple = False

while(empty_ripple == False):

    empty_ripple = True

    position=0                                          #linear search of degree one nodes
    while position < h:
        if degrees[position] == 1:                      #if degree 1 is found
            empty_ripple = False
            degrees[position] -= 1                       #decrease degree
            ripple_IDs.append(IDs[position])            #update ripples
            ripple_payload.append((XORs[position]))
            del XORs[position]                          #update vector XORs
        position= position + 1

    [IDs.remove(item) for item in ripple_IDs]            #remove from the ID list the released nodes
    degrees = [x for x in degrees if x != 0]             #remove from the degree list the released nodes


    print
    print 'DOPO AVER TROVATO I NODI DI DEGREE = 1'
    print 'Degrees vector after first step', degrees      #check what happened
    print 'IDs vector after first step', IDs
    print 'XORs vector after first step', XORs

    print
    print 'ID ripple status', ripple_IDs
    print 'Payload ripple status', ripple_payload

    #scanning the ripple
    for each_element in ripple_IDs:                 #prendi ogni elemento del ripple...
        for each_node in IDs:                       #...e ogni elemento del vettore degli ID...
            for u in xrange(len(each_node)):        #...e scorri per vedere each_element[0] e presente oppure no
                if each_element[0]==each_node[u]:
                    indice_ID = IDs.index(each_node)
                    degrees[indice_ID] -= 1
                    indice_ripple = ripple_IDs.index(each_element)
                    XORs[indice_ID] = XORs[indice_ID] ^ ripple_payload[indice_ripple]
    #
    for element in ripple_IDs:                            #aggiornamento delle variabili
        for vector in IDs:                               #questo ciclo serve per passare da [1,4,5] a [1,5] qualora l'elemento
            for z in xrange(len(vector)):                # [4] fosse nel ripple. Va fatto per forza fuori dallo scan
                if element[0] == vector[z-1]:
                    vector.remove(element[0])

    IDs = [x for x in IDs if x != []]               #rimuove la eventualita di liste vuote

    ripple_IDs=[]                                    #riazzera il ripple
    ripple_payload=[]

print
print 'AGGIORNATO'
print 'Degrees vector', degrees
print 'IDs vector', IDs
print 'XORs vector', XORs



#CASI PATOLOGICI IDs vector=[[17], [17], [20, 17], [], [12], [16, 17]] viene mappato nel ripple come:
#ID ripple status [[17], [17], [12]]
#e quindi facciamo la XOR 2 volte...









