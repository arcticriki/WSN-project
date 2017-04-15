import numpy as np                              # import of package numpy for mathematical tools
import random as rnd                            # import of package random for homonym tools
import matplotlib.pyplot as plt                 # import of package matplotlib.pyplot for plottools
import time as time
from Node import *
import cProfile
from RSD import *
from math import factorial




#t1 = time.time()                                 # initial timestamp

# -- PARAMETER INITIALIZATION SECTION --------------------------------------------------------------


payload=10

n = 100                                          # number of nodes
k = 20                                           # number of sensors
L = 5                                          # square dimension
c0= 0.1                                         # parameter for RSD
delta = 0.5                                    # Prob['we're not able to recover the K pkts']<=delta

positions = np.zeros((n, 2))                    # matrix containing info on all node positions
node_list = []                                  # list of references to node objects
dmax = 5                                        # maximum distance for communication
dmax2 = dmax * dmax                             # square of maximum distance for communication
sensors_indexes = rnd.sample(range(0, n), k)    # generation of random indices for sensors

# -- DEGREE INITIALIZATION --

d ,pdf = Robust_Soliton_Distribution(n, k, c0, delta) #See RSD doc
#to_be_encoded = np.sum(d)                        #use to check how many pkts we should encode ***OPEN PROBLEM***

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

#t = time.time()
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

#elapsed = time.time() - t
#print 'Tempo di determinazione dei vicini:', elapsed

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

# -- DECODING PHASE ---------------------------------------------------------------------------------------------------
# -- Inizialization -------------------------

decoding_ratio = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5]

Ps=np.zeros(len(decoding_ratio))

for xx in xrange(len(decoding_ratio)):

    epsilon=decoding_ratio[xx] * k - k                      #we need h=(k+epsilon) over n nodes...
    h=k+epsilon                                             #...to succefully decode with high probability

    decoding_indices = rnd.sample(range(0, n), h)  #selecting h random nodes in the graph

    degrees=[0]*h               #initialize vectors: degrees, e.g : (1,2,1,3...)
    IDs=[0]*h                   #IDS, e.g: ([57],[57,12],[12],[57,4,3]...)
    XORs=[0]*h                  #XORs, e.g: ((010...0), (110..1) ..)

    M = factorial(n)/(10*factorial(h)*factorial(n-h))       #see pag.179, paper [2]
    M=10

    number_decoding_err=np.zeros(M)

    for ii in xrange(M):

        for node in range(h):
            degree, ID, XOR = node_list[decoding_indices[node]].storage_info()

            degrees[node]=degree    #fill the 3 vectors in order to start the alg
            IDs[node]=ID
            XORs[node]=XOR


        # print 'INIZIO'
        # print 'Degrees vector', degrees, len(degrees)
        # print 'IDs vector', IDs, len(IDs)
        # print 'XORs vector', XORs, len(XORs)

        #-- MESSAGE PASSING ALGORITHM ------------------------------------------------------------------------------------

        ripple_payload=[]          #initialize the ripple: we distinguish the "payload ripple"
        ripple_IDs=[]              # and the "IDs ripple". The indexes are parallel

        hashmap = np.zeros((n,2))  #vector nx2: pos[ID-1,0]-> "1" pkt of (ID-1) is decoded, "0" otherwise; pos[ID-1,1]->num_hashmap
        num_hashmap = 0            #key counter: indicates the index of the next free row in decoded matrix

        decoded = np.zeros((k,payload), dtype=np.int64)  #matrix k*payload: the i-th row stores the total XOR of the decoded pkts
        empty_ripple = False       #boolean variable that stops the MP

        while(empty_ripple == False):

            empty_ripple = True
            #STEP 1: looking on the right of the bipartite graph
            position=0                                                      #linear search of degree one nodes
            while position < len(degrees):
                if degrees[position] == 1:                                  #if degree 1 is found...
                    if hashmap[IDs[position][0]-1,0] == 0:                  #...if it is the 1st time we find it...
                        decoded[num_hashmap,:] = XORs[position]             #..store it in the decoded matrix...
                        hashmap[IDs[position][0]-1, 0] = 1                  #...update the hasmap saying that we have found it...
                        hashmap[IDs[position][0]-1, 1] = num_hashmap        #... and update hashmap and num_hasmap
                        num_hashmap += 1

                    empty_ripple = False                                    #update boolean var
                    del degrees[position]                                   #delete the branch on the bipartite graph (degree)
                    ripple_IDs.append(IDs[position])                        #update ID ripple
                    del IDs[position]                                       #delete the branch on the bipartite graph (ID)
                    ripple_payload.append(XORs[position])                   ##update XOR ripple
                    del XORs[position]                                      #delete the branch on the bipartite graph (XORs)
                else:
                    position= position + 1

            # print
            # print 'DOPO AVER TROVATO I NODI DI DEGREE = 1'
            # print 'Degrees vector after first step', degrees      #check what happened
            # print 'IDs vector after first step', IDs
            # print 'XORs vector after first step', XORs

            # print
            # print 'ID ripple status', ripple_IDs
            # print 'Payload ripple status', ripple_payload

            #STEP 2: scanning the ripple and solving the symbols
            for each_element in ripple_IDs:                                     #take each element in the ripple (IDs)...
                for each_node in IDs:                                           #...and look for every node in the IDs vector..
                    u = 0                                                       #(in the ID vector we have lists of IDs, thus we
                    while u < len(each_node):                                   #linearly scan every list)
                        if each_element[0] == each_node[u]:                     #...if an element is found, simplify it:
                            indice_ID = IDs.index(each_node)
                            degrees[indice_ID] -= 1                             #decrease its degree
                            indice_ripple = ripple_IDs.index(each_element)
                            XORs[indice_ID] = XORs[indice_ID] ^ ripple_payload[indice_ripple]  #do the XOR
                            temp =  each_node                                   #update the list
                            del temp[u]
                            IDs[indice_ID] = temp
                            each_node = temp

                        else:
                            u += 1

            i=0
            while i<len(IDs):                                                   #update the vector od XORs
                if degrees[i]==0:
                    IDs.remove([])
                    del XORs[i]
                    degrees.remove(0)                                           #remove released nodes
                else:
                    i += 1


            #STEP 3: reset the ripple and continue the algorithm
            ripple_IDs=[]
            ripple_payload=[]

        #----------------------------------------------------------------------------------------------------------------------
        #---- Validation phase ----------------------------------

        # print 'Decodificati',len(decoded),'\n' ,decoded
        # print 'AGGIORNATO'
        # print 'Degrees vector', degrees
        # print 'IDs vector', IDs
        # print 'XORs vector', XORs

        sort_decoded = np.zeros((k, payload), dtype=np.int64)       #in order to evaluate the alg, we sort the decoded matrix

        for i in xrange(len(sensors_indexes)):
            if hashmap[sensors_indexes[i],0] == 1:
                a = hashmap[sensors_indexes[i],1]
                sort_decoded[i, :] = decoded[a, :]

        #print '\nDecoded packets: REORDERED \n', decoded2
        decoding_error=0                                            #decoding_error: "0" if everything was OK, "1" if not
        aa= source_pkt - sort_decoded
        diff = sum(sum(source_pkt - sort_decoded))
        if diff !=0:
            decoding_error +=1

        number_decoding_err[ii] = decoding_error

    Ps[xx] = 1 - (sum(number_decoding_err)/len(decoding_error))

print Ps