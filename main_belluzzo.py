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



n = 20                                          # number of nodes
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




#-- DECODING PHASE ---------------------------------------------------------------------------------------------------
errati=0.0
totali=1000
t=time.time()
for ii in xrange(totali):
    epsilon= 2*k                              #we need h=(k+epsilon) over n nodes to succefully decode with high probability
    h=k+epsilon
    decoding_indices = rnd.sample(range(0, n), h)   #selecting h random nodes in the graph

    hashmap = np.zeros((n,2))         #vector nx2: pos[ID-1,0]-> "1" pkt of (ID-1) is decoded, "0" otherwise; pos[ID-1,1]->num_hashmap
    num_hashmap = 0                 #key counter: indicates the index of the next free row in decoded matrix

    decoded = np.zeros((k,payload), dtype=np.int64)   #matrix k*payload: the i-th row stores the total XOR of the decoded pkts
    isolated_storage_nodes=0        #counts the number of isolated storage nodes, useless for the decoding procedure


    #for i in xrange(h):         #filling of the variables, through method storage_info()
    i = -1
    added = 0
    condition_vector = np.zeros(3)
    round = 0
    condition_vector[0] = h

    while num_hashmap < k :
        i += 1
        if i == condition_vector[0]:
            #print 'Round :', i-condition_vector[0]+1 +round
            round += 1
            if condition_vector[1] == condition_vector[2]:
                #print '\n DECODING FAILURE'
                break
            else:
                condition_vector=[ condition_vector[0]+condition_vector[2], condition_vector[2], 0]
        try:
            degree,ID,XOR = node_list[decoding_indices[i]].storage_info()  #get the useful info
        except (IndexError):
            print 'Indice problematico ',i, len(decoding_indices)
            degree, _, _ = node_list[decoding_indices[i]].storage_info()  # get the useful info

        #print 'Degree of packet %d is %d ' % (i, degree)

        if degree == 0 :                       #if the pkt has degree=0 -> no pkts to decode
            isolated_storage_nodes += 1
        elif degree == 1 and hashmap[ID[0] - 1, 0]==0:                     #if the pkt has degree=1 -> immediately decoded
            hashmap[ID[0] - 1, 0] = 1             #pkt decoded
            hashmap[ID[0] - 1, 1] = num_hashmap
            #decoded[num_hashmap][0:payload] = XOR           #copy the payload
            decoded[num_hashmap, :] = XOR
            num_hashmap += 1                                #update num_hashmap and decoded
        else:                                 #if the pkt has degree>1 -> investigate if is possible to decode, or wait
            j = 0                             #temp variable for the scanning process
            not_decoded = 0                   #number of undecoded pkt, over the total in vector ID
            temp_ID = []                       #temp list for un-processed ID pkts
            while j < len(ID) and not_decoded < 2:                #we scan the IDs connected to the node
                if hashmap[ID[j]-1, 0] == 1:
                    for bit in xrange(payload):                                     #XOR bit per bit
                        x = hashmap[ID[j]-1,1]
                        XOR[bit] = XOR[bit]^decoded[x,bit]         #XOR(new)=XOR+decoded[the node which is connected to and has already been solved]
                    j += 1
                else:
                    not_decoded += 1
                    temp_ID.append(ID[j])
                    j += 1

            if not_decoded == 1:
                hashmap[temp_ID[0] -1, 0] = 1  # pkt decoded
                hashmap[temp_ID[0] -1, 1] = num_hashmap
                decoded[num_hashmap, :] = XOR
                num_hashmap +=1
            elif not_decoded == 2:
                decoding_indices.append(decoding_indices[i])
                condition_vector[2] += 1

                    #save the info appending it in the node_list
                    #increase h by 1
                    #Possible to append the info modified by this while cicle!
                    #   In this manner we are able to do less operation in the second time


    #print '\nDecoded packets BEFORE :\n', decoded

    #-- DEBUGGING -----------------------------------------------------------------------
    decoded2 = np.zeros((k,payload), dtype=np.int64)

    for i in xrange(len(sensors_indexes)):
        if hashmap[sensors_indexes[i],0] == 1:
            a = hashmap[sensors_indexes[i],1]
            decoded2[i,:] = decoded[a,:]

    #print '\nDecoded packets: REORDERED \n', decoded2

    aa=source_pkt - decoded2
    diff = sum(sum(source_pkt - decoded2))
    if diff !=0:
        errati +=1

    #print '\nDifferenza tra matrice di pkt generati e matrice di pacchetti decodificati', diff ,'\n',aa

    #print '\nHash table:\n', hashmap



elapsed = time.time()-t
print 'Time taken by reiterated decoding procedure', elapsed
decoding_prob = (totali-errati)/totali
failure_prob = errati/totali
print 'errorors' ,errati
print '\nThe decoding probability is ', decoding_prob
print '\nThe failure probability is ', failure_prob

