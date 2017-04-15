import numpy as np                              # import of package numpy for mathematical tools
import random as rnd                            # import of package random for homonym tools
import matplotlib.pyplot as plt                 # import of package matplotlib.pyplot for plottools
import time as time
from Node import *
import cProfile
from RSD import *
from math import factorial





t1 = time.time()                                 # initial timestamp

# -- PARAMETER INITIALIZATION SECTION --------------------------------------------------------------


payload=10



n = 20                                          # number of nodes
k = 4                                          # number of sensors
L = 10                                          # square dimension
c0= 0.1                                         # parameter for RSD
delta = 0.5                                     # Prob['we're not able to recover the K pkts']<=delta

positions = np.zeros((n, 2))                    # matrix containing info on all node positions
node_list = []                                  # list of references to node objects
dmax = 5                                        # maximum distance for communication
dmax2 = dmax * dmax                             # square of maximum distance for communication
sensors_indexes = rnd.sample(range(0, n), k)    # generation of random indices for sensors

# -- DEGREE INITIALIZATION --

d, pdf = Robust_Soliton_Distribution(n, k, c0, delta) #See RSD doc
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

print '\nPacchetti generati \n', source_pkt ,'\n'


# -- PKT  DISSEMINATION -----------------------------------------------------------------------------------------------
j = 0
while j < k:
    for i in xrange(n):
        if node_list[i].dim_buffer != 0:
            j += node_list[i].send_pkt(0)
        if j == k:
            break
tot = 0
distribution_post_dissemination = np.zeros(k+1)
for i in xrange(n):
    index = node_list[i].num_encoded
    distribution_post_dissemination[index] += 1.0/to_be_encoded
    tot += node_list[i].num_encoded
print '\nNumero di pacchetti codificati:', tot, 'su un totale di:', to_be_encoded, '\n'

# plt.title('Post dissemination')
# y = distribution_post_dissemination
# x = np.linspace(1, k, k, endpoint=True)
# plt.axis([0, k, 0, 0.6])
# plt.plot(x,y , label='post dissemination')
# y2 = np.zeros(k)
# y2[:len(pdf)] = pdf
# plt.plot(x, y2, color='red', label='robust soliton')
# plt.legend(loc='upper left')
# plt.grid()
# plt.show()


#-- DECODING PHASE ---------------------------------------------------------------------------------------------------
epsilon = 2 * k                         #we need h=(k+epsilon) over n nodes to succefully decode with high probability
h = k + epsilon                         # Number of node from which we retrieve the pkts
errati = 0.0                            # Number of iteration in which we do not decode
errati2 = 0.0                           # Number of iteration in which we do not decode
M = factorial(n)/(10*factorial(h)*factorial(n-h))   # Computation of the number of iterations to perform, see paper 2
t=time.time()
M=5
for ii in xrange(M):
    #-- parameters initialization phase --------------------

    decoding_indices = rnd.sample(range(0, n), h)   #selecting h random nodes in the graph
    hashmap = np.zeros((n,2))           #vector nx2: pos[ID-1,0]-> "1" pkt of (ID-1) is decoded, "0" otherwise; pos[ID-1,1]->num_hashmap
    num_hashmap = 0                     #key counter: indicates the index of the next free row in decoded matrix
    decoded = np.zeros((k,payload), dtype=np.int64)   #matrix k*payload: the i-th row stores the total XOR of the decoded pkts
    isolated_storage_nodes = 0          #counts the number of isolated storage nodes, useless for the decoding procedure
                                        # or simply the number of zero degree node
    i = 0                               # variabile accessori per il ciclo di iterazione sulla lista degli h nodi scelti tra gli n
    condition_vector = np.zeros(3)      # list of 3 indicies useful for nex cycle
    condition_vector[0] = h             # first index represent the lenght of the vector of indicies

    print '\n'

    while num_hashmap < k :             # While we have not decoded the k source pkts do:

        if i == condition_vector[0]:  # if we watched the whole vector enter this section
            if condition_vector[1] == condition_vector[2] or condition_vector[2]==0:      # if we queued the same number of pkt of previous round
                #print '\n                                                     DECODING FAILURE'
                break                                           # exit, they cannot be decoded, MP failure
            else:
                condition_vector=[ condition_vector[0]+condition_vector[2], condition_vector[2], 0.0]

        degree,ID,XOR = node_list[decoding_indices[i]].storage_info()  #get the useful info

        if degree == 0 :                       #if the pkt has degree=0 -> no pkts to decode
            isolated_storage_nodes += 1
        elif degree == 1 and hashmap[ID[0] - 1, 0]==0:                     #if the pkt has degree=1 -> immediately decoded
            hashmap[ID[0] - 1, 0] = 1             #pkt decoded
            hashmap[ID[0] - 1, 1] = num_hashmap
            decoded[num_hashmap, :] = XOR
            num_hashmap += 1                                #update num_hashmap and decoded
            print 'aggiunto un decodificato', num_hashmap, decoding_indices[i], '\nXOR', XOR
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
                print 'aggiunto un decodificato', num_hashmap , decoding_indices[i] , '\nXOR' ,XOR
            elif not_decoded == 2:
                decoding_indices.append(decoding_indices[i])
                condition_vector[2] += 1

        i += 1  # increment cycle variable

    if num_hashmap < k:
        errati2 += 1        # if we do not decode the k pkts that we make an error

    # -- DEBUGGING -----------------------------------------------------------------------
    decoded2 = np.zeros((k, payload), dtype=np.int64)

    for i in xrange(len(sensors_indexes)):
        if hashmap[sensors_indexes[i], 0] == 1:
            a = hashmap[sensors_indexes[i], 1]
            decoded2[i, :] = decoded[a, :]

    diff = sum(sum(source_pkt - decoded2))
    if diff != 0:
        errati += 1
        print  decoded2, '\n\n'


# ----- FUORI DAL CICLO ----------

elapsed = time.time() - t
print 'Time taken by reiterated decoding procedure', elapsed

#print 'Sensor ID is %d and its position is (x=%d, y=%d) ' % (self.ID, self.X, self.Y)
print '\n errati metodo matrici  %d  mentre errati metodo somma semplice  %d' % (errati, errati2)

decoding_prob = (M - errati) / M
failure_prob = errati / M
print 'errorors', errati
print '\nThe decoding probability is ', decoding_prob
print '\nThe failure probability is ', failure_prob

decoding_prob = (M - errati2) / M
failure_prob = errati2 / M
print 'errorors', errati
print '\nThe decoding probability is ', decoding_prob
print '\nThe failure probability is ', failure_prob




