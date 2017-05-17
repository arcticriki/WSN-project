import numpy as np  # import of package numpy for mathematical tools
import random as rnd  # import of package random for homonym tools
import matplotlib.pyplot as plt  # import of package matplotlib.pyplot for plottools
import time as time
from Node2 import *         #importa la versione due della classe nodo
from send_mail import *
import cProfile
from RSD import *
from math import factorial
import csv
import copy

def main(n0, k0, eta0, C1, num_MP,L):
# -- PARAMETER INITIALIZATION SECTION --------------------------------------------------------------
    payload = 10
    C1 = C1
    eta = eta0
    n = n0                                   # number of nodes
    k = k0                                   # number of sensors
    #L = L                                    # square dimension
    c0 = 0.1                                 # parameter for RSD
    delta = 0.5                              # Prob['we're not able to recover the K pkts']<=delta

    positions = np.zeros((n, 2))  # matrix containing info on all node positions
    node_list = []  # list of references to node objects
    dmax = 5  # maximum distance for communication
    dmax2 = dmax * dmax  # square of maximum distance for communication
    sensors_indexes = rnd.sample(range(0, n), k)  # generation of random indices for sensors

# -- DEGREE INITIALIZATION --

    d, pdf = Robust_Soliton_Distribution(n, k, c0, delta)  # See RSD doc
    # to_be_encoded = np.sum(d)                        #use to check how many pkts we should encode ***OPEN PROBLEM***

# -- X_d INITIALIZATION --
    #THIS PARAMETER MUST BE COMPUTED THROUGH THE OPTIMIZATION PROBLEM
    Xd = np.ones(k)*1
    # compute denomitator of formula 5
    partial = 0
    for i in xrange(k):
        partial += Xd[i] * (i+1.0) * pdf[i]

    denominator = n * partial
    #rint 'n=',n, '\n\nX_d=', Xd, '\n\n\mu', pdf,'\n\ndenominator=', denominator

# -- NETWORK INITIALIZATION --
# Generation of storage nodes
    for i in xrange(n):  # for on 0 to n indices
        x = rnd.uniform(0.0, L)  # generation of random coordinate x
        y = rnd.uniform(0.0, L)  # generation of random coordinate y
        pid = (d[i]*Xd[d[i]-1])/denominator                 #compute steady state probability, formula 5 paper 1
                                                            # step 2 algorithm 1.
        node_list.append(Storage(i + 1, x, y, int(d[i]), n, k, C1, pid))  # creation of Storage node
        positions[i, :] = [x, y]


# Generation of sensor nodes
    for i in sensors_indexes:  # for on sensors position indices
        x = rnd.uniform(0.0, L)  # generation of random coordinate x
        y = rnd.uniform(0.0, L)  # generation of random coordinate y
        pid = (d[i]*Xd[d[i]-1])/denominator                 #compute steady state probability, formula 5 paper 1
                                                            # step 2 algorithm 1.
        node_list[i] = Sensor(i + 1, x, y, int(d[i]), n, k, C1, pid)  # creation of sensor node, function Sensor(), extend Storage class
        positions[i, :] = [x, y]  # support variable for positions info, used for comp. optim. reasons



# Find nearest neighbours using euclidean distance
    t = time.time()
    nearest_neighbor = []                       # simplifying assumption, if no neighbors exist withing the range
                                                # we consider the nearest neighbor
    nn_distance = 2 * L * L                     # maximum distance square equal the diagonal of the square [L,L]
    for i in xrange(n):                         # cycle on all nodes
        checker = False                         # boolean variable used to check if neighbors are found (false if not)
        for j in xrange(n):                     # compare each node with all the others
            x = positions[i, 0] - positions[j, 0]  # compute x distance between node i and node j
            y = positions[i, 1] - positions[j, 1]  # compute y distance between node i and node j
            dist2 = x * x + y * y               # compute distance square, avoid comp. of sqrt for comp. optim. reasons
            if dist2 <= dmax2:                  # check if distance square is less or equal the max coverage dist
                if dist2 != 0:                  # avoid considering self node as neighbor
                    node_list[i].neighbor_write(node_list[j])  # append operation on node's neighbor list
                    checker = True              # at least one neighbor has been founded
            if not checker and dist2 <= nn_distance and dist2 != 0:  # in order to be sure that the graph is connected
                                                # we determine the nearest neighbor
                                                # even if its distance is greater than the max distance
                nn_distance = dist2             # if distance of new NN is less than distance of previous NN, update it
                nearest_neighbor = node_list[i] # save NN reference, to use only if no neighbors are found

        if not checker:  # if no neighbors are found withing max dist, use NN
            print 'Node %d has no neighbors within the range, the nearest neighbor is chosen.' % i
            node_list[i].neighbor_write(nearest_neighbor)  # Connect node with NN

    elapsed = time.time() - t
    print '\nTempo di determinazione dei vicini:', elapsed

# Computation of probabilistic forwarding table
    for i in xrange(n):                         # we go through all nodes
        M = node_list[i].node_degree            # estract the numeber of neighbors of this node
        pi_i = node_list[i].pid                 # estract the pi greco d of this node
        node_list[i].metropolis_prob = np.zeros(M+1)  #initialize the vetor which will contain the transition probabilities
                                                # last node is the node itself, sembra dal metropolis che un nodo possa auto inviarsi
                                                # un pacchetto
        somma = 0
        for ii in xrange(M):                    # for each node we repeat the same operation for each attached neighbor
                                                # we use the metropolis algorithm to compute the transition probabilities
            neighbor = node_list[i].neighbor_list[ii]       # estract first neighbor of this node
            pi_j = neighbor.pid                             # estract pi greco d of this neighbor
            node_list[i].metropolis_prob[ii] = min(1.0, pi_j/pi_i)/M   # compute the transition probability of this neighbor
                                                # it follows formula 2 of paper 1
                                                # essentially it is the pdf of the transition prob, it can be use with
                                                # stats.rv_discrete along with the list of indicies to sample a value, as in the RSD
            somma += node_list[i].metropolis_prob[ii]       # keeps in account the cumulative distribution function
        node_list[i].metropolis_prob[-1] = 1-somma          # add last value, corresponding to the prob of self send
        node_list[i].neighbor_list.append(node_list[i])     # add itself in the neighbor list
        node_list[i].node_degree += 1



# Compute the number of random walks each sensing node must generate.
    numerator = 0
    for d in xrange(k):
        numerator += Xd[d] * (d+1.0) * pdf[d]
    b = int(n*numerator/k)
    print 'Number of random walks b =',b
    for i in sensors_indexes:
        node_list[i].number_random_walk = b

 # -- PKT GENERATION  --------------------------------------------------------------------------------------------------
    source_pkt = np.zeros((k, payload), dtype=np.int64)
    for i in xrange(k):
        source_pkt[i, :] = node_list[sensors_indexes[i]].pkt_gen2()

    #print source_pkt


# -- PKT  DISSEMINATION -----------------------------------------------------------------------------------------------
    j = 0
    t = time.time()
    print b*k
    while j < b * k:
        for i in xrange(n):
            if node_list[i].dim_buffer != 0:
                j += node_list[i].send_pkt(1)   # 1 means we use the metropolis algoritm for dissemination
            if j == b * k:
                break
        print j
    print 'Time taken by dissemination: ',time.time()-t


# -- XORING PRCEDURE ---------------------------------------------------------------------------------------------------
    for i in xrange(n):
        node_list[i].encoding()






    tot = 0
    distribution_post_dissemination = np.zeros(k + 1)       # ancillary variable used to compute the distribution post dissemination
    for i in xrange(n):
        index = node_list[i].num_encoded                    # retrive the actual encoded degree
        distribution_post_dissemination[index] += 1.0 / n   # augment the prob. value of the related degree
        tot += node_list[i].num_encoded                     # compute the total degree reached

    #return distribution_post_dissemination[1:], pdf
    plt.title('Post dissemination')
    y = distribution_post_dissemination[1:]
    x = np.linspace(1, k, k, endpoint=True)
    plt.axis([0, k, 0, 0.6])
    plt.plot(x, y, label='post dissemination')  # plot the robust pdf vs the obtained distribution after dissemination
    y2 = np.zeros(k)
    y2[:len(pdf)] = pdf
    plt.plot(x, y2, color='red', label='robust soliton')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

if __name__ == "__main__":

    n0 = 100
    k0 = 20
    eta0 = [1.8]
    C1 = 5
    num_MP = 10
    L = 15

    #main(n0, k0, eta0, C1, num_MP, L)
    cProfile.run('main(n0, k0, eta0, C1, num_MP, L)')