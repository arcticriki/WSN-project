import numpy as np  # import of package numpy for mathematical tools
import random as rnd  # import of package random for homonym tools
import matplotlib.pyplot as plt  # import of package matplotlib.pyplot for plottools
import time as time
from Node import *         #importa la versione due della classe nodo
import cProfile
from RSD import *
from math import factorial
import csv
import copy
from message_passing import *
from joblib import Parallel, delayed
import multiprocessing

def main(n0, k0, eta0, C1, num_MP,L):
# -- PARAMETER INITIALIZATION SECTION --------------------------------------------------------------
    payload = 1
    C1 = C1
    eta = eta0
    n = n0                                   # number of nodes
    k = k0                                   # number of sensors
    #L = L                                   # square dimension
    c0 = 0.01                                 # parameter for RSD
    delta = 0.5                              # Prob['we're not able to recover the K pkts']<=delta

    positions = np.zeros((n, 2))  # matrix containing info on all node positions
    node_list = []  # list of references to node objects
    dmax = 5  # maximum distance for communication
    dmax2 = dmax * dmax  # square of maximum distance for communication
    sensors_indexes = rnd.sample(range(0, n), k)  # generation of random indices for sensors

# -- DEGREE INITIALIZATION --

    d, pdf, R = Robust_Soliton_Distribution(n, k, c0, delta)  # See RSD doc
    # to_be_encoded = np.sum(d)                        #use to check how many pkts we should encode ***OPEN PROBLEM***

# -- NETWORK INITIALIZATION --
# Generation of storage nodes
    for i in xrange(n):  # for on 0 to n indices
        x = rnd.uniform(0.0, L)  # generation of random coordinate x
        y = rnd.uniform(0.0, L)  # generation of random coordinate y
        node_list.append(Storage(i + 1, x, y, int(d[i]), n, k, 0, 0, 0, 0, 0, c0, delta))  # creation of Storage node

        positions[i, :] = [x, y]


# Generation of sensor nodes
    for i in sensors_indexes:  # for on sensors position indices
        x = rnd.uniform(0.0, L)  # generation of random coordinate x
        y = rnd.uniform(0.0, L)  # generation of random coordinate y
        node_list[i] = Sensor(i + 1, x, y, int(d[i]), n, k, 0, 0, 0, 0, 0, c0, delta)  # creation of sensor node, function Sensor(), extend Storage class
        positions[i, :] = [x, y]  # support variable for positions info, used for comp. optim. reasons



# Find nearest neighbours using euclidean distance
    #t = time.time()
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

    #elapsed = time.time() - t
    #print '\nTempo di determinazione dei vicini:', elapsed


 # -- PKT GENERATION  --------------------------------------------------------------------------------------------------
    source_pkt = np.zeros((k, payload), dtype=np.int64)
    for i in xrange(k):
        source_pkt[i, :] = node_list[sensors_indexes[i]].pkt_gen3()
    #print source_pkt


# -- PKT  DISSEMINATION -----------------------------------------------------------------------------------------------
    for i in xrange(n):
        d_i = node_list[i].code_degree
        idxes = rnd.sample(sensors_indexes, d_i)  # generation of random indices for sensor
        store = np.zeros(payload, dtype=np.int64)

        for ii in idxes:
            node_list[i].ID_list.append(ii)
            store = store ^ node_list[ii].pkt_generated_gen3
        node_list[i].storage = store
        node_list[i].num_encoded = d_i

# -- DECODING PHASE ---------------------------------------------------------------------------------------------------
# -- Initialization -------------------------
    t = time.time()
    passo = 0.1                                # incremental step of the epsilon variable
    decoding_performance = np.zeros(len(eta))  # ancillary variable which contains the decoding probability values
    for iii in xrange(len(eta)):
        h = int(k * eta[iii])
        errati = 0.0                           # Number of iteration in which we do not decode

        for x in xrange(num_MP):
            errati += message_passing(node_list, n, k, h)

        decoding_performance[iii] = (num_MP - errati) / num_MP

    # print 'Time taken by message passing:', time.time()-t
    return decoding_performance

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    print 'Numero di core utilizzati:', num_cores

    print 'Figure 3 and 4. \n'
    iteration_to_mediate = 4
    eta = np.arange(1.0,2.6,step=0.1)

    y0 = np.zeros((iteration_to_mediate, len(eta)))
    y1 = np.zeros((iteration_to_mediate, len(eta)))
    y2 = np.zeros((iteration_to_mediate, len(eta)))
    y3 = np.zeros((iteration_to_mediate, len(eta)))
    y4 = np.zeros((iteration_to_mediate, len(eta)))
    y5 = np.zeros((iteration_to_mediate, len(eta)))


    # -- Iterazione su diversi sistemi --

    t = time.time()
    tt= time.time()
    y0 = Parallel(n_jobs=num_cores)(delayed(main)(n0=100, k0=10, eta0=eta, C1=5, num_MP= 3000, \
                                                  L=5) for ii in xrange(iteration_to_mediate))
    print 'n=100 k=10: ', time.time() - tt
    tt = time.time()
    y1 = Parallel(n_jobs=num_cores)(delayed(main)(n0=100, k0=20, eta0=eta, C1=5, num_MP=3000, \
                    L=5) for ii in xrange(iteration_to_mediate))
    print 'n=100 k=20: ', time.time() - tt
    tt = time.time()
    y2 = Parallel(n_jobs=num_cores)(delayed(main)(n0=200, k0=20, eta0=eta, C1=5, num_MP=3000, \
                    L=5) for ii in xrange(iteration_to_mediate))
    print 'n=200 k=20: ', time.time() - tt
    tt = time.time()
    y3 = Parallel(n_jobs=num_cores)(delayed(main)(n0=200, k0=40, eta0=eta, C1=5, num_MP=3000, \
                    L=5) for ii in xrange(iteration_to_mediate))
    print 'n=200 k=40: ', time.time() - tt
    tt = time.time()
    y4 = Parallel(n_jobs=num_cores)(delayed(main)(n0=500, k0=50, eta0=eta, C1=5, num_MP=3000, \
                    L=5) for ii in xrange(iteration_to_mediate))
    print 'n=500 k=50: ', time.time() - tt
    tt = time.time()
    y5 = Parallel(n_jobs=num_cores)(delayed(main)(n0=1000, k0=100, eta0=eta, C1=5, num_MP=3000, \
                    L=5) for ii in xrange(iteration_to_mediate))
    print 'n=1000 k=100: ', time.time() - tt



    y0 = np.sum(y0, 0) / iteration_to_mediate
    y1 = np.sum(y1, 0) / iteration_to_mediate
    y2 = np.sum(y2, 0) / iteration_to_mediate
    y3 = np.sum(y3, 0) / iteration_to_mediate
    y4 = np.sum(y4, 0) / iteration_to_mediate
    y5 = np.sum(y5, 0) / iteration_to_mediate


    # # -- Salvataggio su file --
    # with open('Figure 3-Paper1','wb') as file:
    #     wr=csv.writer(file,quoting=csv.QUOTE_ALL)
    #     wr.writerow(y0)
    #     wr.writerow(y1)
    #     wr.writerow(y2)
    #     wr.writerow(y3)
    #
    #
    # with open('Figure 4-Paper1', 'wb') as file:
    #     wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    #     wr.writerow(y2)
    #     wr.writerow(y4)
    #     wr.writerow(y5)


    plt.axis([1, eta[-1], 0, 1])
    plt.xlabel('Decoding ratio $\eta$')
    plt.ylabel('Successfull decoding probability P$_s$')
    x = np.linspace(1, eta[-1], len(y0), endpoint=True)
    plt.plot(x, y0, label='100 nodes and 10 sources', color='blue', linewidth=1, marker='o', markersize=4.0)
    plt.plot(x, y1, label='100 nodes and 20 sources', color='red', linewidth=1, marker='o', markersize=4.0)
    plt.plot(x, y2, label='200 nodes and 20 sources', color='grey', linewidth=1, marker='o', markersize=4.0)
    plt.plot(x, y3, label='200 nodes and 40 sources', color='magenta', linewidth=1, marker='o', markersize=4.0)
    plt.rc('legend', fontsize=10)
    plt.legend(loc=4)
    plt.grid()
    plt.savefig('Immagini/Paper1_algo1/00_Figure3_comparison_ottimo_two_way.pdf', dpi=150, transparent=False)
    plt.close()

    plt.axis([1, eta[-1], 0, 1])
    plt.xlabel('Decoding ratio $\eta$')
    plt.ylabel('Successfull decoding probability P$_s$')
    x = np.linspace(1, eta[-1], len(y0), endpoint=True)
    plt.plot(x, y2, label='200 nodes and 40 sources', color='blue', linewidth=1, marker='o', markersize=4.0)
    plt.plot(x, y4, label='500 nodes and 50 sources', color='red', linewidth=1, marker='o', markersize=4.0)
    plt.plot(x, y5, label='1000 nodes and 100 sources', color='grey', linewidth=1, marker='o', markersize=4.0)
    plt.rc('legend', fontsize=10)
    plt.legend(loc=4)
    plt.grid()
    plt.savefig('Immagini/Paper1_algo1/00_Figure4_comparison_ottimo_two_way.pdf', dpi=150, transparent=False)
    plt.close()
