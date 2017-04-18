import numpy as np  # import of package numpy for mathematical tools
import random as rnd  # import of package random for homonym tools
import matplotlib.pyplot as plt  # import of package matplotlib.pyplot for plottools
import time as time
from Node import *
import cProfile
from RSD import *
from math import factorial
import csv
import copy

def main(n0,k0,eta0,C1):
    # -- PARAMETER INITIALIZATION SECTION --------------------------------------------------------------
    payload = 10
    C1 = C1
    eta = eta0
    n = n0                                   # number of nodes
    k = k0                                   # number of sensors
    L = 10                                   # square dimension
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

    # -- NETWORK INITIALIZATION --
    # Generation of storage nodes
    for i in xrange(n):  # for on 0 to n indices
        x = rnd.uniform(0.0, L)  # generation of random coordinate x
        y = rnd.uniform(0.0, L)  # generation of random coordinate y
        node_list.append(Storage(i + 1, x, y, d[i], n, k, C1))  # creation of Storage node
        positions[i, :] = [x, y]

    # Generation of sensor nodes
    for i in sensors_indexes:  # for on sensors position indices
        x = rnd.uniform(0.0, L)  # generation of random coordinate x
        y = rnd.uniform(0.0, L)  # generation of random coordinate y
        node_list[i] = Sensor(i + 1, x, y, d[i], n,k, C1)  # creation of sensor node, function Sensor(), extend Storage class
        positions[i, :] = [x, y]  # support variable for positions info, used for comp. optim. reasons

    # t = time.time()
    # Find nearest neighbours using euclidean distance
    nearest_neighbor = []  # simplifying assumption, if no neighbors exist withing the range
    # we consider the nearest neighbor
    nn_distance = 2 * L * L  # maximum distance square equal the diagonal of the square [L,L]
    for i in xrange(n):  # cycle on all nodes
        checker = False  # boolean variable used to check if neighbors are found (false if not)
        for j in xrange(n):  # compare each node with all the others
            x = positions[i, 0] - positions[j, 0]  # compute x distance between node i and node j
            y = positions[i, 1] - positions[j, 1]  # compute y distance between node i and node j
            dist2 = x * x + y * y  # compute distance square, avoid comp. of sqrt for comp. optim. reasons
            if dist2 <= dmax2:  # check if distance square is less or equal the max coverage dist
                if dist2 != 0:  # avoid considering self node as neighbor
                    node_list[i].neighbor_write(node_list[j])  # append operation on node's neighbor list
                    checker = True  # at least one neighbor has been founded
            if not checker and dist2 <= nn_distance and dist2 != 0:  # in order to be sure that the graph is connected
                # we determine the nearest neighbor
                # even if its distance is greater than the max distance
                nn_distance = dist2  # if distance of new NN is less than distance of previous NN, update it
                nearest_neighbor = node_list[i]  # save NN reference, to use only if no neighbors are found

        if not checker:  # if no neighbors are found withing max dist, use NN
            print 'Node %d has no neighbors within the range, the nearest neighbor is chosen.' % i
            node_list[i].neighbor_write(nearest_neighbor)  # Connect node with NN

    # elapsed = time.time() - t
    # print 'Tempo di determinazione dei vicini:', elapsed

    # -- PKT GENERATION  --------------------------------------------------------------------------------------------------
    source_pkt = np.zeros((k, payload), dtype=np.int64)
    for i in xrange(k):
        source_pkt[i, :] = node_list[sensors_indexes[i]].pkt_gen()

    # print '\nPacchetti generati \n', source_pkt


    # -- PKT  DISSEMINATION -----------------------------------------------------------------------------------------------
    j = 0
    while j < k:
        for i in xrange(n):
            if node_list[i].dim_buffer != 0:
                j += node_list[i].send_pkt(0)
            if j == k:
                break

    # -- DECODING PHASE ---------------------------------------------------------------------------------------------------
    # -- Initialization -------------------------

    passo = 0.1  # incremental step of the epsilon variable
    decoding_performance = np.zeros(len(eta))  # ancillary variable which contains the decoding probability values
    for iii in xrange(len(eta)):
        h = int(k * eta[iii])
        errati = 0.0  # Number of iteration in which we do not decode
        errati2 = 0.0
        M = factorial(n) / (10 * factorial(h) * factorial(n - h))  # Computation of the number of iterations to perform, see paper 2

        num_iterazioni = 500  # True number of iterations

        for ii in xrange(num_iterazioni):
            decoding_indices = rnd.sample(range(0, n), h)  # selecting h random nodes in the graph

            #print 'iterazione ',ii

            degrees = [0] * h
            IDs     = [0] * h
            XORs    = [0] * h

            for node in range(h):
                degree, ID, XOR = node_list[decoding_indices[node]].storage_info()

                degrees[node] = copy.deepcopy(degree)
                IDs[node] = copy.deepcopy(ID)
                XORs[node] = copy.deepcopy(XOR)

            # -- MP. Naive approach --------------------------------

            ripple_payload = []  # auxialiary vectors
            ripple_IDs = []
            hashmap = np.zeros((n, 2))  # vector nx2: pos[ID-1,0]-> "1" pkt of (ID-1) is decoded, "0" otherwise; pos[ID-1,1]->num_hashmap
            num_hashmap = 0  # key counter: indicates the index of the next free row in decoded matrix

            decoded = np.zeros((k, payload), dtype=np.int64)  # matrix k*payload: the i-th row stores the total XOR of the decoded pkts
            empty_ripple = False

            while (empty_ripple == False):

                empty_ripple = True

                position = 0  # linear search of degree one nodes

                while position < len(degrees):

                    if degrees[position] == 1:  # if degree 1 is found

                        if hashmap[IDs[position][0] - 1, 0] == 0:
                            decoded[num_hashmap, :] = XORs[position]
                            hashmap[IDs[position][0] - 1, 0] = 1
                            hashmap[IDs[position][0] - 1, 1] = num_hashmap
                            num_hashmap += 1
                        empty_ripple = False
                        del degrees[position]  # decrease degree
                        ripple_IDs.append(IDs[position])  # update ripples
                        del IDs[position]
                        ripple_payload.append(XORs[position])
                        del XORs[position]  # update vector XORs
                    else:
                        position = position + 1

                # scanning the ripple
                for each_element in ripple_IDs:  # prendi ogni elemento del ripple...
                    for each_node in IDs:  # ...e ogni elemento del vettore degli ID...
                        u = 0
                        while u < len(each_node):
                            if each_element[0] == each_node[u]:
                                indice_ID = IDs.index(each_node)
                                degrees[indice_ID] -= 1
                                indice_ripple = ripple_IDs.index(each_element)
                                XORs[indice_ID] = XORs[indice_ID] ^ ripple_payload[indice_ripple]
                                temp = each_node
                                del temp[u]
                                IDs[indice_ID] = temp
                                each_node = temp

                            else:
                                u += 1

                i = 0
                while i < len(IDs):
                    if degrees[i] == 0:
                        IDs.remove([])
                        # XORs.remove(XORs[i])
                        del XORs[i]
                        degrees.remove(0)
                    else:
                        i += 1

            # decoded2 = np.zeros((k, payload), dtype=np.int64)
            #
            # for iiii in xrange(len(sensors_indexes)):
            #     if hashmap[sensors_indexes[iiii], 0] == 1:
            #         a = hashmap[sensors_indexes[iiii], 1]
            #         decoded2[iiii, :] = decoded[a, :]
            #
            # diff = sum(sum(source_pkt - decoded2))
            # if diff != 0:
            #     errati += 1

            if num_hashmap < k:
                errati2 += 1  # if we do not decode the k pkts that we make an error

        decoding_performance[iii] = (num_iterazioni - errati2) / num_iterazioni

    return decoding_performance



if __name__ == "__main__":
    iteration_to_mediate = 3
    eta = [1,1.1,1.2,1.3,1.4,1.5,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5]
    y0 = np.zeros((iteration_to_mediate,len(eta)))
    y1 = np.zeros((iteration_to_mediate,len(eta)))
    y2 = np.zeros((iteration_to_mediate,len(eta)))
    y3 = np.zeros((iteration_to_mediate,len(eta)))

    # -- Iterazione su diversi sistemi --
    for i in xrange(iteration_to_mediate):
        t = time.time()
        print i
        y0[i,:] = main(n0=100, k0=10, eta0=eta, C1=3)
        y1[i,:] = main(n0=100, k0=20, eta0=eta, C1=3)
        y2[i,:] = main(n0=200, k0=20, eta0=eta, C1=3)
        y3[i,:] = main(n0=200, k0=40, eta0=eta, C1=3)
        print time.time()-t

    y0 = y0.mean(0)     # calcolo delle prestazioni medie
    y1 = y1.mean(0)
    y2 = y2.mean(0)
    y3 = y3.mean(0)


    # -- Salvataggio su file --
    with open('Prova','wb') as file:
        wr=csv.writer(file,quoting=csv.QUOTE_ALL)
        wr.writerow(y0)
        wr.writerow(y1)
        wr.writerow(y2)
        wr.writerow(y3)

    # -- Plot --
    plt.title('Decoding performances')
    x = np.linspace(1, 2.5, 16, endpoint=True)
    plt.axis([1, 2.5, 0, 1])
    plt.plot(x, y0, label='100 nodes and 10 sources',color='blue'   ,linewidth=2)
    plt.plot(x, y1, label='100 nodes and 20 sources',color='red'    ,linewidth=2)
    plt.plot(x, y2, label='200 nodes and 20 sources',color='grey'   ,linewidth=2)
    plt.plot(x, y3, label='200 nodes and 40 sources',color='magenta',linewidth=2)
    plt.legend(loc=4)
    plt.grid()
    plt.show()



   #cProfile.run('main(n0=200, k0=40)')
