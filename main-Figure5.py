import numpy as np  # import of package numpy for mathematical tools
import random as rnd  # import of package random for homonym tools
import matplotlib.pyplot as plt  # import of package matplotlib.pyplot for plottools
import time as time
from Node import *
import cProfile
from RSD import *
from math import factorial
import csv


def main(n0,k0, eta0):
    # -- PARAMETER INITIALIZATION SECTION --------------------------------------------------------------
    payload = 10
    eta = eta0
    n = n0                                   # number of nodes
    k = k0                                   # number of sensors
    L = 5                                     # square dimension
    c0 = 0.1                                 # parameter for RSD
    delta = 0.5                              # Prob['we're not able to recover the K pkts']<=delta

    positions = np.zeros((n, 2))             # matrix containing info on all node positions
    node_list = []                           # list of references to node objects
    dmax = 5                                 # maximum distance for communication
    dmax2 = dmax * dmax                      # square of maximum distance for communication
    sensors_indexes = rnd.sample(range(0, n), k)  # generation of random indices for sensors

    # -- DEGREE INITIALIZATION --

    d, pdf = Robust_Soliton_Distribution(n, k, c0, delta)  # See RSD doc
    to_be_encoded = np.sum(d)                              # use to check how many pkts we should encode ***OPEN PROBLEM***

    # -- NETWORK INITIALIZATION --
    # -- Generation of storage nodes --
    for i in xrange(n):                     # for on 0 to n indices
        x = rnd.uniform(0.0, L)             # generation of random coordinate x
        y = rnd.uniform(0.0, L)             # generation of random coordinate y
        node_list.append(Storage(i + 1, x, y, d[i], n, k))  # creation of Storage node
        positions[i, :] = [x, y]

    # -- Generation of sensor nodes --
    for i in sensors_indexes:               # for on sensors position indices
        x = rnd.uniform(0.0, L)             # generation of random coordinate x
        y = rnd.uniform(0.0, L)             # generation of random coordinate y
        node_list[i] = Sensor(i + 1, x, y, d[i], n,k)  # creation of sensor node, function Sensor(), extend Storage class
        positions[i, :] = [x, y]            # support variable for positions info, used for comp. optim. reasons

    t = time.time()

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

    elapsed = time.time() - t
    #print 'Tempo di determinazione dei vicini:', elapsed

    # -- PKT GENERATION  --
    source_pkt = np.zeros((k, payload), dtype=np.int64)             # initialization of source pkt storage variable
    for i in xrange(k):
        source_pkt[i, :] = node_list[sensors_indexes[i]].pkt_gen()  # retrival of souce pkts payload


    # -- PKT  DISSEMINATION --
    j = 0                                   # ancillary variable
    while j < k:
        for i in xrange(n):
            if node_list[i].dim_buffer != 0:
                j += node_list[i].send_pkt(0)   # if the node has pkt to sent then the forward procedure is performed
                                                # 0 determine the metodology used to determine how to choose to which node
                                                # forward the pkt. 1 means Metropolis algorithm
            if j == k:
                break
    tot = 0
    distribution_post_dissemination = np.zeros(k + 1)       # ancillary variable used to compute the distribution post dissemination
    for i in xrange(n):
        index = node_list[i].num_encoded                    # retrive the actual encoded degree
        distribution_post_dissemination[index] += 1.0 / n   # augment the prob. value of the related degree
        tot += node_list[i].num_encoded                     # compute the total degree reached
    print '\nNumero di pacchetti codificati:', tot, 'su un totale di:', to_be_encoded, '\n'    #print the total degree reached

    plt.title('Post dissemination')
    y = distribution_post_dissemination[1:]
    x = np.linspace(1, k, k, endpoint=True)
    plt.axis([0, k, 0, 0.6])
    plt.plot(x,y , label='post dissemination')              # plot the robust pdf vs the obtained distribution after dissemination
    y2 = np.zeros(k)
    y2[:len(pdf)] = pdf
    plt.plot(x, y2, color='red', label='robust soliton')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


    # -- DECODING PHASE ---------------------------------------------------------------------------------------------------
    passo = 0.1                                 # incremental step of the epsilon variable
    decoding_performance = np.zeros(len(eta))         # ancillary variable which contains the decoding probability values
    for iii in xrange(len(eta)):

        #epsilon = int(passo * iii * k)          # computation of epsilon
        #h = k + epsilon                         # to succefully decode with high probability

        h = int(k * eta[0])

        errati2 = 0.0                           # Number of iteration in which we do not decode
        M = factorial(n) / (10 * factorial(h) * factorial(n - h))  # Computation of the number of iterations to perform, see paper 2

        num_iterazioni = 200                 # True number of iterations
        for ii in xrange(num_iterazioni):

            # -- parameters initialization phase --------------------
            decoding_indices = rnd.sample(range(0, n), h)  # selecting h random nodes in the graph
            hashmap = np.zeros((n,2))           # bidimensional vector which works as an hashmap, row indicate ID-1, position 0
                                                # reveal if that ID is been decoded or not, position 1 indicate at which row
                                                # in decoded varibale is stored the decoded payload

            num_hashmap = 0                     # key counter: indicates the index of the next free row in decoded matrix
            decoded = np.zeros((k, payload),dtype=np.int64)  # matrix k * payload: the i-th row stores the total XOR of the decoded pkts

            i = 0  # variabile accessori per il ciclo di iterazione sulla lista degli h nodi scelti tra gli n
            condition_vector = np.zeros(3)      # ancillary list of indicies, only MATTIA know how it works, ask him.
            condition_vector[0] = h             # first index represent the lenght of the vector of indicies

            while num_hashmap < k:  # While we have not decoded the k source pkts do:
                if i == condition_vector[0]:   # if we watched the whole vector enter this section
                    if condition_vector[1] == condition_vector[2] or condition_vector[2] == 0:
                        # if we queued the same number of pkt of previous round than we have not decoded any new pkt
                        # thus we will never decode new pkt, and we have a decoding failure.
                        break
                    else:
                        condition_vector = [condition_vector[0] + condition_vector[2], condition_vector[2], 0.0]
                        # Here we augment the index related to the lenght of the index list by the value of queued li

                degree, ID, XOR = node_list[decoding_indices[i]].storage_info()  # get the useful info

                if degree == 1 and hashmap[ID[0] - 1, 0] == 0:  # if the pkt has degree=1 -> immediately decoded
                    hashmap[ID[0] - 1, 0] = 1  # pkt decoded
                    hashmap[ID[0] - 1, 1] = num_hashmap
                    decoded[num_hashmap, :] = XOR
                    num_hashmap += 1  # update num_hashmap and decoded
                    # print 'aggiunto un decodificato', num_hashmap, decoding_indices[i], '\nXOR', XOR
                else:  # if the pkt has degree>1 -> investigate if is possible to decode, or wait
                    j = 0  # temp variable for the scanning process
                    not_decoded = 0  # number of undecoded pkt, over the total in vector ID
                    temp_ID = []  # temp list for un-processed ID pkts
                    while j < len(ID) and not_decoded < 2:  # we scan the IDs connected to the node
                        if hashmap[ID[j] - 1, 0] == 1:
                            for bit in xrange(payload):  # XOR bit per bit
                                x = hashmap[ID[j] - 1, 1]
                                XOR[bit] = XOR[bit] ^ decoded[
                                    x, bit]  # XOR(new)=XOR+decoded[the node which is connected to and has already been solved]
                            j += 1
                        else:
                            not_decoded += 1
                            temp_ID.append(ID[j])
                            j += 1

                    if not_decoded == 1:
                        hashmap[temp_ID[0] - 1, 0] = 1  # pkt decoded
                        hashmap[temp_ID[0] - 1, 1] = num_hashmap
                        decoded[num_hashmap, :] = XOR
                        num_hashmap += 1
                        # print 'aggiunto un decodificato', num_hashmap , decoding_indices[i] , '\nXOR' ,XOR
                    elif not_decoded == 2:
                        decoding_indices.append(decoding_indices[i])
                        condition_vector[2] += 1

                i += 1  # increment cycle variable

            if num_hashmap < k:
                print num_hashmap
                errati2 += 1  # if we do not decode the k pkts that we make an error

        decoding_performance[iii] = (num_iterazioni - errati2) / num_iterazioni

    return decoding_performance



if __name__ == "__main__":
    iteration_to_mediate = 10
    number_of_points_in_x_axis = 10
    y0 = np.zeros((iteration_to_mediate, number_of_points_in_x_axis))
    y1 = np.zeros((iteration_to_mediate, number_of_points_in_x_axis))
    #y2 = np.zeros((iteration_to_mediate, number_of_points_in_x_axis))
    #y3 = np.zeros((iteration_to_mediate, number_of_points_in_x_axis))

    # -- Iterazione su diversi sistemi --
    tempi1 = np.zeros((iteration_to_mediate,number_of_points_in_x_axis))
    tempi2 = np.zeros((iteration_to_mediate,number_of_points_in_x_axis))

    for i in xrange(iteration_to_mediate):
        tt = time.time()
        print 'iterazione',i+1,'di', iteration_to_mediate
        for ii in xrange(number_of_points_in_x_axis):
            t = time.time()
            y0[i, ii] = main(n0=500*(ii+1), k0=50*(ii+1), eta0=[1.4])
            elalpsed = time.time() - t
            tempi1[i, ii] = elalpsed
            print 'Caso eta = 1.4 e n =',500*(ii+1),'eseguito in',elalpsed,'secondi'

        for ii in xrange(number_of_points_in_x_axis):
            t = time.time()
            y1[i, ii] = main(n0=500*(ii+1), k0=50*(ii+1), eta0=[1.7])
            elalpsed = time.time() - t
            tempi2[i, ii] = elalpsed
            print 'Caso eta = 1.7 e n =', 500 * (ii + 1), 'eseguito in', elalpsed, 'secondi'
        print '\nTempo totale di ciclo =',time.time()-tt, '\n'


    tempi1 = tempi1.mean(0)
    tempi2 = tempi2.mean(0)

    plt.title('Tempi di esecuzione')
    x = np.linspace(0, 10, number_of_points_in_x_axis, endpoint=True)
    plt.axis([0, 10, 0, tempi2[9]])
    plt.plot(x, tempi1, label='eta 1.4', color='blue', linewidth=2)
    plt.plot(x, tempi2, label='eta 1.7', color='red', linewidth=2)
    # plt.plot(x, y2, label='200 nodes and 20 sources',color='grey'   ,linewidth=2)
    # plt.plot(x, y3, label='200 nodes and 40 sources',color='magenta',linewidth=2)
    plt.legend(loc=4)
    plt.grid()
    plt.show()

    y0 = y0.mean(0)     # calcolo delle prestazioni medie
    y1 = y1.mean(0)
    #y2 = y2.mean(0)
    #y3 = y3.mean(0)


    # -- Salvataggio su file --
    with open('Figura 5','wb') as file:
        wr=csv.writer(file,quoting=csv.QUOTE_ALL)
        wr.writerow(y0)
        wr.writerow(y1)
        #wr.writerow(y2)
        #wr.writerow(y3)

    # -- Plot --
    plt.title('Decoding performances')
    x = np.linspace(500, 5000, number_of_points_in_x_axis , endpoint=True)
    plt.axis([500, 5000, 0, 1])
    plt.plot(x, y0, label='eta 1.4',color='blue'   ,linewidth=2)
    plt.plot(x, y1, label='eta 1.7',color='red'    ,linewidth=2)
    #plt.plot(x, y2, label='200 nodes and 20 sources',color='grey'   ,linewidth=2)
    #plt.plot(x, y3, label='200 nodes and 40 sources',color='magenta',linewidth=2)
    plt.legend(loc=4)
    plt.grid()
    plt.show()



   #cProfile.run('main(n0=200, k0=40)')
