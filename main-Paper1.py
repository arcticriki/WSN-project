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
    c0 = 0.01                                 # parameter for RSD
    delta = 0.01                              # Prob['we're not able to recover the K pkts']<=delta

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
    Xd = np.ones(k)*3
    #Xd = [2.99573, 2.43482,  2.21432,  2.10541,  2.05016,  2.0268,  2.02545,  2.04131, 2.07213,  2.11726,  2.17724,  2.2538,
    #      2.35006,  2.47121,  2.62574, 2.82823,  3.106, 3.51784,  4.22502, 5.96721]
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
        pid = (d[i]*Xd[int(d[i])-1])/denominator                 #compute steady state probability, formula 5 paper 1
                                                            # step 2 algorithm 1.
        node_list.append(Storage(i + 1, x, y, int(d[i]), n, k, C1, pid))  # creation of Storage node
        positions[i, :] = [x, y]


    plt.title('Post dissemination')
    y1 = positions[:,1]
    y2 = positions[sensors_indexes,1]

    x1 = positions[:,0]
    x2 = positions[sensors_indexes,0]
    plt.axis([0, L, 0, L])
    plt.plot(x1, y1, linestyle='', marker='o', label='STORAGE')  # plot the robust pdf vs the obtained distribution after dissemination
    plt.plot(x2, y2, linestyle='', marker='o', color='red', label='SENSORS')
    plt.legend(loc='upper left')
    plt.grid()
    #plt.show(block=False)
    plt.savefig('Immagini/Paper1_algo1/00_disposition_n='+str(n)+'_k='+str(k)+'.pdf', dpi=150, transparent=False)
    plt.close()

# Generation of sensor nodes
    for i in sensors_indexes:  # for on sensors position indices
        x = rnd.uniform(0.0, L)  # generation of random coordinate x
        y = rnd.uniform(0.0, L)  # generation of random coordinate y
        pid = (d[i]*Xd[int(d[i])-1])/denominator                 #compute steady state probability, formula 5 paper 1
                                                            # step 2 algorithm 1.
        node_list[i] = Sensor(i + 1, x, y, int(d[i]), n, k, C1, pid)  # creation of sensor node, function Sensor(), extend Storage class
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
    b = int(round(n*numerator/k))
    #print 'Number of random walks b =',b
    for i in sensors_indexes:
        node_list[i].number_random_walk = b

 # -- PKT GENERATION  --------------------------------------------------------------------------------------------------
    source_pkt = np.zeros((k, payload), dtype=np.int64)
    codificati_in_partenza = 0
    for i in xrange(k):
        source_pkt[i, :], a = node_list[sensors_indexes[i]].pkt_gen2()
        codificati_in_partenza += a
    #print source_pkt
    #print 'Codificati dai sensori ',codificati_in_partenza


# -- PKT  DISSEMINATION -----------------------------------------------------------------------------------------------
    for i in xrange(n):
        node_list[i].funzione_ausiliaria()

    j = 0
    t = time.time()
    #print 'Total pkt to be disseminated ',b*k
    while j < (b * k)-codificati_in_partenza:
        for i in xrange(n):
            if node_list[i].dim_buffer != 0:
                j += node_list[i].send_pkt(1)   # 1 means we use the metropolis algoritm for dissemination
            if j == (b * k)-codificati_in_partenza:
                break
        #print j
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
    #plt.show(block=False)
    plt.savefig('Immagini/Paper1_algo1/00_post_diss_n='+str(n)+'_k='+str(k)+'.pdf', dpi=150, transparent=False)
    plt.close()


# -- DECODING PHASE ---------------------------------------------------------------------------------------------------
# -- Initialization -------------------------
    t = time.time()
    passo = 0.1  # incremental step of the epsilon variable
    decoding_performance = np.zeros(len(eta))  # ancillary variable which contains the decoding probability values
    for iii in xrange(len(eta)):
        h = int(k * eta[iii])
        errati = 0.0  # Number of iteration in which we do not decode
        errati2 = 0.0
        M = factorial(n) / (10 * factorial(h) * factorial(n - h))  # Computation of the number of iterations to perform, see paper 2

        for ii in xrange(num_MP):
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

            if num_hashmap < k:
                errati2 += 1  # if we do not decode the k pkts that we make an error

        decoding_performance[iii] = (num_MP - errati2) / num_MP

    print 'Time taken by message passing:', time.time()-t

    return decoding_performance




if __name__ == "__main__":
    #cProfile.run('main(n0, k0, eta0, C1, num_MP, L)')



    print 'Figure 3 and 4. \n'
    iteration_to_mediate = 1
    eta = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5]

    y0 = np.zeros((iteration_to_mediate, len(eta)))
    y1 = np.zeros((iteration_to_mediate, len(eta)))
    y2 = np.zeros((iteration_to_mediate, len(eta)))
    y3 = np.zeros((iteration_to_mediate, len(eta)))
    #y4 = np.zeros((iteration_to_mediate, len(eta)))
    #y5 = np.zeros((iteration_to_mediate, len(eta)))

    # -- Iterazione su diversi sistemi --
    for i in xrange(iteration_to_mediate):
        t = time.time()
        tt = time.time()
        y0[i, :] = main(n0=100, k0=10, eta0=eta, C1=5, num_MP=3000, L=5)
        #elapsed = time.time() - tt
        #print elapsed
        tt = time.time()
        y1[i, :] = main(n0=100, k0=20, eta0=eta, C1=5, num_MP=3000, L=5)
        #elapsed = time.time() - tt
        #print elapsed
        tt = time.time()
        y2[i, :] = main(n0=200, k0=20, eta0=eta, C1=5, num_MP=3000, L=5)
        #elapsed = time.time() - tt
        #print elapsed
        tt = time.time()
        y3[i, :] = main(n0=200, k0=40, eta0=eta, C1=5, num_MP=3000, L=5)
        #elapsed = time.time() - tt
        #print elapsed
        # tt = time.time()
        # y4[i, :] = main(n0=500, k0=50, eta0=eta, C1=5, num_MP= 1000, L=5)
        # elapsed = time.time() - tt
        # print elapsed
        # tt = time.time()
        # y5[i, :] = main(n0=1000, k0=100, eta0=eta, C1=5, num_MP= 1000, L=5)
        # elapsed = time.time() - tt
        # print elapsed
        elapsed = time.time() - t
        print 'Iterazione', i + 1, 'di', iteration_to_mediate, 'eseguita in', elapsed, 'secondi'

    y0 = y0.mean(0)  # calcolo delle prestazioni medie
    y1 = y1.mean(0)
    y2 = y2.mean(0)
    y3 = y3.mean(0)

    # -- Salvataggio su file --
    with open('Figure3Paper1.txt','wb') as file:
         wr=csv.writer(file,quoting=csv.QUOTE_ALL)
         wr.writerow(y0)
         wr.writerow(y1)
         wr.writerow(y2)
         wr.writerow(y3)

    plt.title('Decoding performances')
    x = np.linspace(1, 2.5, 16, endpoint=True)
    plt.axis([1, 2.5, 0, 1])
    plt.plot(x, y0, label='100 nodes and 10 sources', color='blue', linewidth=2)
    plt.plot(x, y1, label='100 nodes and 20 sources', color='red', linewidth=2)
    plt.plot(x, y2, label='200 nodes and 20 sources', color='grey', linewidth=2)
    plt.plot(x, y3, label='200 nodes and 40 sources', color='magenta', linewidth=2)
    plt.legend(loc=4)
    plt.grid()
    plt.savefig('Immagini/Paper1_algo1/00_Figure3_comparison.pdf', dpi=150, transparent=False)
    plt.close()

    #names = ['Figure3Paper1.txt']
    #send_mail(names)


