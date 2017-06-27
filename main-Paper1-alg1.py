import numpy as np  # import of package numpy for mathematical tools
import random as rnd  # import of package random for homonym tools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # import of package matplotlib.pyplot for plottools
import time as time
from Node2 import *         #importa la versione due della classe nodo
from send_mail import *
import cProfile
from RSD import *
from math import factorial
import csv
import copy
from joblib import Parallel, delayed
import multiprocessing


def message_passing(node_list,n, k, h):
    decoding_indices = rnd.sample(range(0, n), h)  # selecting h random nodes in the graph


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
    hashmap = np.zeros(
        (n, 2))  # vector nx2: pos[ID-1,0]-> "1" pkt of (ID-1) is decoded, "0" otherwise; pos[ID-1,1]->num_hashmap
    num_hashmap = 0  # key counter: indicates the index of the next free row in decoded matrix

    decoded = np.zeros((k, payload),
                       dtype=np.int64)  # matrix k*payload: the i-th row stores the total XOR of the decoded pkts
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
        return 1  # if we do not decode the k pkts that we make an error
    else:
        return 0


#from plot_grafo import *

c0 = 0.2
delta = 0.05
def main(n0, k0, eta0, C1, num_MP,L,length_random_walk,solution):
# -- PARAMETER INITIALIZATION SECTION --------------------------------------------------------------
    payload = 10
    C1 = C1
    C2 = 0
    C3= 0
    eta = eta0
    n = n0                                   # number of nodes
    k = k0                                   # number of sensors
    #L = L                                   # square dimension
    #c0 = 0.01                                 # parameter for RSD
    #delta = 0.05                              # Prob['we're not able to recover the K pkts']<=delta

    positions = np.zeros((n, 2))             # matrix containing info on all node positions
    node_list = []                           # list of references to node objects
    dmax = 1                                 # maximum distance for communication, posto a 1.5 per avere 21 neighbors medi
    dmax2 = dmax * dmax                      # square of maximum distance for communication
    sensors_indexes = rnd.sample(range(0, n), k)  # generation of random indices for sensors


# -- DEGREE INITIALIZATION --

    d, pdf, _ = Robust_Soliton_Distribution(n, k, c0, delta)  # See RSD doc

# -- X_d INITIALIZATION --
    #THIS PARAMETER MUST BE COMPUTED THROUGH THE OPTIMIZATION PROBLEM
    Xd = 0
    if solution == 'ones':
        Xd = np.ones(k)
        #print 'Generati xd ones'

    if solution == 'Larghi_Math':
        with open('Dati/OptProblem1/'+solution+'/'+str(k)+'_'+str(c0)+'_'+str(delta)+'_L_KR.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)  # , quotechar='|')
            for row in reader:
                Xd = row
        for i in xrange(len(Xd)):
            Xd[i] = float(Xd[i])
        #print len(Xd)
        #print 'Caricati Xd '+ solution

    if solution == 'Stretti_Math':
        with open('Dati/OptProblem1/'+solution+'/'+str(k)+'_'+str(c0)+'_'+str(delta)+'_S_KR.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)  # , quotechar='|')
            for row in reader:
                Xd = row
        for i in xrange(len(Xd)):
            Xd[i] = float(Xd[i])
        #print len(Xd)
        #print 'Caricati Xd ' + solution




    # compute denomitator of formula 5
    partial = 0
    for i in xrange(k):
        partial += Xd[i] * (i+1.0) * pdf[i]

    denominator = n * partial

# -- NETWORK INITIALIZATION --
# Generation of storage nodes
    for i in xrange(n):  # for on 0 to n indices
        x = rnd.uniform(0.0, L)  # generation of random coordinate x
        y = rnd.uniform(0.0, L)  # generation of random coordinate y
        pid = (d[i]*Xd[int(d[i])-1])/denominator                 #compute steady state probability, formula 5 paper 1
                                                            # step 2 algorithm 1.
        node_list.append(Storage(i + 1, x, y, int(d[i]), n, k, C1,C2, C3, pid, length_random_walk, c0, delta))  # creation of Storage node
        positions[i, :] = [x, y]


# Generation of sensor nodes
    for i in sensors_indexes:  # for on sensors position indices
        x = rnd.uniform(0.0, L)  # generation of random coordinate x
        y = rnd.uniform(0.0, L)  # generation of random coordinate y
        pid = (d[i]*Xd[int(d[i])-1])/denominator                 #compute steady state probability, formula 5 paper 1
                                                            # step 2 algorithm 1.
        node_list[i] = Sensor(i + 1, x, y, int(d[i]), n, k, C1, C2, C3, pid,length_random_walk, c0, delta)  # creation of sensor node, function Sensor(), extend Storage class
        positions[i, :] = [x, y]  # support variable for positions info, used for comp. optim. reasons




# Find nearest neighbours using euclidean distance
    #t = time.time()
    for i in xrange(n):                     # cycle on all nodes
        nearest_neighbor = []               # simplifying assumption, if no neighbors exist withing the range
                                            # we consider the nearest neighbor
        d_min = 2 * L * L                   # maximum distance square equal the diagonal of the square [L,L]
        checker = False                     # boolean variable used to check if neighbors are found (false if not)
        for j in xrange(n):                 # compare each node with all the others
            x = positions[i, 0] - positions[j, 0]  # compute x distance between node i and node j
            y = positions[i, 1] - positions[j, 1]  # compute y distance between node i and node j
            dist2 = x * x + y * y           # compute distance square, avoid comp. of sqrt for comp. optim. reasons
            if dist2 <= dmax2:              # check if distance square is less or equal the max coverage dist
                if dist2 != 0:              # avoid considering self node as neighbor
                    node_list[i].neighbor_write(node_list[j])  # append operation on node's neighbor list
                    checker = True          # at least one neighbor has been founded
            elif dist2 <= d_min:
                d_min = dist2
                nearest_neighbor =node_list[j]

        if not checker:                     # if no neighbors are found withing max dist, use NN
            #print 'Node %d has no neighbors within the range, the nearest neighbor is chosen.' % i
            node_list[i].neighbor_write(nearest_neighbor)  # Connect node with NN


    # Plot the network topology
    #plot_grafo(node_list, n, k, sensors_indexes,L)




    #elapsed = time.time() - t
    #print '\nTempo di determinazione dei vicini:', elapsed

    #medio = 0
    #for i in xrange(n):
    #    medio += node_list[i].node_degree                  # compute the mean number of neighbors
    #print 'Numero medio di vicini',medio/n

    #plot_grafo(node_list, n , k , sensors_indexes)         # plot the graph representing the network


# Computation of probabilistic forwarding table
    for i in xrange(n):                         # we go through all nodes
        M = node_list[i].node_degree            # estract the numeber of neighbors of this node
        pi_i = node_list[i].pid                 # estract the pi greco d of this node
        node_list[i].metropolis_prob = np.zeros(M+1)  #initialize the vetor which will contain the transition probabilities
                                                # last node is the node itself, sembra dal metropolis che un nodo possa auto inviarsi
                                                # un pacchetto
        #node_list[i].cumulative_prob = np.ones(M+1)
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
            #node_list[i].cumulative_prob[ii] = somma

        node_list[i].metropolis_prob[-1] = 1-somma
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
    #t= time.time()
    for i in xrange(k):
        source_pkt[i, :], a = node_list[sensors_indexes[i]].pkt_gen2()
        codificati_in_partenza += a
    #print source_pkt
    #print 'Codificati dai sensori ',codificati_in_partenza
    #print 'Time taken by pkt generation', time.time()-t



# -- PKT  DISSEMINATION -----------------------------------------------------------------------------------------------

    for i in node_list:
        i.funzione_ausiliaria()

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
    #print 'Time taken by dissemination: ',time.time()-t



# -- XORING PRCEDURE ---------------------------------------------------------------------------------------------------
    #t = time.time()
    for i in xrange(n):
        node_list[i].encoding()
    #print 'Time taken by encoding: ', time.time() - t

    tot = 0
    distribution_post_dissemination = np.zeros(k + 1)       # ancillary variable used to compute the distribution post dissemination
    for i in xrange(n):
        index = node_list[i].num_encoded                    # retrive the actual encoded degree
        distribution_post_dissemination[index] += 1.0 / n   # augment the prob. value of the related degree
        tot += node_list[i].num_encoded                     # compute the total degree reached

    # return distribution_post_dissemination[1:], pdf
    plt.title('Post dissemination - '+solution)
    y = distribution_post_dissemination[1:]
    x = np.linspace(1, k, k, endpoint=True)
    plt.axis([0, k, 0, 0.6])
    plt.plot(x, y, label='post dissemination')  # plot the robust pdf vs the obtained distribution after dissemination
    y2 = np.zeros(k)
    y2[:len(pdf)] = pdf
    plt.plot(x, y2, color='red', label='robust soliton')
    plt.legend(loc=1)
    plt.grid()
    #plt.show(block=False)
    plt.savefig('Immagini/Paper1_algo1/Post_dissemination/post_diss_RW='+str(length_random_walk)+'_n='+str(n)+'_k='+str(k)+'_values'+solution+'.pdf', dpi=150, transparent=False)
    plt.close()



# -- DECODING PHASE --------
# -- Initialization -------------------------
    t = time.time()
    passo = 0.1  # incremental step of the epsilon variable
    decoding_performance = np.zeros(len(eta))  # ancillary variable which contains the decoding probability values
    for iii in xrange(len(eta)):
        h = int(k * eta[iii])
        errati = 0.0  # Number of iteration in which we do not decode

        for x in xrange(num_MP):
            errati += message_passing(node_list,n,k,h)

        decoding_performance[iii] = (num_MP - errati) / num_MP


    #print 'Time taken by message passing:', time.time()-t

    return decoding_performance



























if __name__ == "__main__":
    #cProfile.run('main(n0, k0, eta0, C1, num_MP, L)')

    num_cores = multiprocessing.cpu_count()
    print 'Numero di core utilizzati:', num_cores
## Decoding pribability of several network dimensions -----------------------------------------------------------------
    # iteration_to_mediate = 1
    # #punti = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 160, 200, 500, 1000])
    # punti = [50]
    #
    # for length_random_walk in punti:
    #     print 'Lunghezza della random walk:', length_random_walk
    #
    #     eta = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]
    #     # sol = 'ones'
    #     sol = 'Larghi_Math'
    #     # sol = 'Stretti_Math'
    #     y0 = np.zeros((iteration_to_mediate, len(eta)))
    #     y1 = np.zeros((iteration_to_mediate, len(eta)))
    #     y2 = np.zeros((iteration_to_mediate, len(eta)))
    #     y3 = np.zeros((iteration_to_mediate, len(eta)))
    #     y3 = np.zeros((iteration_to_mediate, len(eta)))
    #     y3 = np.zeros((iteration_to_mediate, len(eta)))
    #
    #     mp1 = 2500
    #     mp2 = 2000
    #     mp3 = 2000
    #
    #     # mp1 = 1
    #     # mp2 = 1
    #     # mp3 = 1
    #     # -- Iterazione su diversi sistemi --
    #     parallel = time.time()
    #     tt = time.time()
    #     y0 = Parallel(n_jobs=num_cores)(delayed(main)(n0=100, k0=10, eta0=eta, C1=5, num_MP=mp1, L=5,length_random_walk=length_random_walk,solution=sol) for ii in xrange(iteration_to_mediate))
    #     print 'n=100 k=10: ', time.time() - tt
    #     tt = time.time()
    #     y1 = Parallel(n_jobs=num_cores)(delayed(main)(n0=100, k0=20, eta0=eta, C1=5, num_MP=mp1, L=5,length_random_walk=length_random_walk,solution=sol) for ii in xrange(iteration_to_mediate))
    #     print 'n=100 k=20: ', time.time() - tt
    #     tt = time.time()
    #     y2 = Parallel(n_jobs=num_cores)(delayed(main)(n0=200, k0=20, eta0=eta, C1=5, num_MP=mp1, L=5,length_random_walk=length_random_walk,solution=sol) for ii in xrange(iteration_to_mediate))
    #     print 'n=200 k=20: ', time.time() - tt
    #     tt = time.time()
    #     y3 = Parallel(n_jobs=num_cores)(delayed(main)(n0=200, k0=40, eta0=eta, C1=5, num_MP=mp2, L=5,length_random_walk=length_random_walk,solution=sol) for ii in xrange(iteration_to_mediate))
    #     print 'n=200 k=40: ', time.time() - tt
    #     tt = time.time()
    #     #y5 = Parallel(n_jobs=num_cores)(delayed(main)(n0=2000, k0=100, eta0=eta, C1=5, num_MP=10, L=5, length_random_walk=length_random_walk) for ii in xrange(iteration_to_mediate))
    #     #print 'n=1000 k=500: ', time.time() - tt
    #     print 'Parallel time: ', time.time() - parallel
    #
    #     for i in xrange(iteration_to_mediate-1):
    #         y0[0] += y0[i + 1]
    #         y1[0] += y1[i + 1]
    #         y2[0] += y2[i + 1]
    #         y3[0] += y3[i + 1]
    #
    #     y0 = y0[0] / iteration_to_mediate
    #     y1 = y1[0] / iteration_to_mediate
    #     y2 = y2[0] / iteration_to_mediate
    #     y3 = y3[0] / iteration_to_mediate
    #
    #
    #     # -- Salvataggio su file --
    #     with open('Risultati_txt/Paper1_algo1/plot_Fig3_variazione_Random_Walk='+str(length_random_walk),'wb') as file:
    #          wr=csv.writer(file,quoting=csv.QUOTE_ALL)
    #          wr.writerow(y0)
    #          wr.writerow(y1)
    #          wr.writerow(y2)
    #          wr.writerow(y3)
    #
    #     plt.title('Decoding performances')
    #     x = np.linspace(1, 2.5, 16, endpoint=True)
    #     plt.axis([1, 2.5, 0, 1])
    #     plt.plot(x, y0, label='100 nodes and 10 sources', color='blue', linewidth=1,marker='o',markersize=4.0)
    #     plt.plot(x, y1, label='100 nodes and 20 sources', color='red', linewidth=1,marker='o',markersize=4.0)
    #     plt.plot(x, y2, label='200 nodes and 20 sources', color='grey', linewidth=1,marker='o',markersize=4.0)
    #     plt.plot(x, y3, label='200 nodes and 40 sources', color='magenta', linewidth=1,marker='o',markersize=4.0)
    #     plt.rc('legend', fontsize=10)
    #     plt.legend(loc=4)
    #     plt.grid()
    #     plt.savefig('Immagini/Paper1_algo1/00_Figure3_comparison_LR='+str(length_random_walk)+'c_0'+str(c0)+'delta='+str(delta)+'.pdf', dpi=150, transparent=False)
    #     plt.close()

    #names = ['Figure3Paper1.txt']
    #send_mail(names)


## Comparison different solution for the optimization problem ----------------------------------------------------------
    # print 'Comparison for different solution of the optimization problem'
    #
    # iteration_to_mediate = 8
    # print 'Iteration to mediate ', iteration_to_mediate
    # # punti = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 160, 200, 500, 1000])
    # punti = [50]
    #
    # eta = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]
    # sol = ['ones', 'Larghi_Math', 'Stretti_Math']
    #
    # for length_random_walk in punti:
    #     print 'Lunghezza della random walk:', length_random_walk
    #
    #     y0 = np.zeros((iteration_to_mediate, len(eta)))
    #     y1 = np.zeros((iteration_to_mediate, len(eta)))
    #     y2 = np.zeros((iteration_to_mediate, len(eta)))
    #
    #     mp1 = 1500
    #     mp2 = 2000
    #     mp3 = 2000
    #     n0 = 100
    #     k0 = 20
    #     # mp1 = 1
    #     # mp2 = 1
    #     # mp3 = 1
    #     # -- Iterazione su diversi sistemi --
    #     parallel = time.time()
    #     tt = time.time()
    #     y0 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0, k0=k0, eta0=eta, C1=5, num_MP=mp1, L=5, length_random_walk=length_random_walk,solution=sol[0]) for ii in xrange(iteration_to_mediate))
    #     y1 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0, k0=k0, eta0=eta, C1=5, num_MP=mp1, L=5, length_random_walk=length_random_walk,solution=sol[1]) for ii in xrange(iteration_to_mediate))
    #     y2 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0, k0=k0, eta0=eta, C1=5, num_MP=mp1, L=5, length_random_walk=length_random_walk,solution=sol[2]) for ii in xrange(iteration_to_mediate))
    #     print 'Total time: ', time.time() - parallel
    #
    #     for i in xrange(iteration_to_mediate - 1):
    #         y0[0] += y0[i + 1]
    #         y1[0] += y1[i + 1]
    #         y2[0] += y2[i + 1]
    #
    #     y0 = y0[0] / iteration_to_mediate
    #     y1 = y1[0] / iteration_to_mediate
    #     y2 = y2[0] / iteration_to_mediate
    #
    #
    #     plt.title('Decoding performances - '+str(n0)+' nodes and '+str(k0)+' sources')
    #     plt.xlabel('Decoding ratio $\eta$')
    #     plt.ylabel('Successfull decoding probability P$_s$')
    #     x = np.linspace(1, 2.5, len(y0)-1, endpoint=True)
    #     plt.axis([1, 2.5, 0, 1])
    #     plt.plot(x, y0[1:len(y0)], label=str(sol[0]), color='blue', linewidth=1, marker='o', markersize=4.0)
    #     plt.plot(x, y1[1:len(y0)], label=str(sol[1]), color='red', linewidth=1, marker='o', markersize=4.0)
    #     plt.plot(x, y2[1:len(y0)], label=str(sol[2]), color='green', linewidth=1, marker='o', markersize=4.0)
    #     plt.rc('legend', fontsize=10)
    #     plt.legend(loc=4)
    #     plt.grid()
    #     plt.savefig('Immagini/Paper1_algo1/00_Comparison_OPTsolutions_LR='+str(length_random_walk)+\
    #                 '_c0='+str(c0)+'_delta='+str(delta)+'_n='+str(n0)+'_k='+str(k0)+'.pdf', dpi=150, transparent=False)
    #     plt.close()


## Figura di confronto tra diverse random walk in termini di decoding prob ---------------------------------------------

    print 'Comparison for diffente length of random walk - fixed solution'

    iteration_to_mediate = 8
    print 'Iteration to mediate ', iteration_to_mediate
    eta = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]
    punti = [1,5,19,50,100,500]
    y = np.zeros((len(punti), len(eta)))
    eta = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]
    sol = ['ones', 'Larghi_Math', 'Stretti_Math']

    mp1 = 2000
    mp2 = 2000
    mp3 = 2000
    n0 = 100
    k0 = 10
    # mp1 = 1
    # mp2 = 1
    # mp3 = 1
    cont = -1

    t=time.time()
    for length_random_walk in punti:
        cont += 1
        print 'Lunghezza della random walk:', length_random_walk
        y0 = np.zeros((iteration_to_mediate, len(eta)))

        # -- Iterazione su diversi sistemi --
        tt = time.time()
        y0 = Parallel(n_jobs=num_cores)(
            delayed(main)(n0=n0, k0=k0, eta0=eta, C1=5, num_MP=mp1, L=5, length_random_walk=length_random_walk,solution=sol[1]) for ii in xrange(iteration_to_mediate))
        print 'n='+str(n0)+' k='+str(k0)+': ', time.time() - tt


        for i in xrange(iteration_to_mediate - 1):
            y0[0] += y0[i + 1]

        y0 = y0[0] / iteration_to_mediate

        y[cont][:]=y0
    print 'Total time:', time.time()-t

    plt.title('Decoding performances - Change RW - ' + str(n0) + ' nodes and ' + str(k0) + ' sources')
    plt.xlabel('Decoding ratio $\eta$')
    plt.ylabel('Successfull decoding probability P$_s$')
    x = np.linspace(1, 2.5, len(y0) - 1, endpoint=True)
    plt.axis([1, 2.5, 0, 1])
    for i in xrange(len(punti)):
        plt.plot(x, y[i][1:len(y0)], label=punti[i], linewidth=1, marker='o', markersize=4.0)
    plt.rc('legend', fontsize=10)
    plt.legend(loc=4)
    plt.grid()
    plt.savefig('Immagini/Paper1_algo1/00_Comparison_Length_Random_Walk_'+str(punti)+'_c0=' + str(c0) \
                + '_delta=' + str(delta) + '_n=' + str(n0) + '_k=' + str(k0) + '.pdf', dpi=150,transparent=False)
    plt.close()
