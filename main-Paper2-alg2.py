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
from scipy.stats import norm
from plot_grafo import *

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





c0 = 0.2
delta = 0.05
def main(n0, k0, eta0, C, num_MP,L,length_random_walk):
# -- PARAMETER INITIALIZATION SECTION --------------------------------------------------------------
    payload = 10
    C1 = C[0]
    C2 = C[1]
    C3 = C[2]
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

    d, pdf , _ = Robust_Soliton_Distribution(n, k, c0, delta)  # See RSD doc
    # to_be_encoded = np.sum(d)                        #use to check how many pkts we should encode ***OPEN PROBLEM***

    # -- NETWORK INITIALIZATION --
    # Generation of storage nodes
    for i in xrange(n):  # for on 0 to n indices
        x = rnd.uniform(0.0, L)             # generation of random coordinate x
        y = rnd.uniform(0.0, L)             # generation of random coordinate y
        node_list.append(Storage(i + 1, x, y, 0, n, k, C1,C2, C3,0 , 0, c0, delta))  # creation of Storage node
        positions[i, :] = [x, y]

    # Generation of sensor nodes
    for i in sensors_indexes:               # for on sensors position indices
        x = rnd.uniform(0.0, L)             # generation of random coordinate x
        y = rnd.uniform(0.0, L)             # generation of random coordinate y
        node_list[i] = Sensor(i + 1, x, y, 0, n,k, C1,C2, C3, 0, 0, c0, delta)  # creation of sensor node, function Sensor(), extend Storage class
        positions[i, :] = [x, y]  # support variable for positions info, used for comp. optim. reasons

    # t = time.time()
    # Find nearest neighbours using euclidean distance

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


    # Compute the network topology
    #plot_grafo(node_list, n, k, sensors_indexes,L)


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
                j += node_list[i].send_pkt(2)
            if j == k:
                break

    tot = 0
    distribution_post_dissemination = np.zeros(k + 1)       # ancillary variable used to compute the distribution post dissemination
    for i in xrange(n):
        index = node_list[i].num_encoded                    # retrive the actual encoded degree
        distribution_post_dissemination[index] += 1.0 / n   # augment the prob. value of the related degree
        tot += node_list[i].num_encoded                     # compute the total degree reached


    # return  distribution_post_dissemination[1:] , pdf
    # plt.title('Post dissemination')
    # y = distribution_post_dissemination[1:]
    # x = np.linspace(1, k, k, endpoint=True)
    # plt.axis([0, k, 0, 0.6])
    # plt.plot(x,y , label='post dissemination')              # plot the robust pdf vs the obtained distribution after dissemination
    # y2 = np.zeros(k)
    # y2[:len(pdf)] = pdf
    # plt.plot(x, y2, color='red', label='robust soliton')
    # plt.legend(loc='upper left')
    # plt.grid()
    # plt.show()


    #[node_list[i].baro() for i in xrange(n)]


    stima_n = np.zeros(n)
    stima_k = np.zeros(n)
    mm_n = np.zeros(2)          # min - max for n
    mm_k = np.zeros(2)          # min - max for n
    mm_n[0] = n
    mm_k[0] = k

    for i in xrange(n):

        stima_n[i] = node_list[i].n_stimato_hop

        if stima_n[i] > mm_n[1]:
            mm_n[1] = stima_n[i]
        elif stima_n[i] < mm_n[0] and stima_n[i]>0:
            mm_n[0] = stima_n[i]

        stima_k[i] = node_list[i].k_stimato_hop

        if stima_k[i] > mm_k[1]:
            mm_k[1] = stima_k[i]
        elif stima_k[i] < mm_k[0] and stima_k[i]>0:
            mm_k[0] = stima_k[i]

    #print np.sort(stima_n)

    step = 3*int(k/10)
    plt.hist(stima_n, bins=np.arange(mm_n[0],mm_n[1]+step,step=step) , label='Estimated $\hat n$ frequency', color='blue', linewidth=2)
    plt.xlim([mm_n[0]-20, mm_n[1]+20])
    plt.legend(loc=1)
    plt.savefig('Immagini/Paper2_algo2/00_Figure9_n='+str(n)+'_k='+str(k)+'_c0=' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,transparent=False)
    plt.close()

    step=1
    plt.hist(stima_k,bins=np.arange(mm_k[0],mm_k[1]+step,step=step), label='Estimated $\hat k$ frequency', color='blue', linewidth=2)
    plt.xlim([mm_k[0]-10, mm_k[1]+10])
    plt.legend(loc=1)
    plt.savefig('Immagini/Paper2_algo2/00_Figure10_n='+str(n)+'_k='+str(k)+'_c0=' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,transparent=False)
    plt.close()




    # -- DECODING PHASE --------
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

    # iteration_to_mediate = 12
    # print 'Numero di medie da eseguire: ', iteration_to_mediate
    #
    # eta = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]
    #
    # # ----------- FIGURE 3 AND 4 -------------------
    # print 'Figure 3 and 4. \n'
    #
    # y0 = np.zeros((iteration_to_mediate, len(eta)))
    # y1 = np.zeros((iteration_to_mediate, len(eta)))
    # y2 = np.zeros((iteration_to_mediate, len(eta)))
    # y3 = np.zeros((iteration_to_mediate, len(eta)))
    # # y4 = np.zeros((iteration_to_mediate, len(eta)))
    # # y5 = np.zeros((iteration_to_mediate, len(eta)))
    #
    # mp1=1500
    # mp2=1000
    # mp3=1000
    # C=(5,20,100)
    # # mp1=1
    # # mp2=1
    # # mp3=1
    #
    # parallel = time.time()
    # tt = time.time()
    # y0 = Parallel(n_jobs=num_cores)(delayed(main)(n0=100, k0=10, eta0=eta, C=C, num_MP= mp1 , L=np.sqrt(100*9/40),length_random_walk=1) for ii in xrange(iteration_to_mediate))
    # print 'n=100 k=10: ', time.time() - tt
    # tt = time.time()
    # y1 = Parallel(n_jobs=num_cores)(delayed(main)(n0=100, k0=20, eta0=eta, C=C, num_MP= mp1, L=np.sqrt(100*9/40),length_random_walk=1) for ii in xrange(iteration_to_mediate))
    # print 'n=100 k=20: ', time.time() - tt
    # tt = time.time()
    # y2 = Parallel(n_jobs=num_cores)(delayed(main)(n0=200, k0=20, eta0=eta, C=C, num_MP= mp1, L=np.sqrt(200*9/40),length_random_walk=1) for ii in xrange(iteration_to_mediate))
    # print 'n=200 k=20: ', time.time() - tt
    # tt = time.time()
    # y3 = Parallel(n_jobs=num_cores)(delayed(main)(n0=200, k0=40, eta0=eta, C=C, num_MP= mp2, L=np.sqrt(200*9/40),length_random_walk=1) for ii in xrange(iteration_to_mediate))
    # print 'n=200 k=40: ', time.time() - tt
    # # tt = time.time()
    # # y4 = Parallel(n_jobs=num_cores)(delayed(main)(n0=500, k0=50, eta0=eta, C=C, num_MP= mp2, L=np.sqrt(500*9/40), length_random_walk=1) for ii in xrange(iteration_to_mediate))
    # # print 'n=500 k=50: ', time.time() - tt
    # # tt = time.time()
    # # y5 = Parallel(n_jobs=num_cores)(delayed(main)(n0=1000, k0=100, eta0=eta, C=C, num_MP= mp3, L=np.sqrt(1000*9/40), length_random_walk=1) for ii in xrange(iteration_to_mediate))
    # # print 'n=1000 k=100: ', time.time() - tt
    # print 'Parallel time: ', time.time() - parallel
    #
    # for i in xrange(iteration_to_mediate - 1):
    #     y0[0] += y0[i + 1]
    #     y1[0] += y1[i + 1]
    #     y2[0] += y2[i + 1]
    #     y3[0] += y3[i + 1]
    #     # y4[0] += y4[i + 1]
    #     # y5[0] += y5[i + 1]
    #
    # y0 = y0[0] / iteration_to_mediate
    # y1 = y1[0] / iteration_to_mediate
    # y2 = y2[0] / iteration_to_mediate
    # y3 = y3[0] / iteration_to_mediate
    # # y4 = y4[0] / iteration_to_mediate
    # # y5 = y5[0] / iteration_to_mediate
    #
    # # -- Salvataggio su file --
    # with open('Risultati_txt/Paper2_algo2/Figure 3', 'wb') as file:
    #     wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    #     wr.writerow(y0)
    #     wr.writerow(y1)
    #     wr.writerow(y2)
    #     wr.writerow(y3)
    #
    # # with open('Risultati_txt/Paper2_algo2/Figure 4','wb') as file:
    # #     wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    # #     wr.writerow(y2)
    # #     wr.writerow(y4)
    # #     wr.writerow(y5)
    #
    # plt.title('Decoding performances')
    # plt.axis([1, 2.5, 0, 1])
    # x = np.linspace(1, 2.5, len(y0)-1, endpoint=True)
    # plt.plot(x, y0[0:len(y0)-1], label='100 nodes and 10 sources', color='blue', linewidth=2)
    # plt.plot(x, y1[0:len(y1)-1], label='100 nodes and 20 sources', color='red', linewidth=2)
    # plt.plot(x, y2[0:len(y2)-1], label='200 nodes and 20 sources', color='grey', linewidth=2)
    # plt.plot(x, y3[0:len(y3)-1], label='200 nodes and 40 sources', color='magenta', linewidth=2)
    # plt.legend(loc=4)
    # plt.grid()
    # plt.savefig('Immagini/Paper2_algo2/00_Figure3_c_0' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150, transparent=False)
    # plt.close()
    #
    # plt.title('Decoding performances- BARO 1')
    # plt.axis([1, 2.5, 0, 1])
    # plt.plot(x, y0[1:len(y0)], label='100 nodes and 10 sources', color='blue', linewidth=2)
    # plt.plot(x, y1[1:len(y1)], label='100 nodes and 20 sources', color='red', linewidth=2)
    # plt.plot(x, y2[1:len(y2)], label='200 nodes and 20 sources', color='grey', linewidth=2)
    # plt.plot(x, y3[1:len(y3)], label='200 nodes and 40 sources', color='magenta', linewidth=2)
    # plt.legend(loc=4)
    # plt.grid()
    # plt.savefig('Immagini/Paper2_algo2/00_Figure3_BARO_c_0' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,
    #             transparent=False)
    # plt.close()


    # plt.title('Decoding performances')
    # plt.axis([1, 2.5, 0, 1])
    # x = np.linspace(1, 2.5, len(y2), endpoint=True)
    # plt.plot(x, y2, label='200 nodes and 20 sources', color='blue', linewidth=2)
    # x = np.linspace(1, 2.5, len(y4), endpoint=True)
    # plt.plot(x, y4, label='500 nodes and 50 sources', color='red', linewidth=2)
    # x = np.linspace(1, 2.5, len(y5), endpoint=True)
    # plt.plot(x, y5, label='1000 nodes and 100 sources', color='grey', linewidth=2)
    # plt.legend(loc=4)
    # plt.grid()
    # plt.savefig('Immagini/Paper2_algo2/00_Figure4_c_0' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,transparent=False)
    # plt.close()



    # ----------- FIGURE 6 -------------------

    # print 'Figure 6. \n'
    #
    # iteration_to_mediate = 64
    #
    # C_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    # number_of_points_in_x_axis = len(C_list)
    # y8 = np.zeros(number_of_points_in_x_axis)
    # y9 = np.zeros(number_of_points_in_x_axis)
    #
    # parallel = time.time()
    # for i in xrange(number_of_points_in_x_axis):
    #     tempo = time.time()
    #
    #     appoggio1 = Parallel(n_jobs=num_cores)(delayed(main)(n0=500, k0=50, eta0=[1.8], C1=C_list[i], num_MP=500, L=1,length_random_walk=1) for ii in xrange(iteration_to_mediate))
    #     appoggio2 = Parallel(n_jobs=num_cores)(delayed(main)(n0=1000, k0=100, eta0=[1.6], C1=C_list[i], num_MP=500, L=1, length_random_walk=1) for ii in xrange(iteration_to_mediate))
    #
    #     y8[i] = np.sum(appoggio1) / iteration_to_mediate
    #     y9[i] = np.sum(appoggio2) / iteration_to_mediate
    #     print 'Tempo di esecuzione con C=',C_list[i],' t=',time.time()-tempo
    #
    # print 'Parallel time: ', time.time() - parallel
    #
    #
    # # -- Salvataggio su file --
    # with open('Risultati_txt/Paper2_algo2/Figure 6','wb') as file:
    #     wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    #     wr.writerow(y8)
    #     wr.writerow(y9)
    #
    #
    # plt.title('Decoding performances')
    # x = C_list
    #
    # plt.axis([0, 5, 0.5, 1])
    # plt.plot(x, y8, label='500 nodes and 50 souces', color='blue', linewidth=2)
    # plt.plot(x, y9, label='1000 nodes and 100 souces', color='red', linewidth=2)
    # plt.legend(loc=4)
    # plt.grid()
    # plt.savefig('Immagini/Paper2_algo2/00_Figure6_c_0' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,transparent=False)
    # plt.close()
    #


    # ----------- FIGURE 5 -------------------
    # print '\n\nFigure 5. \n'
    #
    # iteration_to_mediate = 8
    # number_of_points_in_x_axis = 10
    # y6 = np.zeros( number_of_points_in_x_axis)
    # y7 = np.zeros( number_of_points_in_x_axis)
    #
    #
    #
    # parallel = time.time()
    # for ii in xrange(number_of_points_in_x_axis):
    #     tempo = time.time()
    #
    #     appoggio1 = Parallel(n_jobs=num_cores)(delayed(main)(n0=500 * (ii + 1), k0=50 * (ii + 1), \
    #                     eta0=[1.4], C1=3, num_MP=1000, L=5,length_random_walk=1) for ii in xrange(iteration_to_mediate))
    #     appoggio2 = Parallel(n_jobs=num_cores)(delayed(main)(n0=500 * (ii + 1), k0=50 * (ii + 1), \
    #                     eta0=[1.7], C1=3, num_MP=1000, L=5, length_random_walk=1) for ii in xrange(iteration_to_mediate))
    #
    #     y6[ii] = np.sum(appoggio1) / iteration_to_mediate
    #     y7[ii] = np.sum(appoggio2) / iteration_to_mediate
    #     print 'Tempo di esecuzione con n=',500 * (ii + 1),'e k=',50 * (ii + 1),' t=',time.time()-tempo
    #
    # print 'Parallel time: ', time.time() - parallel
    #
    # # -- Salvataggio su file --
    # with open('Risultati_txt/Paper2_algo2/Figure 5', 'wb') as file:
    #     wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    #     wr.writerow(y6)
    #     wr.writerow(y7)
    #
    #
    # plt.title('Decoding performances')
    # x = np.linspace(500, 5000, number_of_points_in_x_axis, endpoint=True)
    # plt.axis([500, 5000, 0, 1])
    # plt.plot(x, y6, label='eta 1.4', color='blue', linewidth=2)
    # plt.plot(x, y7, label='eta 1.7', color='red', linewidth=2)
    # plt.legend(loc=4)
    # plt.grid()
    # plt.savefig('Immagini/Paper2_algo2/00_Figure5_c_0' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,transparent=False)
    # plt.close()

    #names = ['Figure 3.txt','Figure 4.txt','Figure 5.txt.txt','Figure 6.txt']
    #send_mail(names)








    # # Figura in cui varia C2
    # iteration_to_mediate = 24
    # print 'Numero di medie da eseguire: ', iteration_to_mediate
    #
    # eta = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]
    #
    # #----------- FIGURE 3 AND 4 -------------------
    # print 'Figure C comparison. \n'
    #
    # num_c = 5
    # y = np.zeros((num_c, len(eta)))
    # y0 = np.zeros((iteration_to_mediate, len(eta)))
    # totale = time.time()
    #
    # init= 50
    # for c in xrange(num_c):
    #
    #     mp1 = 1500
    #     mp2 = 1000
    #     mp3 = 1000
    #     C = (5, init*(c+1), 50)
    #     print 'Sto eseguendo il conto con c2 =', C[1]
    #     # mp1=1
    #     # mp2=1
    #     # mp3=1
    #
    #     #parallel = time.time()
    #     tt = time.time()
    #     y0 = Parallel(n_jobs=num_cores)(delayed(main)(n0=100, k0=10, eta0=eta, C=C, num_MP=mp1, L=np.sqrt(100 * 9 / 40), length_random_walk=1) for ii in xrange(iteration_to_mediate))
    #     print 'n=100 k=10: ', time.time() - tt
    #     # tt = time.time()
    #     # y1 = Parallel(n_jobs=num_cores)(
    #     #     delayed(main)(n0=100, k0=20, eta0=eta, C=C, num_MP=mp1, L=np.sqrt(100 * 9 / 40), length_random_walk=1) for ii in
    #     #     xrange(iteration_to_mediate))
    #     # print 'n=100 k=20: ', time.time() - tt
    #     # tt = time.time()
    #     # y2 = Parallel(n_jobs=num_cores)(
    #     #     delayed(main)(n0=200, k0=20, eta0=eta, C=C, num_MP=mp1, L=np.sqrt(200 * 9 / 40), length_random_walk=1) for ii in
    #     #     xrange(iteration_to_mediate))
    #     # print 'n=200 k=20: ', time.time() - tt
    #     # tt = time.time()
    #     # y3 = Parallel(n_jobs=num_cores)(
    #     #     delayed(main)(n0=200, k0=40, eta0=eta, C=C, num_MP=mp2, L=np.sqrt(200 * 9 / 40), length_random_walk=1) for ii in
    #     #     xrange(iteration_to_mediate))
    #     # print 'n=200 k=40: ', time.time() - tt
    #     # tt = time.time()
    #     # y4 = Parallel(n_jobs=num_cores)(delayed(main)(n0=500, k0=50, eta0=eta, C=C, num_MP= mp2, L=np.sqrt(500*9/40), length_random_walk=1) for ii in xrange(iteration_to_mediate))
    #     # print 'n=500 k=50: ', time.time() - tt
    #     # tt = time.time()
    #     # y5 = Parallel(n_jobs=num_cores)(delayed(main)(n0=1000, k0=100, eta0=eta, C=C, num_MP= mp3, L=np.sqrt(1000*9/40), length_random_walk=1) for ii in xrange(iteration_to_mediate))
    #     # print 'n=1000 k=100: ', time.time() - tt
    #     #print 'Parallel time: ', time.time() - parallel
    #
    #     for i in xrange(iteration_to_mediate - 1):
    #         y0[0] += y0[i + 1]
    #
    #
    #     y[c,:] = y0[0] / iteration_to_mediate
    #
    # print 'Tempo totale di esecuzione ', time.time()-totale
    # plt.title('Decoding performances')
    # plt.axis([1, 2.5, 0, 1])
    # x = np.linspace(1, 2.5, y.shape[1]-1, endpoint=True)
    # for i in xrange(num_c):
    #     plt.plot(x, y[i][0:y.shape[1]-1], label='c2='+str(init*(i+1)), linewidth=2)
    # plt.legend(loc=4)
    # plt.grid()
    # plt.savefig('Immagini/Paper2_algo2/00_COMPARISON C2 VALUE_c_0' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,
    #             transparent=False)
    # plt.close()
    #
    # plt.title('Decoding performances- BARO 1')
    # plt.axis([1, 2.5, 0, 1])
    # for i in xrange(num_c):
    #     plt.plot(x, y[i][1:y.shape[1]], label='c2='+str(init*(i+1)), linewidth=2)
    #
    # plt.legend(loc=4)
    # plt.grid()
    # plt.savefig('Immagini/Paper2_algo2/00_COMPARISON C2 VALUE BARO_c_0' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,
    #             transparent=False)
    # plt.close()











    # # Figura in cui varia C3
    # iteration_to_mediate = 12
    # print 'Numero di medie da eseguire: ', iteration_to_mediate
    #
    # eta = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]
    #
    # # ----------- FIGURE 3 AND 4 -------------------
    # print 'Figure C3 comparison. \n'
    #
    # num_c = 5
    # y = np.zeros((num_c, len(eta)))
    # y0 = np.zeros((iteration_to_mediate, len(eta)))
    # totale = time.time()
    # init=50
    # for c in xrange(num_c):
    #
    #     mp1 = 1500
    #     mp2 = 1000
    #     mp3 = 1000
    #     C = (5, 50, init * (c + 1))
    #     print 'Sto eseguendo il conto con c3 =', C[2]
    #     # mp1=1
    #     # mp2=1
    #     # mp3=1
    #
    #     # parallel = time.time()
    #     tt = time.time()
    #     y0 = Parallel(n_jobs=num_cores)(
    #         delayed(main)(n0=100, k0=10, eta0=eta, C=C, num_MP=mp1, L=np.sqrt(100 * 9 / 40), length_random_walk=1) for
    #         ii in xrange(iteration_to_mediate))
    #     print 'n=100 k=10: ', time.time() - tt
    #     # tt = time.time()
    #     # y1 = Parallel(n_jobs=num_cores)(
    #     #     delayed(main)(n0=100, k0=20, eta0=eta, C=C, num_MP=mp1, L=np.sqrt(100 * 9 / 40), length_random_walk=1) for ii in
    #     #     xrange(iteration_to_mediate))
    #     # print 'n=100 k=20: ', time.time() - tt
    #     # tt = time.time()
    #     # y2 = Parallel(n_jobs=num_cores)(
    #     #     delayed(main)(n0=200, k0=20, eta0=eta, C=C, num_MP=mp1, L=np.sqrt(200 * 9 / 40), length_random_walk=1) for ii in
    #     #     xrange(iteration_to_mediate))
    #     # print 'n=200 k=20: ', time.time() - tt
    #     # tt = time.time()
    #     # y3 = Parallel(n_jobs=num_cores)(
    #     #     delayed(main)(n0=200, k0=40, eta0=eta, C=C, num_MP=mp2, L=np.sqrt(200 * 9 / 40), length_random_walk=1) for ii in
    #     #     xrange(iteration_to_mediate))
    #     # print 'n=200 k=40: ', time.time() - tt
    #     # tt = time.time()
    #     # y4 = Parallel(n_jobs=num_cores)(delayed(main)(n0=500, k0=50, eta0=eta, C=C, num_MP= mp2, L=np.sqrt(500*9/40), length_random_walk=1) for ii in xrange(iteration_to_mediate))
    #     # print 'n=500 k=50: ', time.time() - tt
    #     # tt = time.time()
    #     # y5 = Parallel(n_jobs=num_cores)(delayed(main)(n0=1000, k0=100, eta0=eta, C=C, num_MP= mp3, L=np.sqrt(1000*9/40), length_random_walk=1) for ii in xrange(iteration_to_mediate))
    #     # print 'n=1000 k=100: ', time.time() - tt
    #     # print 'Parallel time: ', time.time() - parallel
    #
    #     for i in xrange(iteration_to_mediate - 1):
    #         y0[0] += y0[i + 1]
    #
    #     y[c, :] = y0[0] / iteration_to_mediate
    #
    # print 'Tempo totale di esecuzione ', time.time() - totale
    # plt.title('Decoding performances')
    # plt.axis([1, 2.5, 0, 1])
    # x = np.linspace(1, 2.5, y.shape[1] - 1, endpoint=True)
    # for i in xrange(num_c):
    #     plt.plot(x, y[i][0:y.shape[1] - 1], label='c3=' + str(init * (i + 1)), linewidth=2)
    # plt.legend(loc=4)
    # plt.grid()
    # plt.savefig('Immagini/Paper2_algo2/00_COMPARISON C3 VALUE_c_0' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,
    #             transparent=False)
    # plt.close()
    #
    # plt.title('Decoding performances- BARO 1')
    # plt.axis([1, 2.5, 0, 1])
    # for i in xrange(num_c):
    #     plt.plot(x, y[i][1:y.shape[1]], label='c3=' + str(init * (i + 1)), linewidth=2)
    #
    # plt.legend(loc=4)
    # plt.grid()
    # plt.savefig('Immagini/Paper2_algo2/00_COMPARISON C3 VALUE BARO_c_0' + str(c0) + 'delta=' + str(delta) + '.pdf',
    #             dpi=150,
    #             transparent=False)
    # plt.close()





    # Figura 11 paper 2
    iteration_to_mediate = 12
    print 'Numero di medie da eseguire: ', iteration_to_mediate

    eta = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]

    # ----------- FIGURE 3 AND 4 -------------------
    print 'Figure 11. \n'


    C2 = range(15,61,5)
    num_c = len(C2)
    y = np.zeros((4, num_c))
    y0 = np.zeros(iteration_to_mediate)
    y1 = np.zeros(iteration_to_mediate)
    y2 = np.zeros(iteration_to_mediate)
    y3 = np.zeros(iteration_to_mediate)
    totale = time.time()
    counter=-1

    mp1 = 1500
    mp2 = 1000
    mp3 = 1000

    # mp1 = 1
    # mp2 = 1
    # mp3 = 1

    for c in C2:
        counter += 1

        C = (5, c, 100)
        print 'Sto eseguendo il conto con c2 =', C[1]
        # parallel = time.time()
        tt = time.time()
        y0 = Parallel(n_jobs=num_cores)(
            delayed(main)(n0=100, k0=10, eta0=[1.5], C=C, num_MP=mp1, L=np.sqrt(100 * 9 / 40), length_random_walk=1) for
            ii in xrange(iteration_to_mediate))
        y1 = Parallel(n_jobs=num_cores)(
            delayed(main)(n0=100, k0=10, eta0=[2.0], C=C, num_MP=mp1, L=np.sqrt(100 * 9 / 40), length_random_walk=1) for
            ii in xrange(iteration_to_mediate))
        y2 = Parallel(n_jobs=num_cores)(
            delayed(main)(n0=200, k0=20, eta0=[1.5], C=C, num_MP=mp1, L=np.sqrt(100 * 9 / 40), length_random_walk=1) for
            ii in xrange(iteration_to_mediate))
        y3 = Parallel(n_jobs=num_cores)(
            delayed(main)(n0=200, k0=20, eta0=[2.0], C=C, num_MP=mp1, L=np.sqrt(100 * 9 / 40), length_random_walk=1) for
            ii in xrange(iteration_to_mediate))
        print 'n=100 k=10: ', time.time() - tt


        for i in xrange(iteration_to_mediate - 1):
            y0[0] += y0[i + 1]
            y1[0] += y1[i + 1]
            y2[0] += y2[i + 1]
            y3[0] += y3[i + 1]

        y[0, counter] = y0[0] / iteration_to_mediate
        y[1, counter] = y1[0] / iteration_to_mediate
        y[2, counter] = y2[0] / iteration_to_mediate
        y[3, counter] = y3[0] / iteration_to_mediate

    print 'Tempo totale di esecuzione ', time.time() - totale
    plt.title('Decoding performances')
    plt.xlabel('System parameter C$_2$')
    plt.ylabel('Successfull decoding probability P$_s$')
    plt.axis([0, C2[-1], 0, 1])
    x = np.linspace(C2[0], C2[-1], y.shape[1] - 1, endpoint=True)
    for i in xrange(y.shape[0]):
        plt.plot(x, y[i][0:y.shape[1] - 1], label='c3=' + str(c), linewidth=2)
    plt.legend(loc=4)
    plt.grid()
    plt.savefig('Immagini/Paper2_algo2/00_Figura11_c_0' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,
                transparent=False)
    plt.close()

    plt.title('Decoding performances- BARO 1')
    plt.axis([0, C2[-1], 0, 1])
    plt.xlabel('System parameter c2')
    plt.ylabel('Successfull decoding probability P$_s$')
    for i in xrange(y.shape[0]):
        plt.plot(x, y[i][1:y.shape[1]], label='c3=' + str(c), linewidth=2)

    plt.legend(loc=4)
    plt.grid()
    plt.savefig('Immagini/Paper2_algo2/00_Figura11 BARO_c_0' + str(c0) + 'delta=' + str(delta) + '.pdf',
                dpi=150,
                transparent=False)
    plt.close()


    # #cProfile.run('main(n0=200, k0=40)')
