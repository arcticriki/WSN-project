import numpy as np  # import of package numpy for mathematical tools
import random as rnd  # import of package random for homonym tools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # import of package matplotlib.pyplot for plottools
import time as time
from Node import *         #importa la versione due della classe nodo
import cProfile
from RSD import *
from math import factorial
import csv
import copy
from joblib import Parallel, delayed
import multiprocessing
from scipy.stats import norm
from plot_grafo import *
from message_passing import *



c0 = 0.2
delta = 0.05
def main(n0, k0, eta0, C, num_MP, L, length_random_walk, target):
# -- PARAMETER INITIALIZATION SECTION --------------------------------------------------------------
    payload = 1
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
    t= time.time()
    j = 0
    while j < k:
        for i in xrange(n):
            if node_list[i].dim_buffer != 0:
                j += node_list[i].send_pkt(2)
            if j == k:
                break

    if target == 2:
        return time.time()-t

    tot = 0
    distribution_post_dissemination = np.zeros(k + 1)       # ancillary variable used to compute the distribution post dissemination
    for i in xrange(n):
        index = node_list[i].num_encoded                    # retrive the actual encoded degree
        distribution_post_dissemination[index] += 1.0 / n   # augment the prob. value of the related degree
        tot += node_list[i].num_encoded                     # compute the total degree reached


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


    step = 3*int(k/10)
    plt.xlabel('Estimation of n', fontsize=25)
    plt.ylabel('Number of nodes', fontsize=25)
    plt.hist(stima_n, bins=np.arange(mm_n[0],mm_n[1]+step,step=step))
    plt.xlim([mm_n[0]-20, mm_n[1]+20])
    plt.savefig('Immagini/Paper2_algo2/Estimation/00_Figure9_n='+str(n)+'_k='+str(k)+'_c0=' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,transparent=False)
    plt.close()

    step=1
    plt.xlabel('Estimation of k', fontsize=25)
    plt.ylabel('Number of nodes', fontsize=25)
    plt.hist(stima_k,bins=np.arange(mm_k[0],mm_k[1]+step,step=step))
    plt.xlim([mm_k[0]-10, mm_k[1]+10])
    plt.savefig('Immagini/Paper2_algo2/Estimation/00_Figure10_n='+str(n)+'_k='+str(k)+'_c0=' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,transparent=False)
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

        if decoding_performance[iii] > target and target != 0:
            return decoding_performance

    # print 'Time taken by message passing:', time.time()-t
    return decoding_performance


def comparison_network_dimensions(iteration_to_mediate, C2, C3, mp):
    ######    ----------- FIGURE 3 AND 4 -------------------
    print 'Figure 7 and 8 paper 2. \n'
    num_cores = multiprocessing.cpu_count()
    print 'Numero di core utilizzati:', num_cores
    print 'Numero di medie da eseguire: ', iteration_to_mediate

    eta = np.arange(1.0, 2.6, step=0.1)

    y0 = np.zeros((iteration_to_mediate, len(eta)))
    y1 = np.zeros((iteration_to_mediate, len(eta)))
    y2 = np.zeros((iteration_to_mediate, len(eta)))
    y3 = np.zeros((iteration_to_mediate, len(eta)))
    y4 = np.zeros((iteration_to_mediate, len(eta)))
    y5 = np.zeros((iteration_to_mediate, len(eta)))
    C=(0,C2,C3)
    target = 0
    parallel = time.time()
    tt = time.time()
    y0 = Parallel(n_jobs=num_cores)(delayed(main)(n0=100, k0=10, eta0=eta, C=C, num_MP=mp, \
                                                  L=np.sqrt(100 * 9 / 40), length_random_walk=1, target=target) for ii
                                    in xrange(iteration_to_mediate))
    print 'n=100 k=10: ', time.time() - tt
    tt = time.time()
    y1 = Parallel(n_jobs=num_cores)(delayed(main)(n0=100, k0=20, eta0=eta, C=C, num_MP= mp, \
                  L=np.sqrt(100*9/40),length_random_walk=1,target=target) for ii in xrange(iteration_to_mediate))
    print 'n=100 k=20: ', time.time() - tt
    tt = time.time()
    y2 = Parallel(n_jobs=num_cores)(delayed(main)(n0=200, k0=20, eta0=eta, C=C, num_MP= mp, \
                  L=np.sqrt(200*9/40),length_random_walk=1,target=target) for ii in xrange(iteration_to_mediate))
    print 'n=200 k=20: ', time.time() - tt
    tt = time.time()
    y3 = Parallel(n_jobs=num_cores)(delayed(main)(n0=200, k0=40, eta0=eta, C=C, num_MP= mp, \
                  L=np.sqrt(200*9/40),length_random_walk=1,target=target) for ii in xrange(iteration_to_mediate))
    print 'n=200 k=40: ', time.time() - tt
    tt = time.time()
    y4 = Parallel(n_jobs=num_cores)(delayed(main)(n0=500, k0=50, eta0=eta, C=C, num_MP= mp, \
                  L=np.sqrt(500*9/40), length_random_walk=1,target=target) for ii in xrange(iteration_to_mediate))
    print 'n=500 k=50: ', time.time() - tt
    tt = time.time()
    y5 = Parallel(n_jobs=num_cores)(delayed(main)(n0=1000, k0=100, eta0=eta, C=C, num_MP= mp, \
                  L=np.sqrt(1000*9/40), length_random_walk=1,target=target) for ii in xrange(iteration_to_mediate))
    print 'n=1000 k=100: ', time.time() - tt
    print 'Parallel time: ', time.time() - parallel

    y0 = np.sum(y0, 0) / iteration_to_mediate
    y1 = np.sum(y1, 0) / iteration_to_mediate
    y2 = np.sum(y2, 0) / iteration_to_mediate
    y3 = np.sum(y3, 0) / iteration_to_mediate
    y4 = np.sum(y4, 0) / iteration_to_mediate
    y5 = np.sum(y5, 0) / iteration_to_mediate

    ##-- Salvataggio su file --
    with open('Risultati_txt/Paper2_algo2/Figure 7', 'wb') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(y0)
        wr.writerow(y1)
        wr.writerow(y2)
        wr.writerow(y3)

    with open('Risultati_txt/Paper2_algo2/Figure 8','wb') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(y2)
        wr.writerow(y4)
        wr.writerow(y5)

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
    plt.savefig('Immagini/Paper2_algo2/00_Figure7_c_0' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,
                transparent=False)
    plt.close()


    x = np.linspace(1, eta[-1], len(y0), endpoint=True)
    plt.axis([1, eta[-1], 0, 1])
    plt.xlabel('Decoding ratio $\eta$')
    plt.ylabel('Successfull decoding probability P$_s$')
    plt.plot(x, y2, label='200 nodes and 20 sources', color='blue', linewidth=1,marker='o',markersize=4.0)
    plt.plot(x, y4, label='500 nodes and 50 sources', color='red', linewidth=1,marker='o',markersize=4.0)
    plt.plot(x, y5, label='1000 nodes and 100 sources', color='grey', linewidth=1,marker='o',markersize=4.0)
    plt.rc('legend', fontsize=10)
    plt.legend(loc=4)
    plt.grid()
    plt.savefig('Immagini/Paper2_algo2/00_Figure8_c_0' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,transparent=False)
    plt.close()

    x = np.linspace(1, eta[-1], len(y0), endpoint=True)
    plt.axis([1, eta[-1], 0, 1])
    plt.xlabel('Decoding ratio $\eta$')
    plt.ylabel('Successfull decoding probability P$_s$')
    x = np.linspace(1, eta[-1], len(y0), endpoint=True)
    plt.plot(x, y0, label='100 nodes and 10 sources', linewidth=1, marker='o', markersize=4.0)
    plt.plot(x, y1, label='100 nodes and 20 sources', linewidth=1, marker='o', markersize=4.0)
    plt.plot(x, y2, label='200 nodes and 20 sources', linewidth=1, marker='o', markersize=4.0)
    plt.plot(x, y3, label='200 nodes and 40 sources', linewidth=1, marker='o', markersize=4.0)
    plt.plot(x, y4, label='500 nodes and 50 sources', linewidth=1, marker='o', markersize=4.0)
    plt.plot(x, y5, label='1000 nodes and 100 sources', linewidth=1, marker='o', markersize=4.0)
    plt.rc('legend', fontsize=10)
    plt.legend(loc=4)
    plt.grid()
    plt.savefig('Immagini/Paper2_algo2/00_FINAL_c0=' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,
                transparent=False)
    plt.close()


def comparison_C2(iteration_to_mediate, C3, mp):
    # ############### # Figura in cui varia C2   VERSIONE 2 ---------------------------------------------------------
    print 'Figure C_2 comparison. \n'
    num_cores = multiprocessing.cpu_count()
    print 'Numero di core utilizzati:', num_cores
    print 'Numero di medie da eseguire: ', iteration_to_mediate
    target = 0  # Mettere a 0 quando si vuole fare il grafico della decoding prob.
    eta = [[1.5], [2.0], [1.5], [2.0]]

    C_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60]

    totale = time.time()

    n0 = [100, 100, 200, 200]
    k0 = [10, 10, 40, 40]

    y = np.zeros((len(n0),len(C_list) ))
    cont = -1

    for c in xrange(len(C_list)):
        cont += 1
        C = (0, C_list[c], C3)

        y0 = np.zeros((iteration_to_mediate, len(eta)))
        y1 = np.zeros((iteration_to_mediate, len(eta)))
        y2 = np.zeros((iteration_to_mediate, len(eta)))
        y3 = np.zeros((iteration_to_mediate, len(eta)))
        y4 = np.zeros((iteration_to_mediate, len(eta)))
        y5 = np.zeros((iteration_to_mediate, len(eta)))

        parallel = time.time()
        tt = time.time()
        y0 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0[0], k0=k0[0], eta0=eta[0], C=C, num_MP=mp, L=np.sqrt(n0[0] * 9 / 40), \
                           length_random_walk=1,target=target) for ii in xrange(iteration_to_mediate))
        print 'n=100 k=10 eta=1.5: ', time.time() - tt
        y[0, cont] = np.sum(y0, 0) / iteration_to_mediate

        tt = time.time()
        y1 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0[1], k0=k0[1], eta0=eta[1], C=C, num_MP=mp, L=np.sqrt(n0[1] * 9 / 40), \
                        length_random_walk=1,target=target) for ii in xrange(iteration_to_mediate))
        print 'n=100 k=10 eta=2.0: ', time.time() - tt
        y[1, cont] = np.sum(y1, 0) / iteration_to_mediate

        tt = time.time()
        y2 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0[2], k0=k0[2], eta0=eta[2], C=C, num_MP=mp, L=np.sqrt(n0[2] * 9 / 40), \
                        length_random_walk=1,target=target) for ii in xrange(iteration_to_mediate))
        print 'n=200 k=40 eta=1.5: ', time.time() - tt
        y[2, cont] = np.sum(y2, 0) / iteration_to_mediate

        tt = time.time()
        y3 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0[3], k0=k0[3], eta0=eta[3], C=C, num_MP=mp, L=np.sqrt(n0[3] * 9 / 40), \
                         length_random_walk=1, target=target) for ii in xrange(iteration_to_mediate))
        print 'n=200 k=40 eta=2.0: ', time.time() - tt
        y[3, cont] = np.sum(y3, 0) / iteration_to_mediate


        print 'Iteration with C2 =', C[1], ', duration ', time.time() - parallel


    with open('Immagini/Paper2_algo2/C2 variation', 'wb') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        for i in xrange(len(n0)):
            wr.writerow(y[i][:])

    print 'Tempo totale di esecuzione ', time.time() - totale

    plt.xlabel('Parameter $C_2$')
    plt.ylabel('Successfull decoding probability P$_s$')
    plt.axis([C_list[0] - 0.5, C_list[-1] + 0.5, 0, 1])
    for i in xrange(len(n0)):
        plt.plot(C_list, y[i][:], label='n ='+str(n0[i])+' k = '+str(n0[i])+' $\eta$ = '+str(eta[i]) , linewidth=1, marker='o', markersize=4.0)
    plt.rc('legend', fontsize=10)
    plt.legend(loc=4)
    plt.grid()
    plt.savefig('Immagini/Paper2_algo2/00_COMPARISON C2 VALUE_n0=' + str(n0) + '_k0=' + str(k0) + '_c0=' + \
                str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,transparent=False)
    plt.close()


def comparison_C3(iteration_to_mediate, C2, mp):
    ################### # Figura in cui varia C3 --------------------------------------------------------------------------

    num_cores = multiprocessing.cpu_count()
    print 'Numero di core utilizzati:', num_cores
    print 'Numero di medie da eseguire: ', iteration_to_mediate

    eta = 0
    target = 0.8
    if target == 0:
        eta =  np.arange(1.0, 2.6, step=0.1)
    else:
        eta =  np.arange(1.4, 3.6, step=0.1)

    y0 = np.zeros((iteration_to_mediate, len(eta)))
    y1 = np.zeros((iteration_to_mediate, len(eta)))
    y2 = np.zeros((iteration_to_mediate, len(eta)))
    y3 = np.zeros((iteration_to_mediate, len(eta)))
    y4 = np.zeros((iteration_to_mediate, len(eta)))
    y5 = np.zeros((iteration_to_mediate, len(eta)))

    print 'Figure C3 comparison. \n'
    C_list = [10, 15, 20, 30, 40, 50, 60, 80, 100, 140, 200, 350, 500]

    num_c = len(C_list)
    y = np.zeros((num_c, len(eta)))
    y0 = np.zeros((iteration_to_mediate, len(eta)))
    totale = time.time()

    n0 = [100, 100, 200, 200, 500, 1000]

    k0 = [10, 20, 20, 40, 50, 100]

    dec_ratio = np.ones((len(n0), iteration_to_mediate, len(C_list)))

    cont = -1

    for c in xrange(num_c):
        cont += 1
        C = (0, C2, C_list[c])
        parallel = time.time()
        tt = time.time()
        y0 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0[0], k0=k0[0], eta0=eta, C=C, num_MP=mp, L=np.sqrt(n0[0] * 9 / 40),  \
                      length_random_walk=1,target=target) for ii in xrange(iteration_to_mediate))
        print 'n='+str(n0[0])+' k='+str(k0[0])+': ', time.time() - tt
        for i in xrange(iteration_to_mediate):
            ii = 0
            while ii < len(eta) and y0[i][ii]< target:
                dec_ratio[0][i][cont] = eta[ii]
                ii += 1

        tt = time.time()
        y1 = Parallel(n_jobs=num_cores)(delayed(main)(n0=100, k0=20, eta0=eta, C=C, num_MP=mp, L=np.sqrt(100 * 9 / 40), \
                     length_random_walk=1,target=target) for ii in xrange(iteration_to_mediate))
        print 'n=100 k=20: ', time.time() - tt

        for i in xrange(iteration_to_mediate):
            ii = 0
            while ii < len(eta) and y1[i][ii]< target:
                dec_ratio[1][i][cont] = eta[ii]
                ii += 1


        tt = time.time()
        y2 = Parallel(n_jobs=num_cores)(delayed(main)(n0=200, k0=20, eta0=eta, C=C, num_MP=mp, L=np.sqrt(200 * 9 / 40), \
                     length_random_walk=1,target=target) for ii in xrange(iteration_to_mediate))
        print 'n=200 k=20: ', time.time() - tt
        tt = time.time()
        for i in xrange(iteration_to_mediate):
            ii = 0
            while ii < len(eta) and y2[i][ii] < target:
                dec_ratio[2][i][cont] = eta[ii]
                ii += 1

        y3 = Parallel(n_jobs=num_cores)(delayed(main)(n0=200, k0=40, eta0=eta, C=C, num_MP=mp, L=np.sqrt(200 * 9 / 40), \
                     length_random_walk=1,target=target) for ii in xrange(iteration_to_mediate))
        print 'n=200 k=40: ', time.time() - tt
        for i in xrange(iteration_to_mediate):
            ii = 0
            while ii < len(eta) and y3[i][ii] < target:
                dec_ratio[3][i][cont] = eta[ii]
                ii += 1


        y4 = Parallel(n_jobs=num_cores)(delayed(main)(n0=500, k0=50, eta0=eta, C=C, num_MP=mp, L=np.sqrt(500 * 9 / 40), \
                     length_random_walk=1, target=target) for ii in xrange(iteration_to_mediate))
        print 'n=500 k=50: ', time.time() - tt
        for i in xrange(iteration_to_mediate):
            ii = 0
            while ii < len(eta) and y4[i][ii] < target:
                dec_ratio[4][i][cont] = eta[ii]
                ii += 1

        y5 = Parallel(n_jobs=num_cores)(delayed(main)(n0=1000, k0=100, eta0=eta, C=C, num_MP=mp, L=np.sqrt(1000 * 9 / 40), \
                       length_random_walk=1, target=target) for ii in xrange(iteration_to_mediate))
        print 'n=1000 k=100: ', time.time() - tt
        for i in xrange(iteration_to_mediate):
            ii = 0
            while ii < len(eta) and y5[i][ii] < target:
                dec_ratio[5][i][cont] = eta[ii]
                ii += 1
        print 'Tempo totale per c3 = '+str(C_list[c])+': ', time.time() - parallel

    # decommenta quello che ti serve, gli altri lasciali commentati.
    y[c, :] = np.sum(y0, 0) / iteration_to_mediate
    y[c, :] = np.sum(y1, 0) / iteration_to_mediate
    y[c, :] = np.sum(y2, 0) / iteration_to_mediate
    y[c, :] = np.sum(y3, 0) / iteration_to_mediate
    y[c, :] = np.sum(y4, 0) / iteration_to_mediate
    y[c, :] = np.sum(y5, 0) / iteration_to_mediate

    if target == 0:
        print 'Tempo totale di esecuzione ', time.time() - totale
        plt.title('Decoding performances')
        plt.axis([1, 2.5, 0, 1])
        x = np.linspace(1, 2.5, y.shape[1], endpoint=True)
        for i in xrange(num_c):
            plt.plot(x, y[i][:], label='c3=' + str(init * (i + 1)), linewidth=1,marker='o',markersize=4.0)
        plt.rc('legend', fontsize=10)
        plt.legend(loc=4)
        plt.grid()
        plt.savefig('Immagini/Paper2_algo2/00_COMPARISON C3 VALUE_c_0' + str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150,
                    transparent=False)
        plt.close()
    else:
        massimi = np.zeros(len(n0))
        minimi = np.zeros(len(n0))
        medie_dec_ratio = np.zeros((len(n0), len(C_list)))
        for i in xrange(len(n0)):
            medie_dec_ratio[i][:] = sum(dec_ratio[i], 0) / iteration_to_mediate
            massimi[i] = max(medie_dec_ratio[i][:])
            minimi[i] = min(medie_dec_ratio[i][:])


        # -- Salvataggio su file --
        with open('Immagini/Paper2_algo2/C3 variation', 'wb') as file:
            wr = csv.writer(file, quoting=csv.QUOTE_ALL)
            for i in xrange(len(n0)):
                wr.writerow(medie_dec_ratio[i][:])

        plt.xlabel('System parameter C$_3$')
        plt.ylabel('Average decoding ratio $\eta$')
        #plt.axis([0, punti[-1], medie_dec_ratio[3][-1] - 0.2, medie_dec_ratio[0][0] + 0.2])
        plt.axis([C_list[0]-0.5, C_list[-1]+0.5, min(minimi)-0.05, max(massimi)+0.05])
        for i in xrange(len(n0)):
            plt.plot(C_list, medie_dec_ratio[i][:], label='n =' + str(n0[i]) + ' k = ' + str(k0[i]), linewidth=1,
                     marker='o', markersize=4.0)
        plt.rc('legend', fontsize=10)
        plt.legend(loc=1)
        plt.grid()
        plt.savefig('Immagini/Paper2_algo2/C3 comparison_' + str(C_list) + '_c0=' + str(c0) \
                    + '_delta=' + str(delta) + '_n=' + str(n0) + '_k=' + str(k0) + '.pdf', dpi=150, transparent=False)
        plt.close()


def dissemination_cost(iteration_to_mediate, C1,C2,C3):

    ########### Grafico dei tempi ------------------------------------------------------------------------------------
    num_cores = multiprocessing.cpu_count()
    print 'Numero di core utilizzati:', num_cores
    print 'Numero di medie da eseguire: ', iteration_to_mediate

    eta = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5]
    n0 = [100, 200, 500, 1000]
    y0 = np.zeros(iteration_to_mediate)
    y1 = np.zeros(iteration_to_mediate)
    y2 = np.zeros(iteration_to_mediate)
    y3 = np.zeros(iteration_to_mediate)
    y = np.zeros(len(n0))

    mp = 3000

    C = (C1, C2, C3)

    target = 2
    parallel = time.time()
    tt = time.time()
    y0 = Parallel(n_jobs=num_cores)(delayed(main)(n0=100, k0=10, eta0=eta, C=C, num_MP= mp , \
                  L=np.sqrt(100*9/40),length_random_walk=1,target=target) for ii in xrange(iteration_to_mediate))
    y[0] = np.sum(y0, 0) / iteration_to_mediate

    print 'n=100 k=10: ', time.time() - tt

    tt = time.time()
    y1 = Parallel(n_jobs=num_cores)(delayed(main)(n0=200, k0=40, eta0=eta, C=C, num_MP= mp, \
                  L=np.sqrt(200*9/40),length_random_walk=1,target=target) for ii in xrange(iteration_to_mediate))
    y[1] = np.sum(y1, 0) / iteration_to_mediate
    print 'n=200 k=40: ', time.time() - tt

    tt = time.time()
    y2 = Parallel(n_jobs=num_cores)(delayed(main)(n0=500, k0=50, eta0=eta, C=C, num_MP= mp, \
                  L=np.sqrt(500*9/40), length_random_walk=1,target=target) for ii in xrange(iteration_to_mediate))
    y[2] = np.sum(y2, 0) / iteration_to_mediate
    print 'n=500 k=50: ', time.time() - tt
    tt = time.time()
    y3 = Parallel(n_jobs=num_cores)(delayed(main)(n0=1000, k0=100, eta0=eta, C=C, num_MP=mp, \
                  L=np.sqrt(1000 * 9 / 40), length_random_walk=1, target=target) for ii in xrange(iteration_to_mediate))
    y[3] = np.sum(y3, 0) / iteration_to_mediate
    print 'n=1000 k=100: ', time.time() - tt
    print 'Parallel time: ', time.time() - parallel


    # -- Salvataggio su file --
    with open('Risultati_txt/Tempi disseminazione Paper2 algo2 ', 'wb') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(y)

if __name__ == "__main__":

    comparison_network_dimensions(iteration_to_mediate=12, C2=40, C3=400 , mp=3000)
    # Comparison for several network dimensions (n0,k0)=[(100,10);(100,20);(200,20);(200,40);(500,50);(1000,100)]
    # This function produces three separate figures, one comparing first 4 couples n,k, the second considering
    # (200,40);(500,50);(1000,100). It reproduces Fig 7 and 8 of paper 2. The last figure compares alla the network dimensions
    # it is also present in the final report.
    # Moreover it generates a text file containing the vectors of the plots.
    # Graphs can be found at: ../Immagini/Paper2_algo2/00_Figure7_c_0.....
    #                         ../Immagini/Paper2_algo2/00_Figure8_c_0.....
    #                         ../Immagini/Paper2_algo2/00_FINAL_c0=......
    # Text file can be found at: ../Risultati_txt/Paper2_algo2/Figure 7
    #                            ../Risultati_txt/Paper2_algo2/Figure 8


    comparison_C2(iteration_to_mediate=200, C3=400, mp=3000)
    # Comparison for several values of system parameter C2. It generates a figure and a file containing the results.
    # It generates the same results as figure 6 of paper 2.
    # Graphs can be found at: ../Immagini/Paper2_algo2/00_COMPARISON C2 VALUE_n0=........
    # Text file can be found at: ../Immagini/Paper2_algo2/C2 variation
    # NB: This function is heavy time consuming.

    comparison_C3(iteration_to_mediate=12, C2=40,mp=3000)
    # Comparison for several values of system parameter C3. It generates a figure and a file containing the results.
    # It generates the same results as figure 6 of paper 2.
    # Graphs can be found at: ../Immagini/Paper2_algo2/C3 comparison_'.........
    # Text file can be found at: ../Immagini/Paper2_algo2/C3 variation'
    # NB: This function is heavy time consuming.

    dissemination_cost(iteration_to_mediate=12, C2=40, C3=400)
    # Compute the dissemination time, in order to asses the complexity. Presented in the overall comparison in the report of the project
