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
from message_passing import *
#from plot_grafo import *

c0 = 0.2
delta = 0.05

def main(n0, k0, eta0, C1, num_MP,L,length_random_walk,solution,target):
# -- PARAMETER INITIALIZATION SECTION --------------------------------------------------------------
    payload = 1
    C1 = C1
    C2 = 0
    C3= 0
    eta = eta0
    n = n0                                   # number of nodes
    k = k0                                   # number of sensors

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

    if solution == 'Larghi_Math':
        with open('Dati/OptProblem1/'+solution+'/'+str(k)+'_'+str(c0)+'_'+str(delta)+'_L_KR.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)  # , quotechar='|')
            for row in reader:
                Xd = row
        for i in xrange(len(Xd)):
            Xd[i] = float(Xd[i])

    if solution == 'Stretti_Math':
        with open('Dati/OptProblem1/'+solution+'/'+str(k)+'_'+str(c0)+'_'+str(delta)+'_S_KR.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)  # , quotechar='|')
            for row in reader:
                Xd = row
        for i in xrange(len(Xd)):
            Xd[i] = float(Xd[i])

    if solution == 'Annealing':
        with open('Dati/OptProblem1/'+solution+'/N'+str(n)+'K'+str(k)+'.txt', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)  # , quotechar='|')
            for row in reader:
                Xd = row
        for i in xrange(len(Xd)):
            Xd[i] = float(Xd[i])


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

    if target==2:
        return time.time()-t

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

        if decoding_performance[iii] > target and target != 0:
            return  decoding_performance


    #print 'Time taken by message passing:', time.time()-t

    return decoding_performance


def comparison_network_dimensions(iteration_to_mediate,L_RW,mp):
    ## Decoding pribability of several network dimensions -----------------------------------------------------------------
    num_cores = multiprocessing.cpu_count()
    print 'Numero di core utilizzati:', num_cores
    print 'Numero di medie da eseguire: ', iteration_to_mediate
    punti = [L_RW]
    eta = np.arange(1.0, 2.6, step=0.1)

    for length_random_walk in punti:
        print 'Lunghezza della random walk:', length_random_walk
        # sol = 'ones'
        sol = 'Larghi_Math'
        # sol = 'Stretti_Math'
        y0 = np.zeros((iteration_to_mediate, len(eta)))
        y1 = np.zeros((iteration_to_mediate, len(eta)))
        y2 = np.zeros((iteration_to_mediate, len(eta)))
        y3 = np.zeros((iteration_to_mediate, len(eta)))
        y4 = np.zeros((iteration_to_mediate, len(eta)))
        y5 = np.zeros((iteration_to_mediate, len(eta)))

        # -- Iterazione su diversi sistemi --
        parallel = time.time()
        tt = time.time()
        y0 = Parallel(n_jobs=num_cores)(delayed(main)(n0=100, k0=10, eta0=eta, C1=5, num_MP=mp, L=np.sqrt(100 * 9 / 40), \
              length_random_walk=length_random_walk,solution=sol,target = 0) for ii in xrange(iteration_to_mediate))
        print 'n=100 k=10: ', time.time() - tt
        tt = time.time()
        y1 = Parallel(n_jobs=num_cores)(delayed(main)(n0=100, k0=20, eta0=eta, C1=5, num_MP=mp, L=np.sqrt(100 * 9 / 40), \
              length_random_walk=length_random_walk,solution=sol,target = 0) for ii in xrange(iteration_to_mediate))
        print 'n=100 k=20: ', time.time() - tt
        tt = time.time()
        y2 = Parallel(n_jobs=num_cores)(delayed(main)(n0=200, k0=20, eta0=eta, C1=5, num_MP=mp, L=np.sqrt(200 * 9 / 40), \
              length_random_walk=length_random_walk,solution=sol,target = 0) for ii in xrange(iteration_to_mediate))
        print 'n=200 k=20: ', time.time() - tt
        tt = time.time()
        y3 = Parallel(n_jobs=num_cores)(delayed(main)(n0=200, k0=40, eta0=eta, C1=5, num_MP=mp, L=np.sqrt(200 * 9 / 40), \
              length_random_walk=length_random_walk,solution=sol,target = 0) for ii in xrange(iteration_to_mediate))
        print 'n=200 k=40: ', time.time() - tt
        tt = time.time()
        y4 = Parallel(n_jobs=num_cores)(delayed(main)(n0=500, k0=50, eta0=eta, C1=5, num_MP=mp, L=np.sqrt(500 * 9 / 40), \
               length_random_walk=length_random_walk, solution=sol, target=0) for ii in xrange(iteration_to_mediate))
        print 'n=500 k=50: ', time.time() - tt
        tt = time.time()
        y5 = Parallel(n_jobs=num_cores)(delayed(main)(n0=1000, k0=100, eta0=eta, C1=5, num_MP=mp, L=np.sqrt(1000 * 9 / 40), \
               length_random_walk=length_random_walk, solution=sol, target=0) for ii in xrange(iteration_to_mediate))
        print 'n=1000 k=100: ', time.time() - tt
        print 'Parallel time: ', time.time() - parallel

        y0 = np.sum(y0, 0) / iteration_to_mediate
        y1 = np.sum(y1, 0) / iteration_to_mediate
        y2 = np.sum(y2, 0) / iteration_to_mediate
        y3 = np.sum(y3, 0) / iteration_to_mediate
        y4 = np.sum(y4, 0) / iteration_to_mediate
        y5 = np.sum(y5, 0) / iteration_to_mediate


        # -- Salvataggio su file --
        with open('Risultati_txt/Paper1_algo1/plot_Fig3_='+str(length_random_walk),'wb') as file:
             wr=csv.writer(file,quoting=csv.QUOTE_ALL)
             wr.writerow(y0)
             wr.writerow(y1)
             wr.writerow(y2)
             wr.writerow(y3)
             wr.writerow(y4)
             wr.writerow(y5)


    x = np.linspace(1, 2.5, len(eta), endpoint=True)
    plt.xlabel('Decoding ratio $\eta$')
    plt.ylabel('Succesfull decoding probability')
    plt.axis([1, 2.5, 0, 1])
    plt.plot(x, y0, label='100 nodes and 10 sources', color='blue', linewidth=1,marker='o',markersize=4.0)
    plt.plot(x, y1, label='100 nodes and 20 sources', color='red', linewidth=1,marker='o',markersize=4.0)
    plt.plot(x, y2, label='200 nodes and 20 sources', color='grey', linewidth=1,marker='o',markersize=4.0)
    plt.plot(x, y3, label='200 nodes and 40 sources', color='magenta', linewidth=1,marker='o',markersize=4.0)
    plt.rc('legend', fontsize=10)
    plt.legend(loc=4)
    plt.grid()
    plt.savefig('Immagini/Paper1_algo1/00_Figure3_comparison_LR='+str(length_random_walk)+'c_0'+str(c0)+'delta='+str(delta)+'.pdf', dpi=150, transparent=False)
    plt.close()

    x = np.linspace(1, 2.5, len(eta), endpoint=True)
    plt.xlabel('Decoding ratio $\eta$')
    plt.ylabel('Succesfull decoding probability')
    plt.axis([1, 2.5, 0, 1])
    plt.plot(x, y0, label='100 nodes and 10 sources', linewidth=1, marker='o', markersize=4.0)
    plt.plot(x, y1, label='100 nodes and 20 sources', linewidth=1, marker='o', markersize=4.0)
    plt.plot(x, y2, label='200 nodes and 20 sources', linewidth=1, marker='o', markersize=4.0)
    plt.plot(x, y3, label='200 nodes and 40 sources', linewidth=1, marker='o', markersize=4.0)
    plt.plot(x, y4, label='500 nodes and 50 sources', linewidth=1, marker='o', markersize=4.0)
    plt.plot(x, y5, label='1000 nodes and 100 sources', linewidth=1, marker='o', markersize=4.0)
    plt.rc('legend', fontsize=10)
    plt.legend(loc=4)
    plt.grid()
    plt.savefig('Immagini/Paper1_algo1/00_FINAL_comparison_LR=' + str(length_random_walk) + 'c_0' + str(
        c0) + 'delta=' + str(delta) + '.pdf', dpi=150, transparent=False)
    plt.close()


def comparison_solution_opt_problem(iteration_to_mediate, L_RW,n0,k0, mp1):
    ## Comparison different solution for the optimization problem ----------------------------------------------------------

    print 'Comparison for different solution of the optimization problem'

    num_cores = multiprocessing.cpu_count()
    print 'Numero di core utilizzati:', num_cores
    print 'Numero di medie da eseguire: ', iteration_to_mediate
    punti = [L_RW]
    eta = np.arange(1.0, 2.6, step=0.1)
    sol = ['ones', 'Larghi_Math', 'Stretti_Math', 'Annealing']

    n0 = [n0]
    k0 = [k0]




    for a in xrange(len(n0)):
        for length_random_walk in punti:
            print 'Lunghezza della random walk:', length_random_walk

            y0 = np.zeros((iteration_to_mediate, len(eta)))
            y1 = np.zeros((iteration_to_mediate, len(eta)))
            y2 = np.zeros((iteration_to_mediate, len(eta)))
            y3 = np.zeros((iteration_to_mediate, len(eta)))


            # -- Iterazione su diversi sistemi --
            parallel = time.time()
            tt = time.time()
            y0 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0[a], k0=k0[a], eta0=eta, C1=5, num_MP=mp1, L=5, \
                  length_random_walk=length_random_walk,solution=sol[0],target = 0) for ii in xrange(iteration_to_mediate))
            print 'Time taken by ' + str(sol[0]) + ' = ', time.time() - tt
            tt = time.time()
            y1 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0[a], k0=k0[a], eta0=eta, C1=5, num_MP=mp1, L=5, \
                  length_random_walk=length_random_walk,solution=sol[1],target = 0) for ii in xrange(iteration_to_mediate))
            print 'Time taken by ' + str(sol[1]) + ' = ', time.time() - tt
            tt = time.time()
            y2 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0[a], k0=k0[a], eta0=eta, C1=5, num_MP=mp1, L=5, \
                  length_random_walk=length_random_walk,solution=sol[2],target = 0) for ii in xrange(iteration_to_mediate))
            print 'Time taken by ' + str(sol[2]) + ' = ', time.time() - tt
            tt = time.time()
            y3 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0[a], k0=k0[a], eta0=eta, C1=5, num_MP=mp1, L=5, \
                  length_random_walk=length_random_walk, solution=sol[3], target=0) for ii in xrange(iteration_to_mediate))
            print 'Time taken by ' + str(sol[03]) + ' = ', time.time() - tt
            print
            print 'Total time: ', time.time() - parallel

            y0 = np.sum(y0, 0) / iteration_to_mediate
            y1 = np.sum(y1, 0) / iteration_to_mediate
            y2 = np.sum(y2, 0) / iteration_to_mediate
            y3 = np.sum(y3, 0) / iteration_to_mediate


            # plt.title('Decoding performances - '+str(n0)+' nodes and '+str(k0)+' sources')
            plt.xlabel('Decoding ratio $\eta$')
            plt.ylabel('Successfull decoding probability P$_s$')
            x = np.linspace(1, 2.5, len(y0), endpoint=True)
            plt.axis([1, 2.5, 0, 1])
            plt.plot(x, y0, label='All ones', color='blue', linewidth=1, marker='o', markersize=4.0)
            plt.plot(x, y1, label='Bounds on violation prob.', color='green', linewidth=1, marker='o', markersize=4.0)
            plt.plot(x, y2, label='Violation prob.', color='red', linewidth=1, marker='o', markersize=4.0)
            plt.plot(x, y3, label='Optimal solution', color='black', linewidth=1, marker='o', markersize=4.0)
            plt.rc('legend', fontsize=10)
            plt.legend(loc=4)
            plt.grid()
            plt.savefig('Immagini/Paper1_algo1/00_Comparison_OPTsolutions_LR=' + str(length_random_walk) + \
                        '_c0=' + str(c0) + '_delta=' + str(delta) + '_n=' + str(n0[a]) + '_k=' + str(k0[a]) + '.pdf',
                        dpi=150, transparent=False)
            plt.close()

            # -- Salvataggio su file --
            with open('Immagini/Paper1_algo1/00_Comparison_n='+str(n0[a])+'_k='+str(k0[a]),'wb') as file:
                wr=csv.writer(file,quoting=csv.QUOTE_ALL)
                wr.writerow(y0)
                wr.writerow(y1)
                wr.writerow(y2)
                wr.writerow(y3)


def comparison_length_of_random_walk(iteration_to_mediate, punti, n0, k0, mp):
    print 'Comparison for diffente length of random walk - Several network dimensions'
    num_cores = multiprocessing.cpu_count()
    print 'Numero di core utilizzati:', num_cores
    print 'Iteration to mediate ', iteration_to_mediate

    y = np.zeros((4,len(punti)))
    eta = np.arange(1.4, 3.6, step=0.1)


    sol = ['ones', 'Larghi_Math', 'Stretti_Math']
    n0 = [100, 100, 200, 200, 500 ,1000]
    k0 = [10, 20, 20, 40, 50, 100]
    dec_ratio = np.ones((len(n0), iteration_to_mediate, len(punti)))


    cont = -1
    target = 0.8               # ATTENZIONE, QUESTA VARIABILE E' POSIZIONATA ANCHE SOPRA LA DEF DEL MAIN E FUNGE DA VARIABILE GLOBALE
                                # SE SI NECESSITA DI MODIFICARLA, CMODIFICARLA LA SOPRA RIGA 107 (E DINTORNI)
    ttt=time.time()
    for length_random_walk in punti:
        cont += 1
        print 'Lunghezza della random walk:', length_random_walk
        y0 = np.zeros((iteration_to_mediate, len(eta)))
        y1 = np.zeros((iteration_to_mediate, len(eta)))
        y2 = np.zeros((iteration_to_mediate, len(eta)))
        y3 = np.zeros((iteration_to_mediate, len(eta)))
        y4 = np.zeros((iteration_to_mediate, len(eta)))
        y5 = np.zeros((iteration_to_mediate, len(eta)))

        # -- Iterazione su diversi sistemi --
        t = time.time()
        tt = time.time()
        y0 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0[0], k0=k0[0], eta0=eta, C1=5, num_MP=mp, L=np.sqrt(100*9/40), \
                                                     length_random_walk=length_random_walk, solution=sol[1], target = target) for ii in xrange(iteration_to_mediate))
        print 'n='+str(n0[0])+' k='+str(k0[0])+': ', time.time() - tt

        for i in xrange(iteration_to_mediate):
            ii = 0
            while ii < len(eta) and y0[i][ii]< target:
                dec_ratio[0][i][cont] = eta[ii]
                ii += 1
        # tt = time.time()
        # y1 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0[1], k0=k0[1], eta0=eta, C1=5, num_MP=mp, L=np.sqrt(100*9/40), \
        #                                              length_random_walk=length_random_walk, solution=sol[1], target = target) for ii in xrange(iteration_to_mediate))
        # print 'n=' + str(n0[1]) + ' k=' + str(k0[1]) + ': ', time.time() - tt

        for i in xrange(iteration_to_mediate):
            ii = 0
            while ii < len(eta) and y1[i][ii]< target:
                dec_ratio[1][i][cont] = eta[ii]
                ii += 1

        # tt = time.time()
        # y2 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0[2], k0=k0[2], eta0=eta, C1=5, num_MP=mp, L=np.sqrt(200*9/40), \
        #                                              length_random_walk=length_random_walk, solution=sol[1], target = target) for ii in xrange(iteration_to_mediate))
        # print 'n=' + str(n0[2]) + ' k=' + str(k0[2]) + ': ', time.time() - tt

        for i in xrange(iteration_to_mediate):
            ii = 0
            while ii < len(eta) and y2[i][ii]< target:
                dec_ratio[2][i][cont] = eta[ii]
                ii += 1

        tt = time.time()
        y3 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0[3], k0=k0[3], eta0=eta, C1=5, num_MP=mp, L=np.sqrt(200*9/40), \
                                                      length_random_walk=length_random_walk, solution=sol[1], target = target) for ii in xrange(iteration_to_mediate))
        print 'n=' + str(n0[3]) + ' k=' + str(k0[3]) + ': ', time.time() - tt

        for i in xrange(iteration_to_mediate):
            ii = 0
            while ii < len(eta) and y3[i][ii]< target:
                dec_ratio[3][i][cont] = eta[ii]
                ii += 1

        tt = time.time()
        y4 = Parallel(n_jobs=num_cores)(delayed(main)(n0=n0[4], k0=k0[4], eta0=eta, C1=5, num_MP=mp, L=np.sqrt(500*9/40), \
                       length_random_walk=length_random_walk, solution=sol[1], target = target) for ii in xrange(iteration_to_mediate))
        print 'n=' + str(n0[4]) + ' k=' + str(k0[4]) + ': ', time.time() - tt

        for i in xrange(iteration_to_mediate):
            ii = 0
            while ii < len(eta) and y4[i][ii] < target:
                dec_ratio[4][i][cont] = eta[ii]
                ii += 1

        tt = time.time()
        y5= Parallel(n_jobs=num_cores)(delayed(main)(n0=n0[5], k0=k0[5], eta0=eta, C1=5, num_MP=mp, L=np.sqrt(1000*9/40), \
                      length_random_walk=length_random_walk, solution=sol[1],target=target) for ii in xrange(iteration_to_mediate))
        print 'n=' + str(n0[5]) + ' k=' + str(k0[5]) + ': ', time.time() - tt

        for i in xrange(iteration_to_mediate):
            ii = 0
            while ii < len(eta) and y5[i][ii] < target:
                dec_ratio[5][i][cont] = eta[ii]
                ii += 1
        print 'Time random walk '+str(length_random_walk)+': ', time.time()- t


        y0 = np.sum(y0, 0) / iteration_to_mediate
        y1 = np.sum(y1, 0) / iteration_to_mediate
        y2 = np.sum(y2, 0) / iteration_to_mediate
        y3 = np.sum(y3, 0) / iteration_to_mediate
        y4 = np.sum(y4, 0) / iteration_to_mediate
        y5 = np.sum(y5, 0) / iteration_to_mediate


    medie_dec_ratio = np.zeros((len(n0),len(punti)))
    medie_dec_ratio[0][:] = sum(dec_ratio[0], 0) /iteration_to_mediate
    medie_dec_ratio[1][:] = sum(dec_ratio[1], 0) /iteration_to_mediate
    medie_dec_ratio[2][:] = sum(dec_ratio[2], 0) /iteration_to_mediate
    medie_dec_ratio[3][:] = sum(dec_ratio[3], 0) /iteration_to_mediate
    medie_dec_ratio[4][:] = sum(dec_ratio[4], 0) / iteration_to_mediate
    medie_dec_ratio[5][:] = sum(dec_ratio[5], 0) / iteration_to_mediate

    # -- Salvataggio su file --
    with open('Immagini/Paper1_algo1/LRW comparison','wb') as file:
        wr=csv.writer(file,quoting=csv.QUOTE_ALL)
        wr.writerow(medie_dec_ratio[0][:])
        wr.writerow(medie_dec_ratio[1][:])
        wr.writerow(medie_dec_ratio[2][:])
        wr.writerow(medie_dec_ratio[3][:])
        wr.writerow(medie_dec_ratio[4][:])
        wr.writerow(medie_dec_ratio[5][:])


    print 'Total time:', time.time()-ttt


    plt.xlabel('Length of random walk')
    plt.ylabel('Average decoding ratio $\eta$')
    plt.axis([0, punti[-1],medie_dec_ratio[3][-1]-0.2 , medie_dec_ratio[0][0]+0.2])
    lista=[0,3,4,5]
    for i in lista:
        plt.plot(punti, medie_dec_ratio[i][:], label='n =' +str(n0[i])+ ' k = '+str(k0[i]), linewidth=1, marker='o', markersize=4.0)
    plt.rc('legend', fontsize=10)
    plt.legend(loc=1)
    plt.grid()
    plt.savefig('Immagini/Paper1_algo1/00_Comparison_Length_Random_Walk_'+str(punti)+'_c0=' + str(c0) \
                + '_delta=' + str(delta) + '_n=' + str(n0) + '_k=' + str(k0) + '.pdf', dpi=150,transparent=False)
    plt.close()


def dissemination_cost(iteration_to_mediate):
    # FIGURA DEI TEMPI DI DISSEMINAZIONE
    num_cores = multiprocessing.cpu_count()
    print 'Numero di core utilizzati:', num_cores

    print 'Numero di medie da eseguire: ', iteration_to_mediate

    eta = np.arange(1.0, 2.6, step=0.1)
    n0 = [100, 200, 500, 1000]
    y0 = np.zeros(iteration_to_mediate)
    y1 = np.zeros(iteration_to_mediate)
    y2 = np.zeros(iteration_to_mediate)
    y3 = np.zeros(iteration_to_mediate)
    y = np.zeros(len(n0))

    mp1 = 3000
    mp2 = 2500
    mp3 = 2500
    length_random_walk = 50
    sol = 'Larghi_Math'
    parallel = time.time()
    tt = time.time()
    y0 = Parallel(n_jobs=num_cores)(delayed(main)(n0=100, k0=10, eta0=eta, C1=5, num_MP=mp2, L=np.sqrt(1000 * 9 / 40), \
                                                  length_random_walk=length_random_walk, solution=sol, target=2) for ii
                                    in xrange(iteration_to_mediate))
    y[0] = np.sum(y0, 0) / iteration_to_mediate

    print 'n=100 k=10: ', time.time() - tt

    tt = time.time()
    y1 = Parallel(n_jobs=num_cores)(delayed(main)(n0=200, k0=40, eta0=eta, C1=5, num_MP=mp2, L=np.sqrt(1000 * 9 / 40), \
                                                  length_random_walk=length_random_walk, solution=sol, target=2) for ii
                                    in xrange(iteration_to_mediate))
    y[1] = np.sum(y1, 0) / iteration_to_mediate
    print 'n=200 k=40: ', time.time() - tt

    tt = time.time()
    y2 = Parallel(n_jobs=num_cores)(delayed(main)(n0=500, k0=50, eta0=eta, C1=5, num_MP=mp2, L=np.sqrt(1000 * 9 / 40), \
                                                  length_random_walk=length_random_walk, solution=sol, target=2) for ii
                                    in xrange(iteration_to_mediate))
    print 'n=500 k=50: ', time.time() - tt
    y[2] = np.sum(y2, 0) / iteration_to_mediate
    tt = time.time()

    y3 = Parallel(n_jobs=num_cores)(delayed(main)(n0=1000, k0=100, eta0=eta, C1=5, num_MP=mp2, L=np.sqrt(1000 * 9 / 40), \
                                                  length_random_walk=length_random_walk, solution=sol, target=2) for ii
                                    in xrange(iteration_to_mediate))
    y[3] = np.sum(y3, 0) / iteration_to_mediate
    print 'n=1000 k=100: ', time.time() - tt
    print 'Parallel time: ', time.time() - parallel

    # -- Salvataggio su file --
    with open('Risultati_txt/Tempi disseminazione Paper1 algo1 ', 'wb') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(y)



if __name__ == "__main__":

    comparison_network_dimensions(iteration_to_mediate=1, L_RW=100, mp=1)
    # Comparison for several network dimensions (n0,k0)=[(100,10);(100,20);(200,20);(200,40);(500,50);(1000,100)]
    # This function produces two separate figures, one comparing first 4 couples n,k, the other considering all values.
    # Moreover it generates a text file containing the vectors of the plots.
    # Graphs can be found at: ../Immagini/Paper1_algo1/00_Figure3_comparison_LR=......
    #                         ../Immagini/Paper1_algo1/00_FINAL_comparison_LR=......
    # Text file can be found at: ../Risultati_txt/Paper1_algo1/plot_Fig3_=.......

    comparison_solution_opt_problem(iteration_to_mediate=1, L_RW=100,n0=200,k0=40, mp=1)
    # Comparison for several solutions of the optimization problem, it generates a graph and a text file.
    # Graphs can be found at: ../Immagini/Paper1_algo1/00_Comparison_OPTsolutions_LR=......
    # Text file can be found at: ../Risultati_txt/Paper1_algo1/00_Comparison_n......

    punti = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 80]
    #iteration_to_mediate = 200
    comparison_length_of_random_walk(iteration_to_mediate=1,L_RW=punti, n0=200, k0=40,solution=1)
    # Comparison for several length of the random walk (punti). It generates a file containing the results.
    # Text file can be found at: ../Immagini/Paper1_algo1/LRW comparison
    # NB: This function is heavy time consuming.

    dissemination_cost(iteration_to_mediate=1)
    # Compute the dissemination time, in order to asses the complexity. Presented in the overall comparison in the report of the project