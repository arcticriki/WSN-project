import random as rnd
import numpy as np
import time as time
from scipy import stats
import copy
from RSD import *

payload = 1

# -- We conceptually divide the nodes into 2 categories:
#    - STORAGE NODE: generic node that can only store and forward pkts
#    - SENSOR NODE: node that can actually sense the environment and generate pkts. It can also store and forward
#    (NOTE: class SENSOR extends class STORAGE)

#-- STORAGE NODE SPECIFICATIONS ---------------------------------------------------------------------------------
class Storage(object):

    def __init__(self, ID, X, Y, d, n, k, c1, c2, c3, pid, length_random_walk, c0, delta):
        self.c0 = c0                                # Robust Soliton Distribution parameters
        self.delta = delta
        self.C1 = c1                                # parameter C1
        self.C2 = c2                                # parameter C2
        self.C3 = c3                                # parameter C3
        self.ID = ID                                # ID of the node
        self.X  = X                                 # position
        self.Y  = Y                                 # position
        self.neighbor_list = []                     # list of references for neighbor nodes
        self.node_degree = 0                        # number of neighbors
        self.out_buffer  = []                       # outgoing packets buffer
        self.dim_buffer  = 0                        # number of pkt in the outgoing queue
        self.code_degree = d                        # degree of the node, define how many pkts to encode
        self.ID_list     = []                       # ID list of encoded pkts, saved for decoding purposes
        self.storage = np.zeros(payload, dtype=np.int64) # initialization of storage variable [0,..,0]
        self.num_encoded = 0                        # num. of encoded pkts, used to stop encoding process if it reaches
        self.n = n                                  # number of nodes in the network
        self.k = k                                  # number of sensors in the network
        self.visits = np.zeros(n)                   # n-dim list which purpose is to count pkts visits to the node
                                                    # it should be k-dim since only k nodes generate pkts but for
                                                    # computational reason it isn't. OPEN QUESTION
        self.already_coded = np.zeros(n)            # ancillary variable
        self.code_prob = self.code_degree / self.k  # coding probability, paper 2 algorithm 1
        self.pid = pid                              # steady state probability pi greco d, see formula 5 paper 1
        self.metropolis_prob = 0                    # variable containing the transition probabilities computed through the metropolis algorithm.
        self.num_received = 0                       # keep trace of how many pkt we received, algo 1 paper 1
        self.received_from_dissemination = []       # variable containing the pkt received from the dissemination
        self.already_received = np.zeros(n)         # keep trace of the already received pkts, in order not to save them both

        # Ancillary variables
        self.indicies = 0
        self.custm    = 0
        self.length_random_walk =length_random_walk # length of random walk. tunable parameter

        # Ancillary variables, paper 2 algorithm 2
        self.ku = 0.0                               # counter for number of different source pkts that have visited the node within c2
        self.first_arrived2 = np.zeros(3)           # variable containing ID, time, visit counter of first received pkt
        self.hops = []                              # list of list of pkt counters, saved upon pkt receiving
        [self.hops.append([]) for i in xrange(n)]   # how to create [[],[],.....,[]]
        self.stimati = False                        # boolean variable usen in received pkt 22
        self.last_hop  = 0                          # keep trace of last arrival hop counter
        self.n_stimato_hop  = 0                     # estimated values of n and k
        self.k_stimato_hop  = 0

    def get_pos(self):                              # return positions, DEPRECATED
        return self.X, self.Y

    def neighbor_write(self, neighbor):             # connect node with neighbors
        self.neighbor_list.append(neighbor)         # list of reference for neighbors node
        self.node_degree += 1                       # increase neighbor number at each insertion

    def funzione_ausiliaria(self):                      # creation of object custm, used to sample the RSD
        self.indicies = np.arange(0, self.node_degree)  # vector of indicies representing one neighbor
        self.custm = stats.rv_discrete(name='custm', values=(self.indicies, self.metropolis_prob))

    def send_pkt(self, mode):                           # send function used to move messages between nodes
        if self.dim_buffer > 0:                         # if buffer non empty --> we can send something
            if mode == 0:                               # mode 0 = uniform at random selection
                neighbor = rnd.choice(self.neighbor_list)
                pkt = self.out_buffer.pop(0)            # extract one pkt from the output buffer
                self.dim_buffer -= 1                    # reduce number of queued pkts
                return neighbor.receive_pkt21(pkt)      # pass pkt to neighbor and return 1 if blocked or 0 if not blocked
            elif mode == 1:                             # mode 1 = metropolis algo (ancora da implementare per paper 1)
                neighbor_idx = self.custm.rvs()         # randomly sample element from custm, following the distribution
                neighbor = self.neighbor_list[neighbor_idx]   # computed through the metropolis algorithm
                pkt = self.out_buffer.pop(0)            # extract one pkt from the output buffer
                self.dim_buffer -= 1                    # reduce number of queued pkts
                return neighbor.receive_pkt11(pkt)      # pass pkt to neighbor and return 1 if blocked or 0 if not blocked
            elif mode == 2:                             # for algo2 paper 2, parameter estimation
                neighbor = rnd.choice(self.neighbor_list)
                pkt = self.out_buffer.pop(0)            # extract one pkt from the output buffer
                self.dim_buffer -= 1                    # reduce number of queued pkts
                return neighbor.receive_pkt22(pkt)      # pass pkt to neighbor and return 1 if blocked or 0 if not blocked
        else:                                           # empty buffer
            return 0

# RECEIVE PER ALGO 1 PAPER 2
    def receive_pkt21(self, pkt):                   # define what to do on pkt receiving
        self.visits[pkt.ID-1] += 1                  # increase number of visits this pkt has done in this very node

        if self.visits[pkt.ID - 1] == 1:            # if it is the first time the pkt reaches this very node ...
                if self.num_encoded < self.code_degree: #...and we still have to encode something
                    prob = rnd.random()             # generate a random number in the range [0,1)
                    if prob <= self.code_prob:      # if generated number less or equal to coding probability
                      self.ID_list.append(pkt.ID)                   # save ID of node who generated the coded pkt
                      self.storage = self.storage ^ pkt.payload     # code procedure(XOR)
                      self.num_encoded += 1                         # increase num of encoded pkts
                pkt.counter += 1                    # increase pkt counter then put it in the outgoing buffer
                self.out_buffer.append(pkt)         # else, if pkt is at its first visit, or it haven't reached C1nlog10(n)
                self.dim_buffer += 1
                return 0                            # NOTE: this procedure has to be done even if the pkt has already visited
                                                    # the node! That is to say: if pkt x has visited node v before
                                                    # BUT c(x)<C1nlog(n), v accepts it with Prob=0, BUT it forwards it
        if self.visits[pkt.ID-1] > 1:
            if pkt.counter >= self.C1 * self.n * np.log10(self.n):  # if packet already visited the node
                                                    # and its counter is greater than C1nlog10(n) then, discard it
                return 1                            # pkt dropped
            else:
                pkt.counter += 1                    # increase pkt counter then put it in the outgoing buffer
                self.out_buffer.append(pkt)         # else, if pkt is at its first visit, or it haven't reached C1nlog10(n)
                self.dim_buffer += 1
                return 0                            # NOTE: this procedure has to be done even if the pkt has already visited
                                                    # the node! That is to say: if pkt x has visited node v before
                                                    # BUT c(x)<C1nlog(n), v accepts it with Prob=0, BUT it forwards it


# RECEIVE PER ALGO 2 PAPER 2 VERSIONE 2
    def receive_pkt22(self, pkt):                   # define what to do on pkt receiving
        pkt.C2 += 1
        if not self.stimati :                       # procedura pre stima di k ed n
            if self.first_arrived2[2]< self.C2:

                if self.first_arrived2[2] == 0:             # if it is the first pkt arriving, save ID, TIME, COUNTER=1
                    self.first_arrived2[:] = ([pkt.ID, pkt.C2, 0])         # hops

                if self.first_arrived2[0] == pkt.ID:         # se il pacchetto che vedo e' il primo, incremento il contatore
                    self.first_arrived2[2] += 1              # hops

                if not self.hops[pkt.ID-1]:                  # if it is the first time i see a pkt, increase counter ku
                    self.ku += 1.0                           # counter of pkt seen at least once (not increasing if already seen)

                self.hops[pkt.ID - 1].append(pkt.C2)         # save arrival hops for each pkt arriving
                self.last_hop  = pkt.C2                      # save hop counter of last received pkt, used in k estimation

                self.dim_buffer += 1                         # increase the number of queued pkts
                self.out_buffer.append(copy.deepcopy(pkt))   # add pkt to the outgoing queue
                return 0

            else:
                # stima di n
                J_tot_hop  = 0.0
                T_visit_hops  = 0.0
                for i in xrange(self.n):
                    l = len(self.hops[i])
                    if l > 1 :
                        T_visit_hops += (self.hops[i][-1] - self.hops[i][0]) / float(len(self.hops[i]))
                        J_tot_hop += len(self.hops[i])
                    elif l==1:
                        T_visit_hops += 0
                        J_tot_hop += 1
                self.n_stimato_hop = int(round(T_visit_hops / self.ku))

                # stima k
                T_packet_hop  = (self.last_hop  - self.first_arrived2[1]) / J_tot_hop
                self.k_stimato_hop  = int(round(self.n_stimato_hop  / T_packet_hop))

                # robust e campionamento d
                if self.k_stimato_hop >= 1:
                    self.stimati = True
                    self.code_degree , _, _ = Robust_Soliton_Distribution(1 , self.k_stimato_hop , self.c0, self.delta)  # See RSD doc
                    self.code_prob = self.code_degree / float(self.k_stimato_hop)  # compute the code probability, d/k
                else:
                    print 'rimandata'


                self.dim_buffer += 1                    # increase the number of queued pkts
                self.out_buffer.append(copy.deepcopy(pkt))  # add pkt to the outgoing queue
                return 0

        else: # procedura post stima
            self.visits[pkt.ID - 1] += 1  # increase number of visits this pkt has done in this very node
            pkt.counter += 1  # increace pkt counter

            if self.visits[pkt.ID - 1] == 1:            # if it is the first time the pkt reaches this very node ...
                if self.num_encoded < self.code_degree: # ...and we still have to encode something
                    prob = rnd.random()                 # generate a random number in the range [0,1)
                    if prob <= self.code_prob:          # if generated number less or equal to coding probability
                        self.ID_list.append(pkt.ID)     # save ID of node who generated the coded pkt
                        self.storage = self.storage ^ pkt.payload  # code procedure(XOR)
                        self.num_encoded += 1           # increase num of encoded pkts
                self.out_buffer.append(pkt)             # else, if pkt is at its first visit, or it haven't reached C1nlog10(n)
                self.dim_buffer += 1
                return 0                        # NOTE: this procedure has to be done even if the pkt has already visited
                                                # the node! That is to say: if pkt x has visited node v before
                                                # BUT c(x)<C1nlog(n), v accepts it with Prob=0, BUT it forwards it
            if self.visits[pkt.ID - 1] > 1:
                if pkt.counter >= self.C3 * self.n_stimato_hop * np.log10(self.n_stimato_hop):
                    # if packet already visited the node and its counter is greater than C1nlog10(n) then, discard it
                    return 1                            # pkt dropped
                else:
                    self.out_buffer.append(pkt)         # else, if pkt is at its first visit, or it haven't reached C1nlog10(n)
                    self.dim_buffer += 1
                    return 0                    # NOTE: this procedure has to be done even if the pkt has already visited
                                                # the node! That is to say: if pkt x has visited node v before
                                                # BUT c(x)<C1nlog(n), v accepts it with Prob=0, BUT it forwards it

# RECEIVE PER ALGO 1 PAPER 1
    def receive_pkt11(self, pkt):
        pkt.counter += 1                                        # increase the pkt forwardin counter
        if pkt.counter <= self.length_random_walk:
            self.dim_buffer += 1                                # increase the number of queued pkts
            self.out_buffer.append(copy.deepcopy(pkt))          # add pkt to the outgoing queue
            return 0
        else:
            if self.already_received[pkt.ID - 1] == 0:          # avoid double saving of the same pkt
                self.received_from_dissemination.append(copy.deepcopy(pkt))    # use a list to keep all received pkts
                self.already_received[pkt.ID - 1] += 1          # store the knowledge of the fact that the pkt with this
                self.num_received += 1                          # ID is already been saved in thi node
            return 1                                            # 1 means we stopped the pkt


    def storage_info(self):
        return self.num_encoded, self.ID_list       #return code degree of the node and list of ID of XORed pkts

    def encoding(self):                             # procedure that encode the received pkt, algo 1 paper 1
        if self.num_received >= self.code_degree:   # if number of received pkt >= of the node code degree
            #print 'caso ricevuti > d'
            select_nodes = rnd.sample(range(0, self.num_received), self.code_degree) # Select d random packets among the one saved
            self.num_encoded = self.code_degree             # number of codec pkt will match the code degree
            for i in xrange(len(select_nodes)):             # go through the random selceted pkts
                pkt = self.received_from_dissemination[select_nodes[i]]         # estract the pkt from the list of received ones
                self.ID_list.append(pkt.ID)                 # save the ID
                self.storage = self.storage ^ pkt.payload   # code procedure(XOR)
        else:                                               # if number of received pkt is less than the code degree
            #print 'caso ricevuti < d '
            self.num_encoded = self.num_received            # then the num of encoded pkt will match the number of received pkt
            for i in xrange(self.num_received):             # go through all the received pkt
                pkt = self.received_from_dissemination[i]   # extract the pkts
                self.ID_list.append(pkt.ID)                 # save the ID
                self.storage = self.storage ^ pkt.payload   # code procedure(XOR)


# -- SENSOR NODE SPECIFICATIONS ---------------------------------------------------------------------------------------
class Sensor(Storage):

    def __init__(self, ID, X, Y, d, n, k, c1, c2, c3, pid,length_random_walk, c0, delta):
        self.c0 = c0                                # parametri robust, da usare in paper 2 algo 2
        self.delta = delta

        self.C1 = c1                                # parameter C1
        self.C2 = c2                                # parameter C2
        self.C3 = c3                                # parameter C3
        self.ID = ID                                # ID of the node
        self.X = X                                  # position
        self.Y = Y                                  # position
        self.neighbor_list = []                     # list of references for neigbor nodes
        self.node_degree = 0                        # number of neighbors
        self.out_buffer = []                        # outgoing packets buffer
        self.dim_buffer = 0                         # number of pkt in the outgoing queue
        self.code_degree = d                        # degree of the node, define how many pkts to encode
        self.ID_list =[]                            # ID list of encoded pkts, saved for decoding purposes
        self.storage = np.zeros(payload, dtype=np.int64)  # initialization of storage variable [0,..,0]
        self.num_encoded = 0                        # num. of encoded pkts, used to stop encoding process if it reaces d
        self.n = n                                  # number of nodes in the network
        self.k = k                                  # number of sensors in the network
        self.visits = np.zeros(n)                   # n-dim list which purpose is to count pkts visits to the node
                                                    # it should be k-dim since only k nodes generate pkts but for
                                                    # computational reason it isn't. OPEN QUESTION
        self.already_coded = np.zeros(n)
        self.code_prob = self.code_degree / self.k
        self.pid = pid                              # steady state probability pi greco d, see formula 5 paper 1
        self.metropolis_prob = 0                    # variable containing the transition probabilities computed through the metropolis algorithm.
        self.num_received = 0                       # keep trace of how many pkt we received, algo 1 paper 1
        self.received_from_dissemination = []       # variable containing the pkt received from the dissemination
        self.already_received = np.zeros(n)         # keep trace of the already received pkts, in order not to save them both
        self.pkt_generated_gen3 = 0                 # variable containing the pkt generated from gen3, it is an ausiliary variable
        self.length_random_walk = length_random_walk                 # length of random walk. tunable parameter

        ###### VARIABILI ALGO2 PAPER 2
        self.ku = 0                                 # counter for number of different source pkts that have visited the node within c2
        self.first_arrived1 = np.zeros(3)           # variable containing ID, time, visit counter of first received pkt
        self.first_arrived2 = np.zeros(3)           # variable containing ID, time, visit counter of first received pkt
        self.times = []                             # list of list of arrival times
        [self.times.append([]) for i in xrange(n)]  # how to create [[],[],.....,[]]
        self.hops = []                              # list of list of pkt counters, saved upon pkt receiving
        [self.hops.append([]) for i in xrange(n)]   # how to create [[],[],.....,[]]
        self.stimati = False                        # boolean variable usen in received pkt 22
        self.last_time = 0                          # keep trace of last arrival time
        self.last_hop = 0                           # keep trace of last arrival hop counter

        self.n_stimato_time = 0                     # valori stimati di k e n nei due modi
        self.n_stimato_hop = 0
        self.k_stimato_time = 0
        self.k_stimato_hop = 0
        self.d_time = 0
        self.d_hop = 0

    def pkt_gen(self):
        pkt = Pkt(self.ID, payload)                 # generate a PKT object
        self.out_buffer.append(pkt)                 # set generated pkt as ready to be sent adding it to the outgoing buffer
        prob = rnd.random()                         # generate a random number in the range [0,1)
        if prob <= self.code_prob:                  # if generated number less or equal to coding probability
            self.ID_list.append(pkt.ID)             # save ID of node who generated the coded pkt
            self.storage = self.storage ^ pkt.payload  # code procedure(XOR)
            self.num_encoded += 1                   # increase num of encoded pkts
            self.already_coded[pkt.ID - 1] += 1     #store the knowledge of the fact that the pkt with this ID is already been coded in thi node
        self.dim_buffer = 1
        return pkt.payload

    def pkt_gen2(self):
        for i in xrange(self.number_random_walk):   # mette in coda lo stesso pacchetto tante quante sono le random walk che deve generare
            pkt = Pkt(self.ID, payload)             # generate a PKT object
            self.out_buffer.append(pkt)             # set generated pkt as ready to be sent adding it to the outgoing buffer
        self.dim_buffer = self.number_random_walk   # set the dim of the buffer to the number of queued pkts = number of random walk

        return pkt.payload, 0                       # return the pkt payload + non codificato

    def pkt_gen3(self):
        pkt = Pkt(self.ID, payload)                 # generate a PKT object
        self.pkt_generated_gen3 = pkt.payload
        return pkt.payload

# -- PKT SPECIFICATIONS -----------------------------------------------------------------------------------------------
class Pkt(object):
    def __init__(self, ID, pay):
        self.ID = ID
        self.counter = 0
        self.C2 = 0
        self.payload = np.zeros(pay, dtype=np.int64)
        for i in xrange(pay):
            self.payload[i] = np.random.randint(0, 2)