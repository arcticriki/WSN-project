import random as rnd
import numpy as np

payload = 10
C1 = 30


class Storage(object):

    def __init__(self, ID, X, Y, d, n, k):
        self.ID = ID                                # ID of the node
        self.X = X                                  # position
        self.Y = Y                                  # position
        self.neighbor_list = []                     # list of references for neigbor nodes
        self.node_degree = 0                       # number of neighbors
        self.out_buffer = []                        # outgoing packets buffer
        self.code_degree = d                             # degree of the node, define how many pkts to encode
        self.ID_list =[]                            # ID list of encoded pkts, saved for decoding purposes
        self.storage = [np.random.randint(0, 1) for _ in xrange(payload)]  # initialization of storage variable [0,..,0]
        self.num_encoded = 0                        # num. of encoded pkts, used to stop encoding process if it reaces d
        self.n = n                                  # number of nodes in the network
        self.k = k                                  # number of sensors in the network
        self.visits = np.zeros(n)                   # n-dim list which purpose is to count pkts visits to the node
                                                    # it should be k-dim since only k nodes generate pkts but for
                                                    # computational reason it isn't. OPEN QUESTION

    def node_write(self, ID, X, Y): #change ID and position coordinates, DEPRECATED
        self.ID = ID
        self.X = X
        self.Y = Y

    def spec(self):                 # print ID and positions, DEPRECATED
        print 'Storage ID is %d and its position is (x=%d, y=%d)' % (self.ID, self.X, self.Y)

    def get_pos(self):              # return positions, DEPRECATED
        return self.X, self.Y

    def neighbor_write(self, neighbor):         # connect node with neighbors
        self.neighbor_list.append(neighbor)     # list of reference for neighbors node
        self.node_degree += 1                  # increase neighbor number at each insertion

    def send_pkt(self, mode):                   # send function used to move messages between nodes
        if len(self.out_buffer) > 0:  # if buffer non empy --> we can send something
            chosen_node = 0                         # initialize variable of chosen node to which tx
            if mode == 0:                           # mode 0 = uniform at random selection
                chosen_node = rnd.randint(0, self.node_degree - 1)
            elif mode == 1:                         # mode 1 = metropolis algo (ancora da implementare per paper 1)
                print 'Metropolis algorithm not yet implemented'
            pkt = self.out_buffer.pop(0)
            vicino = self.neighbor_list[chosen_node]
            return vicino.receive_pkt(pkt), pkt[0]
        else:
            return 0, 0
                                                # send operation return a pkt and a neighbor

    def receive_pkt(self, pkt):                 # define what to do on pkt receiving
        self.visits[pkt[0]-1] += 1                # increase number of visits this pkt has done in this very node
        if self.visits[pkt[0]-1] > 1 and pkt[1] >= C1* self.n * np.log(self.n):  # if packet already visited the node
                                                # and its counter is greater than C1nlog(n) then, discard it
                                                #print 'Pacchetto %d bloccato.' % pkt[0]     # Check message
            return 1
        else:
            if self.visits[pkt[0] - 1] == 1 and self.num_encoded < self.code_degree:
                # if it is the first time the pkt reaches this very node and we have NOT already coded d pkts
                prob = rnd.random()             # generate a random number in the range [0,1)
                if prob <= self.code_degree / self.k:  # if generated number less or equal to coding probability
                    self.ID_list.append(pkt[0])        # save ID of node who generated the coded pkt
                    self.storage = [self.storage[i] ^ pkt[i + 2] for i in xrange(len(pkt) - 2)]  # code procedure(XOR)
                    self.num_encoded += 1              # increase num of encoded pkts

            pkt[1] += 1                         # increase pkt counter then put it in the outgoing buffer
            self.out_buffer.append(pkt)         # else, if pkt is at its first visit, or it haven't reached C1nlog(n)
            return 0



class Sensor(Storage):

    def __init__(self, ID, X, Y, d, n, k):
        self.ID = ID                            # ID of the node
        self.X = X                              # position
        self.Y = Y                              # position
        self.neighbor_list = []                 # list of references for neigbor nodes
        self.node_degree = 0                   # number of neighbors
        self.out_buffer = []                    # outgoing packets buffer
        self.code_degree = d                         # degree of the node, define how many pkts to encode
        self.ID_list = []                       # ID list of encoded pkts, saved for decoding purposes
        self.storage = [np.random.randint(0, 1) for _ in xrange(payload)]  # initialization of storage variable [0,..,0]
        self.num_encoded = 0                    # number of encoded pkts, used to stop encoding process if it reaces d
        self.n = n                              # number of nodes in the network
        self.k = k                              # number of sensors in the network
        self.visits = np.zeros(n)               # n-dim list which purpose is to count pkts visits to the node
                                                # it should be k-dim since only k nodes generate pkts but for
                                                # computational reason it isn't. OPEN QUESTION

    def spec(self):     # DEPRECATED
        print 'Sensor ID is %d and its position is (x=%d, y=%d)' % (self.ID, self.X, self.Y)

    def pkt_gen(self):
        pkt = [np.random.randint(0, 2) for _ in xrange(payload+2)]  # size specify the whole dim of the packet,size-2
                                                                    # is the dim of the payload
        pkt[0] = self.ID        # first cell of pks is node ID which generated it
        pkt[1] = 0              # second cell of vector is the counter for pkt as specified in page [174] paper 2
        self.out_buffer.append(pkt)  # set generated pkt as ready to be sent adding it to the outgoing buffer
        prob = rnd.random()                 # generate a random number in the range [0,1)
        if prob <= self.code_degree / self.k:    # if generated number less or equal to coding probability
            #print 'sto codificando'         # check message, DEPRECATED
            self.ID_list.append(pkt[0])     # save ID of node who generated the coded pkt
            self.storage = [self.storage[i] ^ pkt[i + 2] for i in xrange(len(pkt) - 2)]  # code procedure(XOR)
            self.num_encoded += 1           # increase num of encoded pkts

