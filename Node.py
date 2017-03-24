import random as rnd
import numpy as np

payload = 10
C1 = 10


class Storage(object):

    def __init__(self, ID, X, Y, d, n, k):
        self.ID = ID                                # ID of the node
        self.X = X                                  # position
        self.Y = Y                                  # position
        self.neighbor_list = []                     # list of references for neigbor nodes
        self.num_neighbor = 0                       # number of neighbors
        self.out_buffer = []                        # outgoing packets buffer
        self.degree = d                             # degree of the node, define how many pkts to encode
        self.ID_list =[]                            # ID list of encoded pkts, saved for decoding purposes
        self.storage = [np.random.randint(0, 1) for _ in xrange(payload)]  # initialization of storage variable [0,..,0]
        self.num_encoded = 0                        # number of encoded pkts, used to stop encoding process if it reaces d
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
        self.num_neighbor += 1                  # increase neighbor number at each insertion

    def send_pkt(self, mode):                   # send function used to move messages between nodes
        chosen_node = 0                         # initialize variable of chosen node to which tx
        if mode == 0:                           # mode 0 = uniform at random selection
            chosen_node = rnd.randint(0, self.num_neighbor-1)
        #elseif mode==1:                              # mode 1 = metropolis algo (ancora da implementare per paper 1)
            #chosen_node = metropolis()
        if self.out_buffer:                      # if buffer non empy --> we can send something
            pkt = self.out_buffer.pop(0)
            vicino = self.neighbor_list[chosen_node]
            vicino.receive_pkt(pkt)  # pop a message from the buffer,

    def receive_pkt(self, pkt):                 # define what to do on pkt receiving
        self.visits[pkt[1]] += 1                # increase number of visits this pkt has done in this very node
        if self.visits[pkt[1]] >= 1 and pkt[1] <= C1* self.n * np.log(self.n) : # if
            if self.visits[pkt[1]] == 1 and self.num_encoded <= self.d:
                if pkt[1] == 1:
                    prob = rnd.random()
                    if prob <= self.degree/self.k:
                        print 'sto codificando'
                        self.ID_list.append(pkt[0])
                        self.storage = [self.storage[i] ^ pkt[i+2] for i in xrange(len(pkt)-2)] #possibili evitare il for? forse no
            pkt[2] += 1
            self.out_buffer.append(pkt)
        else:                                   # if number of
            print 'Pacchetto %d bloccato.'
        #else non fare nulla e blocca quindi il pacchetto



class Sensor(Storage):

    def __init__(self, ID, X, Y, d, n, k):
        self.ID = ID
        self.X = X
        self.Y = Y
        self.neighbor_list = []
        self.out_buffer = []
        self.num_neighbor = 0
        self.degree = d
        self.ID_list =[]
        self.storage = [np.random.randint(0, 1) for _ in xrange(payload)]
        self.n = n
        self.k = k
        self.visits = np.zeros(n)

    def spec(self):
        print 'Sensor ID is %d and its position is (x=%d, y=%d)' % (self.ID, self.X, self.Y)

    def pkt_gen(self):
        pkt = [np.random.randint(0, 2) for _ in xrange(payload+2)]  # size specify the whole dim of the packet,size-2 il the dim of the payload
        pkt[0] = self.ID        # first cell of pks is node ID which generated it
        pkt[1] = 0              # second cell of vector is the counter for pkt as specified in page [174] paper 2
        self.out_buffer.append(pkt)
        prob = rnd.random()
        if prob <= self.degree / self.k:
            print 'sto codificando'
            self.ID_list.append(pkt[0])
            self.storage = [self.storage[i] ^ pkt[i + 2] for i in xrange(len(pkt) - 2)]  # possibili evitare il for? forse no

