import random as rnd
import numpy as np

payload = 10
C1 = 10


class Storage(object):

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

    def node_write(self, ID, X, Y): #change ID and position coordinates
        self.ID = ID
        self.X = X
        self.Y = Y

    def spec(self):
        print 'Storage ID is %d and its position is (x=%d, y=%d)' % (self.ID, self.X, self.Y)

    def get_pos(self):
        return self.X, self.Y

    def neighbor_write(self, neighbor):
        self.neighbor_list.append(neighbor)
        self.num_neighbor += 1

    def send_pkt(self, mode):
        chosen_node = 0                             # initialize variable of chosen node to which tx
        if mode == 0:                               # mode 0 = uniform at random selection
            chosen_node = rnd.randint(0, self.num_neighbor-1)
        #elseif mode==1:                              # mode 1 = metropolis algo (ancora da implementare per paper 1)
            #chosen_node = metropolis()
        if self.out_buffer:                             # if buffer non empy --> we can send something
            pkt = self.out_buffer.pop(0)
            vicino = self.neighbor_list[chosen_node]
            vicino.receive_pkt(pkt)  # pop a message from the buffer,

    def receive_pkt(self, pkt):
        self.visits[pkt[1]] += 1
        if pkt[2] <= C1* self.n * np.log(self.n):
            if pkt[2] == 1:
                prob = rnd.random()
                if prob <= self.degree/self.k:
                    print 'sto codificando'
                    self.ID_list.append(pkt[0])

                    self.storage = [self.storage[i] ^ pkt[i+2] for i in xrange(len(pkt)-2)] #possibili evitare il for? forse no
            pkt[2] += 1
            self.out_buffer.append(pkt)
        #else non fare nulla



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

