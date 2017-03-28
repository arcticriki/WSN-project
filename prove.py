from Node import *
import cProfile

class Pkt(object):
    def __init__(self, ID, pay):
        self.ID = ID
        self.counter = 0
        self.payload = [np.random.randint(0, 2) for _ in xrange(pay)]

def receive_pkt(Storage, pkt):                 # define what to do on pkt receiving
    self.visits[pkt.ID-1] += 1              # increase number of visits this pkt has done in this very node
    if self.visits[pkt.ID-1] > 1 and pkt.counter >= self.iteration:  # if packet already visited the node
                                                # and its counter is greater than C1nlog(n) then, discard it
        return 1                            # pkt dropped
    else:
        if self.visits[pkt.ID - 1] == 1 and self.num_encoded < self.code_degree:
                # if it is the first time the pkt reaches this very node and we have NOT already coded d pkts
            prob = rnd.random()             # generate a random number in the range [0,1)
            if prob <= self.code_prob:      # if generated number less or equal to coding probability
                    self.ID_list.append(pkt.ID)        # save ID of node who generated the coded pkt
                    self.storage = [self.storage[i] ^ pkt.payload[i] for i in xrange(payload)]  # code procedure(XOR)
                    self.num_encoded += 1              # increase num of encoded pkts

        pkt.counter += 1                         # increase pkt counter then put it in the outgoing buffer
        self.out_buffer.append(pkt)         # else, if pkt is at its first visit, or it haven't reached C1nlog(n)
        self.dim_buffer += 1
        return 0




n = 1000                                        # number of nodes
k = 200                                         # number of sensors
L = 25                                          # square dimension

positions = np.zeros((n, 2))                    # matrix containing info on all node positions
node_list = []                                  # list of references to node objects
d = np.ones(n)*2
payload=10
pkt = []
# for i in xrange(n):                             # for on 0 to n indices
#     x = rnd.uniform(0.0, L)                     # generation of random coordinate x
#     y = rnd.uniform(0.0, L)                     # generation of random coordinate y
#     node_list.append(Storage(i + 1, x, y, d[i], n, k))      # creation of storage node, function Storage()
#     pkt.append(Pkt(i+1, payload))
#
Storage(1, 0, 0, 1, 1, 1)
pkt= Pkt(1,payload)
cProfile.run('receive_pkt()')
