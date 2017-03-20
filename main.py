import numpy as np
import random as rnd


class Node(object):

    def __init__(self, ID, X, Y):
        self.ID = ID
        self.X = X
        self.Y = Y

    def node_write(self, ID, X, Y):
        self.ID = ID
        self.X = X
        self.Y = Y

    def spec(self):
        print 'Node ID is %d and its position is (x=%d, y=%d)' % (self.ID, self.X, self.Y)


n = 10
k = 5
L = 50   # lato del quadrato in metri

node_list = [Node(_+1, rnd.randrange(0, L+1) , rnd.randrange(0, L+1)) for _ in xrange(n)]        # initialization

for i in xrange(n):
    node_list[i].spec()