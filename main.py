import numpy as np
import random as rnd
import matplotlib.pyplot as plt


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

    def get_pos(self):
        return self.X, self.Y


n = 100
k = 5
L = 50   # lato del quadrato in metri

node_list = [Node(_+1, rnd.randrange(0, L+1) , rnd.randrange(0, L+1)) for _ in xrange(n)]        # initialization


x = np.zeros(n)
y = np.zeros(n)
for i in xrange(n):
    node_list[i].spec()
    [x[i], y[i]] = node_list[i].get_pos()


plt.title('Graphical representation of sensor positions.')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.xticks([5*k for k in xrange(L/5+1)])
plt.yticks([5*k for k in xrange(L/5+1)])
plt.axis([-1, L+1, -1, L+1])
plt.plot(x, y, linestyle='', marker='o')
plt.show()