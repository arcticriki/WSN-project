import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from Node import Node

n = 1000  # number of nodes
k = 5
L = 50   # square dimension

# n1 = Node(1,0,0)
# n2 = Node(2,1,1)
# n2.neighbor_write(n1)
# n1.ID=8
# n2.neighbor_list[0].spec()
node_list = [Node(_+1, rnd.uniform(0.0, L), rnd.uniform(0.0, L)) for _ in xrange(n)]        # initialization


# Find nearest neighbours using euclidean distance
x = np.zeros(n)
y = np.zeros(n)
for i in xrange(n):
    for j in xrange(n):
        dist = np.sqrt(np.sum(np.square(np.asarray(node_list[i].get_pos()) - np.asarray(node_list[j].get_pos()))))
        if dist <= 1 and dist != 0:
            node_list[i].neighbor_write(node_list[j])

for i in xrange(n):      # printing the results
    node_list[i].spec()
    [x[i], y[i]] = node_list[i].get_pos()


plt.title("Graphical representation of sensors' positions")
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.xticks([5*k for k in xrange(L/5+1)])
plt.yticks([5*k for k in xrange(L/5+1)])
plt.axis([-1, L+1, -1, L+1])
plt.plot(x, y, linestyle='', marker='o')
plt.show()