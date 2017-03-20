import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from Node import Node

n = 100
k = 5
L = 50   # lato del quadrato in metri

node_list = [Node(_+1, rnd.uniform(0.0, L), rnd.uniform(0.0, L)) for _ in xrange(n)]        # initialization


x = np.zeros(n)
y = np.zeros(n)
for i in xrange(n):
    node_list[i].spec()
    [x[i], y[i]] = node_list[i].get_pos()


plt.title('Graphical representation of sensor positions. ')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.xticks([5*k for k in xrange(L/5+1)])
plt.yticks([5*k for k in xrange(L/5+1)])
plt.axis([-1, L+1, -1, L+1])
plt.plot(x, y, linestyle='', marker='o')
plt.show()