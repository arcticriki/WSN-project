import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import time as time
from Node import *

t = time.time()

n = 1000  # number of nodes
k = 200
L = 100   # square dimension

node_list = [Storage(_+1, rnd.uniform(0.0, L), rnd.uniform(0.0, L)) for _ in xrange(n)]        # initialization
sensors_position = rnd.sample(range(0, n), k)
for i in xrange(len(sensors_position)):
    node_list[sensors_position[i]] = Sensor(sensors_position[i]+1, rnd.uniform(0.0, L), rnd.uniform(0.0, L))

# Find nearest neighbours using euclidean distance
x = np.zeros(n)
y = np.zeros(n)
u = np.zeros(n)
v = np.zeros(n)
for i in xrange(n):
    for j in xrange(n):
        dist = np.sqrt(np.sum(np.square(np.asarray(node_list[i].get_pos()) - np.asarray(node_list[j].get_pos()))))
        if dist <= 1 and dist != 0:
            node_list[i].neighbor_write(node_list[j])

for i in xrange(n):      # printing the results
    #node_list[i].spec()
    [x[i], y[i]] = node_list[i].get_pos()
for i in sensors_position:
    [u[i], v[i]] = node_list[i].get_pos()

elapsed = time.time() - t
print elapsed

plt.title("Graphical representation of sensors' positions")
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.xticks([5*k for k in xrange(L/5+1)])
plt.yticks([5*k for k in xrange(L/5+1)])
plt.axis([-1, L+1, -1, L+1])
plt.plot(x, y, linestyle='', marker='o')
plt.plot(u, v, color='red', linestyle='', marker='o')
plt.show()

