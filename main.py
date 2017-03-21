import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import time as time
from Node import *

t = time.time()

n = 1000  # number of nodes
k = 200
L = 100   # square dimension

positions = np.zeros((n, 2))
node_list = []
for i in  xrange(n):
    x = rnd.uniform(0.0, L)
    y = rnd.uniform(0.0, L)
    node_list.append(Storage(i+1, x, y))       # initialization
    positions[i, :] = [x, y]

sensors_position = rnd.sample(range(0, n), k)
for i in xrange(len(sensors_position)):
    x = rnd.uniform(0.0, L)
    y = rnd.uniform(0.0, L)
    node_list[sensors_position[i]] = Sensor(sensors_position[i]+1, x, y)
    positions[i, :] = [x, y]


# Find nearest neighbours using euclidean distance
x = np.zeros(n)
y = np.zeros(n)
u = np.zeros(n)
v = np.zeros(n)
diff=np.zeros(2)
dmax = 1                            # maximum distance for communication
dmax2 = dmax*dmax                   # square of maximum distance for communication
for i in xrange(n):                 # cycle on all nodes
    for j in xrange(n):             # compare each node with all the others
        diff = [positions[i, 0] - positions[j, 0],positions[i, 1] - positions[j, 1]]
        dist = diff[0]*diff[0] + diff[1]*diff[1]
        if dist <= dmax2:
            if dist != 0:
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

