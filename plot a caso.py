
import numpy as np  # import of package numpy for mathematical tools
import random as rnd  # import of package random for homonym tools
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt  # import of package matplotlib.pyplot for plottools
import time as time
from Node2 import *         #importa la versione due della classe nodo
from send_mail import *
import cProfile
from RSD import *
from math import factorial
import csv
import copy
from joblib import Parallel, delayed
import multiprocessing
punti = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 80, 120]
medie_dec_ratio = np.zeros((4,len(punti)))
n0 = [100, 100, 200, 200]
k0 = [10, 20, 20, 40]

# -- Salvataggio su file --
with open('Immagini/Paper1_algo1/Final image','rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    i=0
    for row in reader:
        medie_dec_ratio[i][:] = row
        i+=1



plt.xlabel('Length of random walk')
plt.ylabel('Average decoding ratio $\eta$')
plt.axis([-0.1, punti[-3],1.4 , 3])
for i in xrange(len(n0)):
    plt.plot(punti[0:len(punti)-2], medie_dec_ratio[i][0:len(punti)-2], label='n =' +str(n0[i])+ ' k = '+str(k0[i]), linewidth=1, marker='o', markersize=4.0)
plt.rc('legend', fontsize=12)
plt.legend(loc=1)
plt.grid()
plt.show()
plt.close()

