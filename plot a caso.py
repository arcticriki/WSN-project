
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
# punti = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 80, 120]
# medie_dec_ratio = np.zeros((4,len(punti)))
# n0 = [100, 100, 200, 200]
# k0 = [10, 20, 20, 40]
#
# # -- Salvataggio su file --
# with open('Immagini/Paper1_algo1/Final image','rb') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     i=0
#     for row in reader:
#         medie_dec_ratio[i][:] = row
#         i+=1
#
#
#
# plt.xlabel('Length of random walk')
# plt.ylabel('Average decoding ratio $\eta$')
# plt.axis([-0.1, punti[-3],1.4 , 3])
# for i in xrange(len(n0)):
#     plt.plot(punti[0:len(punti)-2], medie_dec_ratio[i][0:len(punti)-2], label='n =' +str(n0[i])+ ' k = '+str(k0[i]), linewidth=1, marker='o', markersize=4.0)
# plt.rc('legend', fontsize=12)
# plt.legend(loc=1)
# plt.grid()
# plt.show()
# plt.close()



# C_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
# number_of_points_in_x_axis = len(C_list)
# y = np.zeros((3,number_of_points_in_x_axis))
# n0 = [100,  500, 1000]
# k0 = [10,  50, 100]
# c0=0.2
# delta=0.05
# with open('Immagini/Paper2_algo1/Figure 6','rb') as file:
#     reader = csv.reader(file, delimiter=',')
#     # y8 = reader.readrow()
#     # y10 = reader.readrow()
#     # y11 = reader.readrow()
#     i = 0
#     for row in reader:
#         y[i][:] = row
#         i+=1
#
# x = C_list
# plt.xlabel('System parameter C$_1$')
# plt.ylabel('Successfull decoding probability P$_s$')
# plt.axis([0, C_list[-1], 0, 1])
# for i in xrange(len(n0)):
#     plt.plot(x, y[i][:], label=str(n0[i])+' nodes and '+str(k0[i])+' souces', linewidth=1, marker='o', markersize=4.0)
# #plt.plot(x, y10,label=str(n0[2])+' nodes and '+str(k0[2])+' souces', linewidth=1, marker='o', markersize=4.0)
# #plt.plot(x, y11,label=str(n0[3])+' nodes and '+str(k0[3])+' souces', linewidth=1, marker='o', markersize=4.0)
# plt.rc('legend', fontsize=10)
# plt.legend(loc=4)
# plt.grid()
# plt.savefig('Immagini/Paper2_algo1/00_Figure6_n='+str(n0)+'_k='+str(k0)+'_c0=' + str(c0) + '_delta=' + str(delta) + '.pdf', dpi=150,transparent=False)
# plt.close()



C_list = [10, 15, 20, 30, 40, 50, 60, 80, 100, 140, 200,270, 350,500]
num_c = len(C_list)
n0 = [100, 100, 200, 200, 500, 1000]
k0 = [10, 20, 20, 40, 50, 100]
medie_dec_ratio = np.zeros((len(n0), len(C_list)))
medie_dec_ratio1 = np.zeros((len(n0), len(C_list)))
c0=0.2
delta=0.05

with open('Immagini/Paper2_algo2/C3 variation','rb') as file:
    reader = csv.reader(file, delimiter=',')
    i = 0
    for row in reader:
        medie_dec_ratio[i][:] = row
        i+=1

massimi = np.zeros(len(n0))
minimi = np.zeros(len(n0))
for i in xrange(len(n0)):
    massimi[i] = max(medie_dec_ratio[i][:])
    minimi[i] = min(medie_dec_ratio[i][:])

plt.xlabel('System parameter C$_3$')
plt.ylabel('Average decoding ratio $\eta$')
#plt.axis([C_list[0]-0.5, C_list[-1]+0.5, min(minimi)-0.05, max(massimi)+0.05])
plt.axis([0, 500, 1.6, 2.6])
ff = [0,3,4,5]
for f in ff:
    plt.plot(C_list, medie_dec_ratio[f][:], label='n =' + str(n0[f]) + ' k = ' + str(k0[f]), linewidth=1,
             marker='o', markersize=4.0)
plt.rc('legend', fontsize=10)
plt.legend(loc=1)
plt.grid()
plt.savefig('Immagini/Paper2_algo2/C3 comparison_' + str(C_list) + '_c0=' + str(c0) \
            + '_delta=' + str(delta) + '_n=' + str(n0) + '_k=' + str(k0) + '.pdf', dpi=150, transparent=False)
plt.close()


# with open('Immagini/Paper2_algo2/C3 variation','rb') as file:
#     reader = csv.reader(file, delimiter=',')
#     i = 0
#     for row in reader:
#         medie_dec_ratio1[i][:] = row
#         i+=1
#
# massimi = np.zeros(len(n0))
# minimi = np.zeros(len(n0))
# for i in xrange(len(n0)):
#     massimi[i] = max(medie_dec_ratio[i][:])
#     minimi[i] = min(medie_dec_ratio[i][:])
#
# plt.xlabel('System parameter C$_3$')
# plt.ylabel('Average decoding ratio $\eta$')
# #plt.axis([C_list[0]-0.5, C_list[-1]+0.5, min(minimi)-0.05, max(massimi)+0.05])
# plt.axis([C_list[0]-0.5, C_list[-1]+0.5, 1.6, 3.0])
# ff=[3]
# for f in ff:
#     plt.plot(C_list, medie_dec_ratio[f][:], label='n =' + str(n0[f]) + ' k = ' + str(k0[f])+' file 200', linewidth=1,
#              marker='o', markersize=4.0)
#     plt.plot(C_list, medie_dec_ratio1[f][:], label='n =' + str(n0[f]) + ' k = ' + str(k0[f]), linewidth=1,
#              marker='o', markersize=4.0)
# plt.rc('legend', fontsize=10)
# plt.legend(loc=1)
# plt.grid()
# plt.show()
# plt.close()
#
