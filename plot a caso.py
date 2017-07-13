
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



# C_list = [10, 15, 20, 30, 40, 50, 60, 80, 100, 140, 200,270, 350,500]
# num_c = len(C_list)
# n0 = [100, 100, 200, 200, 500, 1000]
# k0 = [10, 20, 20, 40, 50, 100]
# medie_dec_ratio = np.zeros((len(n0), len(C_list)))
# medie_dec_ratio1 = np.zeros((len(n0), len(C_list)))
# c0=0.2
# delta=0.05
#
# with open('Immagini/Paper2_algo2/C3 variation','rb') as file:
#     reader = csv.reader(file, delimiter=',')
#     i = 0
#     for row in reader:
#         medie_dec_ratio[i][:] = row
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
# plt.axis([0, 500, 1.6, 2.6])
# ff = [0,3,4,5]
# for f in ff:
#     plt.plot(C_list, medie_dec_ratio[f][:], label='n =' + str(n0[f]) + ' k = ' + str(k0[f]), linewidth=1,
#              marker='o', markersize=4.0)
# plt.rc('legend', fontsize=10)
# plt.legend(loc=1)
# plt.grid()
# plt.savefig('Immagini/Paper2_algo2/C3 comparison_' + str(C_list) + '_c0=' + str(c0) \
#             + '_delta=' + str(delta) + '_n=' + str(n0) + '_k=' + str(k0) + '.pdf', dpi=150, transparent=False)
# plt.close()


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










# eta = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5]
# sol = ['ones', 'Larghi_Math', 'Stretti_Math', 'Annealing']
# c0=0.2
# delta=0.05
# punti = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 80]
# y = np.zeros((4,len(punti)))
# eta = np.array([1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, \
#                 2.4, 2.5, 2.6, 2.7, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3 ,3.4, 3.5])
#
# n0 = [100, 100, 200, 200, 500 ,1000]
# k0 = [10, 20, 20, 40, 50, 100]
# medie_dec_ratio = np.ones((len(n0), len(punti)))
#
#
#
# i = 0
# with open('Risultati_txt/C1 comparison 100','rb') as file:
#     reader = csv.reader(file, delimiter=',')
#     for row in reader:
#         medie_dec_ratio[i][:] = row
#         i+=1
#         print i
#
# with open('Risultati_txt/C1 comparison 200','rb') as file:
#     reader = csv.reader(file, delimiter=',')
#     for row in reader:
#         medie_dec_ratio[i][:] = row
#         i+=1
#         print i
#
# with open('Risultati_txt/C1 comparison 500','rb') as file:
#     reader = csv.reader(file, delimiter=',')
#     for row in reader:
#         medie_dec_ratio[i][:] = row
#         i+=1
#         print i
#
# with open('Risultati_txt/C1 comparison 1000','rb') as file:
#     reader = csv.reader(file, delimiter=',')
#     for row in reader:
#         medie_dec_ratio[i][:] = row
#         i+=1
#         print i
#
#
#
# massimi = np.zeros(len(n0))
# minimi = np.zeros(len(n0))
# for i in xrange(len(n0)):
#     massimi[i] = max(medie_dec_ratio[i][:])
#     minimi[i] = min(medie_dec_ratio[i][:])
#
# plt.xlabel('Length of random walk')
# plt.ylabel('Average decoding ratio $\eta$')
# plt.axis([0, punti[-1],min(minimi)-0.05 , 2.5])
# ff = [0,3,4,5]
# for f in ff:
#     plt.plot(punti, medie_dec_ratio[f][:], label='n =' +str(n0[f])+ ' k = '+str(k0[f]), linewidth=1, marker='o', markersize=4.0)
# plt.rc('legend', fontsize=10)
# plt.legend(loc=1)
# plt.grid()
# plt.savefig('Immagini/Paper1_algo1/00_Comparison_Length_Random_Walk_'+str(punti)+'_c0=' + str(c0) \
#             + '_delta=' + str(delta) + '_n=' + str(n0) + '_k=' + str(k0) + '.pdf', dpi=150,transparent=False)
# plt.close()


















# eta = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5]
#
# y11 = 0
# y21 = 0
# y22 = 0
#
# with open('Risultati_txt/comparison finale/11','rb') as file:
#     reader = csv.reader(file, delimiter=',')
#     for row in reader:
#         y11 = row
#
# with open('Risultati_txt/comparison finale/21','rb') as file:
#     reader = csv.reader(file, delimiter=',')
#     for row in reader:
#         y21 = row
#
# with open('Risultati_txt/comparison finale/22','rb') as file:
#     reader = csv.reader(file, delimiter=',')
#     for row in reader:
#         y22 = row
#
#
#
#
#
# x = np.linspace(1, 2.5, len(eta), endpoint=True)
# plt.xlabel('Decoding ratio $\eta$')
# plt.ylabel('Succesfull decoding probability')
# plt.axis([1, 2.5, 0, 1])
# plt.plot(x, y11, label='EDFC', linewidth=1, marker='o', markersize=4.0)
# plt.plot(x, y21, label='LTCDS - I', linewidth=1, marker='o', markersize=4.0)
# plt.plot(x, y22, label='LTCDS - II', linewidth=1, marker='o', markersize=4.0)
# plt.rc('legend', fontsize=10)
# plt.legend(loc=4)
# plt.grid()
# plt.savefig('Immagini/00_FINAL_comparison_1000_100.pdf', dpi=150, transparent=False)
# plt.close()


# y11 = 0
# y21 = 0
# y22 = 0
# n0=[100,200,500,1000]
#
# with open('Risultati_txt/Tempi disseminazione Paper1 algo1 ','rb') as file:
#     reader = csv.reader(file, delimiter=',')
#     for row in reader:
#         y11 = row
#
# with open('Risultati_txt/Tempi disseminazione Paper2 algo1 ','rb') as file:
#     reader = csv.reader(file, delimiter=',')
#     for row in reader:
#         y21 = row
#
# with open('Risultati_txt/Tempi disseminazione Paper2 algo2 ','rb') as file:
#     reader = csv.reader(file, delimiter=',')
#     for row in reader:
#         y22 = row
#
#
#
# plt.xlabel('Network dimension n')
# plt.ylabel('Duration dissemination phase')
# plt.axis([100, 1000, 0, 400])
# plt.plot(n0, y11, label='EDFC', linewidth=1, marker='o', markersize=4.0)
# plt.plot(n0, y21, label='LTCDS - I', linewidth=1, marker='o', markersize=4.0)
# plt.plot(n0, y22, label='LTCDS - II', linewidth=1, marker='o', markersize=4.0)
# plt.rc('legend', fontsize=12)
# plt.legend(loc=2)
# plt.grid()
# plt.savefig('Immagini/00_FINAL_comparison_tempi.pdf', dpi=150, transparent=False)
# plt.close()




eta = [1.5, 2.0, 1.5, 2.0]
C_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60]

n0 = [100, 100, 200, 200]
k0 = [10, 10, 20, 20]
c0=0.2
delta= 0.05
i=0
y = np.zeros((len(n0),len(C_list)))

with open('Immagini/Paper2_algo2/C2 variation', 'rb') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        y[i,:] = row
        i += 1


plt.xlabel('Parameter $C_2$')
plt.ylabel('Successfull decoding probability P$_s$')
plt.axis([0, C_list[-1], 0, 1])
for i in xrange(len(n0)):
    plt.plot(C_list, y[i][:], label='n=' + str(n0[i]) + '   k=' + str(k0[i]) + '   $\eta$=' + str(eta[i]), linewidth=1,
             marker='o', markersize=4.0)
plt.rc('legend', fontsize=10)
plt.legend(loc=4)
plt.grid()
plt.savefig('Immagini/Paper2_algo2/00_COMPARISON C2 VALUE_n0=' + str(n0) + '_k0=' + str(k0) + '_c0=' + \
            str(c0) + 'delta=' + str(delta) + '.pdf', dpi=150, transparent=False)
plt.close()
