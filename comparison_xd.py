import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

# #lettura file
# Xd_giosue = 0
# with open('C:\Users\Riccardo\Documents\Universitas\Wireless Systems & Networks\WSN-project\Dati\\60.csv', 'rb') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)  # , quotechar='|')
#     for row in reader:
#         Xd_giosue = row
# for i in xrange(len(Xd_giosue)):
#     Xd_giosue[i] = float(Xd_giosue[i])
#
# Xd_giosue = np.asarray(Xd_giosue)
# print Xd_giosue
#
# Xd_richi = 0
# with open('C:\Users\Riccardo\Documents\Universitas\Wireless Systems & Networks\WSN-project\opt_variables\\x_60.csv', 'rb') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)  # , quotechar='|')
#     for row in reader:
#         Xd_richi = row
# for i in xrange(len(Xd_giosue)):
#     Xd_richi[i] = float(Xd_richi[i])
#
# Xd_richi = np.asarray(Xd_richi)
#
# #------------------------------------------------------- print di R
# k = 100
c = 0.2
delta_d = 0.05
# R = (c*np.log(k/delta_d))*np.sqrt(k)
# value = k/R
# print 'K/R', value
#
# #------------------------------------------------------MSE
#
# MSE = np.sum(np.square(Xd_giosue[0:int(value)]-Xd_richi[0:int(value)]))/value
# print 'MSE', MSE
#
# #--------------------------------------------------- Plot di xd
# d = np.linspace(1,60,60)
# plt.plot(d,Xd_giosue,color='r',marker='o')
# plt.plot(d,Xd_richi,color='b',marker='o')
# plt.ylim(0,5)
# plt.xlim(1,6)
#
# plt.title('plots of xd')
# plt.grid()
# plt.show()
# #-------------------------------------------------- Plot di d*xd
#
# Xd_giosue = [i * d[i] for i in Xd_giosue]
# Xd_richi = [i * d[i] for i in Xd_richi]
#
# d = np.linspace(1,60,60)
# plt.plot(d,Xd_giosue,color='r',marker='o')
# plt.plot(d,Xd_richi,color='b',marker='o')
# plt.ylim(0,10)
# plt.xlim(1,6)
#
# plt.title('plots of d*xd')
# plt.grid()
# plt.show()
#
# #-------------------------------------------------------------- Plot di round(d*xd)
#
# Xd_giosue = [int(number) for number in Xd_giosue]
# Xd_richi = [int(number) for number in Xd_richi]
#
# d = np.linspace(1,60,60)
# plt.plot(d,Xd_giosue,color='r',marker='o')
# plt.plot(d,Xd_richi,color='b',marker='o')
# plt.ylim(0,10)
# plt.xlim(1,6)
#
# plt.title('plots of round(xd*d)')
# plt.grid()
# plt.show()


k=[10,20,40,60,100]
for a in k:
    R = (c * np.log(a / delta_d)) * np.sqrt(a)
    value = a / R
    print 'K/R', value


    Xgio = 0
    with open('Dati/'+str(a)+'.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)  # , quotechar='|')
            for row in reader:
                Xgio = row
    for i in xrange(len(Xgio)):
        Xgio[i] = float(Xgio[i])
    Xgio = np.asarray(Xgio)

    Xric = 0
    with open('opt_variables/x_'+str(a)+'.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)  # , quotechar='|')
            for row in reader:
                Xric = row
    for i in xrange(len(Xric)):
        Xric[i] = float(Xric[i])
    Xric = np.asarray(Xric)
    MSE = np.sum(np.square(Xgio[0:round(value)] - Xric[0:round(value)])) / value
    print MSE