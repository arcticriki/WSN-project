import numpy as np  # import of package numpy for mathematical tools
import matplotlib.pyplot as plt  # import of package matplotlib.pyplot for plottools
import time as time
from RSD import *
import csv


# -- Salvataggio su file --
y = np.zeros((4,16))
i = 0
with open('Primo print', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')#, quotechar='|')
    for row in reader:
        y[i, :] = row
        i += 1

y0_1 = y[0, :]
y1_1 = y[1, :]
y2_1 = y[2, :]
y3_1 = y[3, :]

y = np.zeros((4,16))
i = 0
with open('Secondo Print', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')#, quotechar='|')
    for row in reader:
        y[i, :] = row
        i += 1
y0_2 = y[0, :]
y1_2 = y[1, :]
y2_2 = y[2, :]
y3_2 = y[3, :]


# -- Plot --
# plt.title('Decoding performances')
# x = np.linspace(1, 2.5, 16, endpoint=True)
# plt.axis([1, 2.5, 0, 1])
# plt.plot(x, y0_1, label='100 nodes and 10 sources',color='blue'   ,linewidth=2)
# plt.plot(x, y0_2, label='100 nodes and 20 sources',color='red'    ,linewidth=2)
# plt.legend(loc=4)
# plt.grid()
# plt.show()
# plt.title('Decoding performances')
# x = np.linspace(1, 2.5, 16, endpoint=True)
# plt.axis([1, 2.5, 0, 1])
# plt.plot(x, y1_1, label='100 nodes and 10 sources',color='blue'   ,linewidth=2)
# plt.plot(x, y1_2, label='100 nodes and 20 sources',color='red'    ,linewidth=2)
# plt.legend(loc=4)
# plt.grid()
# plt.show()
# plt.title('Decoding performances')
# x = np.linspace(1, 2.5, 16, endpoint=True)
# plt.axis([1, 2.5, 0, 1])
# plt.plot(x, y2_1, label='100 nodes and 10 sources',color='blue'   ,linewidth=2)
# plt.plot(x, y2_2, label='100 nodes and 20 sources',color='red'    ,linewidth=2)
# plt.legend(loc=4)
# plt.grid()
# plt.show()
# plt.title('Decoding performances')
# x = np.linspace(1, 2.5, 16, endpoint=True)
# plt.axis([1, 2.5, 0, 1])
# plt.plot(x, y3_1, label='100 nodes and 10 sources',color='blue'   ,linewidth=2)
# plt.plot(x, y3_2, label='100 nodes and 20 sources',color='red'    ,linewidth=2)
# plt.legend(loc=4)
# plt.grid()
# plt.show()


x = np.linspace(1, 2.5, 16, endpoint=True)

plt.subplot(2, 2, 1)
plt.title('100 nodes and 10 sources')
plt.axis([1, 2.5, 0, 1])
plt.plot(x, y0_1, label='Mattia',color='blue'   ,linewidth=2)
plt.plot(x, y0_2, label='Riccardo',color='red'    ,linewidth=2)
plt.legend(loc=4)

plt.subplot(2, 2, 2)                            #PLOT CODE FOR TAU/SD/PDF VARIABLES, SUBPLOT TYPE
plt.title('100 nodes and 20 sources')
plt.axis([1, 2.5, 0, 1])
plt.plot(x, y1_1, label='Mattia',color='blue'   ,linewidth=2)
plt.plot(x, y1_2, label='Riccardo',color='red'    ,linewidth=2)
plt.legend(loc=4)

plt.subplot(2, 2, 3)
plt.title('200 nodes and 20 sources')
plt.axis([1, 2.5, 0, 1])
plt.plot(x, y2_1, label='Mattia',color='blue'   ,linewidth=2)
plt.plot(x, y2_2, label='Riccardo',color='red'    ,linewidth=2)
plt.legend(loc=4)

plt.subplot(2, 2, 4)
plt.title('200 nodes and 40 sources')
plt.axis([1, 2.5, 0, 1])
plt.plot(x, y3_1, label='Mattia',color='blue'   ,linewidth=2)
plt.plot(x, y3_2, label='Riccardo',color='red'    ,linewidth=2)
plt.legend(loc=4)

plt.show()












# plt.title('Decoding performances')
# x = np.linspace(1, 2.5, 16, endpoint=True)
# plt.axis([1, 2.5, 0, 1])
# plt.plot(x, y00, label='100 nodes and 10 sources',color='blue'   ,linewidth=2)
# plt.plot(x, y11, label='100 nodes and 20 sources',color='red'    ,linewidth=2)
# plt.plot(x, y22, label='200 nodes and 20 sources',color='grey'   ,linewidth=2)
# plt.plot(x, y33, label='200 nodes and 40 sources',color='magenta',linewidth=2)
# plt.legend(loc=4)
# plt.grid()
# plt.show()