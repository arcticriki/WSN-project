import numpy as np
import matplotlib.pyplot as plt
import time as time
from scipy import stats

# Robust Solition Distribution
def Robust_Soliton_Distribution(k, c0, delta, n):
    R = c0 * np.sqrt(k) * np.log(float(k)/delta)
    print k/R
    d = np.zeros(n)

    #Tau Function
    tau = []
    for i in range(1, k+1):
        if i <= round((k/R) - 1):
            tau.append(float(R)/(i*k))
        elif i >= round((k/R)+1):
            tau.append(0)
        else: #i == (k/R):
            tau.append((float(R)*np.log(float(R)/delta))/k)

    # Ideal Soliton Distribution
    sd = [1.0 / k]
    [sd.append(1.0 / (j * (j - 1))) for j in range(2, k+1)]

    # Beta
    beta =np.sum([tau[i]+sd[i] for i in xrange(k)])

    # Robust Soliton Distribution
    pdf = []
    for i in xrange(k):
        pdf.append((tau[i] + sd[i])/beta)

    # plt.subplot(3, 1, 1)
    # plt.title('Tau')
    # y = tau[0:50]
    # x = np.linspace(1, 50, 50, endpoint=True)
    # plt.bar(x, y)
    # #plt.axis([0, 50, 0, 0.45])
    # plt.subplot(3, 1, 2)                              PLOT CODE FOR TAU/SD/PDF VARIABLES, SUBPLOT TYPE
    # plt.title('Robust soliton cdf')
    # y1 = sd[0:50]
    # plt.bar(x, y1, color='red')
    # #plt.axis([0, 50, 0, 1])
    # plt.subplot(3, 1, 3)
    # plt.title('Robust soliton cdf')
    # y2 = pdf[0:50]
    # plt.bar(x, y2, color='red')
    # #plt.axis([0, 50, 0, ])
    # plt.show()

    xk = np.arange(1, k+1)                                      # Prob[X=xk]=pk. Thus, xk={indices of vector pdf}
    custm = stats.rv_discrete(name='custm', values=(xk, pdf))   # create object stats.custm that represents our distribution

    for i in xrange(n):
        d[i] = custm.rvs()                             #randomly sample an element from custm, following the
                                                            #distribution of custm
    return d


d = Robust_Soliton_Distribution(k=10000, c0=0.2, delta=0.05, n=100000)
print d
# plt.subplot(2, 1, 1)
# plt.title('Robust soliton pdf')
# y = pdf[0:50]
# x = np.linspace(1, 50, 50, endpoint=True)
# plt.bar(x, y)
# plt.axis([0, 50, 0, 0.45])
# plt.subplot(2, 1, 2)
# plt.title('Robust soliton cdf')
# y1 = cdf[0:50]
# plt.bar(x, y1, color='red')
# plt.axis([0, 50, 0, 1])
# plt.show()
