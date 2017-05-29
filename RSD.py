import numpy as np
from scipy import stats
import time as time
import matplotlib.pyplot as plt
# Robust Solition Distribution


def Robust_Soliton_Distribution(n, k, c0, delta):
    R = c0 * np.sqrt(k) * np.log(float(k)/delta)            # computation of R parameter
    d = np.zeros(n)   # initialization of degree variable
    #print k/R

    #Tau Function
    tau = np.zeros(k)  # initialization of tau
    for i in range(1, k + 1):  # computation of tau, it follows the formula on the papers
        if i <= round((k / R) - 1):
            tau[i - 1] = (float(R) / (i * k))
        elif i >= round((k / R) + 1):
            break
        else:  # case i == (k/R), since K/R is not integer equality is never
            tau[i - 1] = (float(R) * np.log(float(R) / delta)) / k  # verified, so ve use >= <= instead

    # for i in range(1, k + 1):  # computation of tau, it follows the formula on the papers
    #     if i < k / R:
    #             tau[i - 1] = (float(R) / (i * k))
    #     elif i > k / R:
    #             break
    #     else:  # case i == (k/R), since K/R is not integer equality is never
    #             tau[i - 1] = (float(R) * np.log(float(R) / delta)) / k  # verified, so ve use >= <= instead

    # Ideal Soliton Distribution
    sd = [1.0 / k]                                          # initialization of soliton distribution with first value
    [sd.append(1.0 / (j * (j - 1))) for j in range(2, k+1)] # append of other values following the formula in papers

    # Beta
    beta = np.sum([tau[i]+sd[i] for i in xrange(k)])        # computation of normalization parameter

    # Robust Soliton Distribution
    pdf = []                                                # initialization of target pdf
    for i in xrange(k):
        pdf.append((tau[i] + sd[i])/beta)                   # computation of pdf using tau, sd and the normalization coeff



    # upper = 50
    # if len(pdf)<50:
    #     upper = len(pdf)
    # plt.subplot(3, 1, 1)
    # plt.title('Tau')
    # #y = np.zeros(50)
    # #y[:len(tau)] = tau
    # y = tau[0:upper]
    # x = np.linspace(1, upper, upper, endpoint=True)
    # #plt.plot(x,y)
    # plt.bar(x, y)
    # plt.axis([0, upper, 0, np.max(tau)])
    # plt.subplot(3, 1, 2)                            #PLOT CODE FOR TAU/SD/PDF VARIABLES, SUBPLOT TYPE
    # plt.title('Robust soliton cdf')
    # #y1 = np.zeros(50)
    # #y1[:len(sd)] = sd
    # y1 = sd[0:upper]
    # #plt.plot(x, y1, color='red')
    # plt.bar(x, y1, color='red')
    # plt.axis([0, upper, 0, 0.6])
    # plt.subplot(3, 1, 3)
    # plt.title('Robust soliton cdf')
    # #y2 = np.zeros(50)
    # #y2[:len(pdf)] = pdf
    # y2 = pdf[0:upper]
    # #plt.plot(x, y2, color='red')
    # plt.bar(x, y2, color='red')
    # plt.axis([0, upper, 0, 0.6])
    # plt.show()

    xk = np.arange(1, k+1)                                  # vector of degree, start from 1 and goes up to k
    custm = stats.rv_discrete(name='custm', values=(xk, pdf))   # creation of obj custm, see documentation

    for i in xrange(n):
        d[i] = custm.rvs()                          # randomly sample n elements from custm, following the
                                                    # distribution of custm
    return d,pdf, R                                       # return sampled degree
