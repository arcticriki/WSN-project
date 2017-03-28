import numpy as np
import matplotlib.pyplot as plt
import time as time


# Robust Solition Distribution
def Robust_Soliton_Distribution(k, c0, delta):
    R = c0 * np.sqrt(k) * np.log(float(k)/delta)

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
    rsd = []
    for i in xrange(k):
        rsd.append((tau[i] + sd[i])/beta)
    return rsd

k = 10000
pdf = Robust_Soliton_Distribution(k, c0=0.2, delta=0.05)
cdf = np.zeros(k)
cdf[0] = pdf[0]
for i in xrange(k-1):
    cdf[i+1] = cdf[i] + pdf[i+1]

t = time.time()
d = np.zeros(k)
for i in xrange(k):
    d[i] = np.random.choice(np.arange(1, k+1), p=pdf)
    print d[i]
elapsed = time.time() - t
print elapsed

plt.subplot(2, 1, 1)
plt.title('Robust soliton pdf')
y = pdf[0:50]
x = np.linspace(1, 50, 50, endpoint=True)
plt.bar(x, y)
plt.axis([0, 50, 0, 0.45])
plt.subplot(2, 1, 2)
plt.title('Robust soliton cdf')
y1 = cdf[0:50]
plt.bar(x, y1, color='red')
plt.axis([0, 50, 0, 1])
plt.show()
