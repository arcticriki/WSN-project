import numpy as np
import matplotlib.pyplot as plt
import pylab  # Plots; also imports array functions cumsum, transpose
import decimal
import random


# def RandomWalk(N=10, d=2):
#     """
#     Use numpy.cumsum and numpy.random.uniform to generate
#     a 2D random walk of length N, each of which has a random DeltaX and
#     DeltaY between -1/2 and 1/2.  You'll want to generate an array of
#     shape (N,d), using (for example), random.uniform(min, max, shape).
#     """
#     return numpy.cumsum(numpy.random.uniform(-0.5, 0.5, (N, d)))
#
#
# def PlotRandomWalkXT(N=10):
#     """
#     Plot X(t) for one-dimensional random walk
#     """
#     X = RandomWalk(N, 1)
#     pylab.plot(X)
#     pylab.show()
#
#
# def PlotRandomWalkXY(N=10):
#     """
#     Plot X, Y coordinates of random walk where
#         X = numpy.transpose(walk)[0]
#         Y = numpy.transpose(walk)[1]
#     To make the X and Y axes the same length,
#     use pylab.figure(figsize=(8,8)) before pylab.plot(X,Y) and
#     pylab.axis('equal') afterward.
#     """
#     walk = RandomWalk(N)
#     X, Y = numpy.transpose(walk)[0:2]
#     pylab.figure(figsize=(8, 8))
#     pylab.plot(X, Y)
#     pylab.axis('equal')
#     pylab.show()
#
#
# def Endpoints(W=10000, N=10, d=2):
#     """
#     Returns a list of endpoints of W random walks of length N.
#     (In one dimension, this should return an array of one-element arrays,
#     to be consistent with higher dimensions.)
#     One can generate the random walks and then peel off the final positions,
#     or one can generate the steps and sum them directly, for example:
#         sum(numpy.random.uniform(-0.5,0.5,(10,100,2))
#     """
#     return sum(numpy.random.uniform(-0.5, 0.5, (N, W, d)))
#
#
# def PlotEndpoints(W=10000, N=10, d=2):
#     """
#     Plot endpoints of random walks.
#     Use numpy.transpose to pull out X, Y.
#     To plot black points not joined by lines use pylab.plot(X, Y, 'k.')
#     Again, use pylab.figure(figsize=(8,8)) before and
#     pylab.axis('equal') afterward.
#     """
#     X, Y = numpy.transpose(Endpoints(W, N, d))
#     pylab.figure(figsize=(8, 8))
#     pylab.plot(X, Y, 'k.')
#     pylab.axis('equal')
#     pylab.show()
#
#
# def HistogramRandomWalk(W=10000, N=10, d=1, bins=50):
#     """
#     Compares the histogram of random walks with the normal distribution
#     predicted by the central limit theorem.
#     #
#     (1) Plots a histogram rho(x) of the probability that a random walk
#     with N has endpoint X-coordinate at position x.
#     Uses pylab.hist(X, bins=bins, normed=1) to produce the histogram
#     #
#     (2) Calculates the RMS stepsize sigma for a random walk of length N
#     (with each step uniform in [-1/2,1/2]
#     Plots rho = (1/(sqrt(2 pi) sigma)) exp(-x**2/(2 sigma**2))
#     for -3 sigma < x < 3 sigma on the same plot (i.e., before pylab.show).
#     Hint: Create x using arange. Squaring, exponentials, and other operations
#     can be performed on whole arrays, so typing in the formula for rho will
#     work without looping over indices, except sqrt, pi, and exp need to be
#     from the appropriate library (pylab, numpy, ...)
#     """
#     X = Endpoints(W, N, d)[:, 0]
#     pylab.hist(X, bins=bins, normed=1)
#     #
#     sigma = numpy.sqrt(N / 12.)
#     x = numpy.arange(-3 * sigma, 3 * sigma, sigma / bins)
#     rho = (1 / (numpy.sqrt(2 * numpy.pi) * sigma)) * numpy.exp(-x ** 2 / (2 * sigma ** 2))
#     pylab.plot(x, rho, "k-")
#     pylab.show()
#
#
# def yesno():
#     response = raw_input('    Continue? (y/n) ')
#     if len(response) == 0:  # [CR] returns true
#         return True
#     elif response[0] == 'n' or response[0] == 'N':
#         return False
#     else:  # Default
#         return True
#
#
# def demo():
#     """Demonstrates solution for exercise: example of usage"""
#     print "Random Walk Demo"
#     print "Random Walk X vs. t"
#     PlotRandomWalkXT(10000)
#     if not yesno(): return
#     print "Random Walk X vs. Y"
#     PlotRandomWalkXY(10000)
#     if not yesno(): return
#     print "Endpoints of many random walks"
#     print "N=1: square symmetry"
#     PlotEndpoints(N=1)
#     if not yesno(): return
#     print "N=10: emergent circular symmetry"
#     PlotEndpoints(N=10)
#     if not yesno(): return
#     print "Central Limit Theorem: Histogram N=10 steps"
#     HistogramRandomWalk(N=10)
#     if not yesno(): return
#     print "1 step"
#     HistogramRandomWalk(N=1)
#     if not yesno(): return
#     print "2 steps"
#     HistogramRandomWalk(N=2)
#     if not yesno(): return
#     print "4 steps"
#     HistogramRandomWalk(N=4)
#     if not yesno(): return
#
#
# if __name__ == "__main__":
#     demo()

# Robust Solition Distribution
def Robust_Soliton_Distribution(k, c0):
    delta = 0.05 #random.random()
    R = c0 * np.sqrt(k) * np.log10(float(k)/delta)

    #Tau Function
    tau = []
    for i in range(1, k):
        if i < ((k/R) - 1):
            tau.append(float(R)/(i*k))
        if i == (k/R):
            tau.append((float(R)*np.log(float(R)/delta))/k)
        if i > ((k/R) - 1):
            tau.append(0)


    #Ideal Soliton Distribution
    sd = []
    for j in range(1,k):
        if j == 1:
            sd.append(float(1)/k)
        if j > 1:
            sd.append(float(1)/(j*(j-1)))


    #Beta
    Beta = 0
    for i in range(0,k-1):
        Beta = Beta + (tau[i]+sd[i])

    #Robust Soliton Distribution
    rsd=[]
    for i in range(0, k - 1):
        rsd.append((tau[i] + sd[i])/Beta)

    plt.subplot(3, 1, 1)
    L = 50
    y1 = tau[0:50]
    plt.title('tau')
    x = np.linspace(1, 50, 50, endpoint=True)
    plt.bar(x, y1)
    plt.subplot(3, 1, 2)

    y2 = sd[0:50]
    plt.title('Ideal distribution')
    plt.bar(x, y2)

    plt.subplot(3, 1, 3)

    y3 = rsd[0:50]
    plt.title('Robust Soliton Distribution')
    plt.bar(x, y3)
    plt.ylabel('some numbers')
    plt.show()

    return rsd


prova = Robust_Soliton_Distribution(10000, 0.2)

# plt.title('Robust soliton')
# y = prova[0:50]
# x = np.linspace(1, 50, 50, endpoint=True)
# plt.bar(x,y)
# plt.xticks(np.linspace(0,52,53,endpoint=True))
# plt.ylabel('some numbers')
# plt.show()