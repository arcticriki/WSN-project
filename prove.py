import multiprocessing
import time as time


def funSquare(num):
    return num ** 2
t=time.time()

if __name__ == '__main__':
    pool = multiprocessing.Pool()
    results = pool.map(funSquare, range(10000000))
    #print(results)

elapsed = time.time() - t
print elapsed


t1=time.time()
results=[]
for i in xrange(10000000):
    results.append(funSquare(i))
#print results

elapsed = time.time() - t1
print elapsed