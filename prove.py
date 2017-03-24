import multiprocessing
import time as time

def worker():
    """worker function"""
    print 'Worker'
    return
t = time.time()
if __name__ == '__main__':
    jobs = []
    for i in range(5000):
        p = multiprocessing.Process(target=worker)
        jobs.append(p)
        p.start()

elapsed = time.time() - t
print elapsed

t = time.time()
jobs = []
for i in range(5000):
    jobs.append(worker())

elapsed = time.time() - t
print elapsed