from RSD import *
import csv
from matplotlib.pyplot import show, plot

eta = np.array([1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5])


y0 = np.zeros(len(eta))


#x=np.array([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,120,160,200,500,1000])
x=np.array([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100])
appoggio = np.zeros((4,len(eta)))
y = np.zeros((len(x), len(eta)))
mean_decoding = np.zeros(len(x))
cont1 = 0   #riga che voglio prendere tra le 4
cont2 = 0   #righe che prendo dentro ai file
riga = 3    #riga che voglio
for i in xrange(x[-1]):
    RW = 1 * ( i + 1 )
    try:
        with open('Risultati_txt/Paper1_algo1/plot_Fig3_variazione_Random_Walk='+str(RW), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')#, quotechar='|')
            cont2 = 0
            for row in reader:
                if cont2 == riga:
                    y[cont1 , :] = row
                    cont1 += 1
                    print cont1
                cont2 +=1
    except IOError:
        a=0

for ii in xrange(len(x)):
    somma = np.zeros(len(eta))
    for i in xrange(len(eta)-1):
        somma[i+1] += y[ii][i]
    y[ii] -= somma

yy = np.zeros(len(x))
for i in xrange(len(x)) :
    yy[i] = y[i].dot(eta)



plt.title('Decoding performances')



plt.axis([-10,x[-1]+10, 1, 2])
plt.plot(x, yy, label='avg dec. prob.', linewidth=2)
plt.legend(loc=4)
plt.grid()
plt.show()



