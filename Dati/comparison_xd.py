import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

Xd_giosue = 0
with open('C:\Users\Riccardo\Documents\Universitas\Wireless Systems & Networks\WSN-project\Dati\\60.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)  # , quotechar='|')
    for row in reader:
        Xd_giosue = row
for i in xrange(len(Xd_giosue)):
    Xd_giosue[i] = float(Xd_giosue[i])

print Xd_giosue

Xd_richi = 0
with open('C:\Users\Riccardo\Documents\Universitas\Wireless Systems & Networks\WSN-project\opt_variables\\x_60.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)  # , quotechar='|')
    for row in reader:
        Xd_richi = row
for i in xrange(len(Xd_giosue)):
    Xd_richi[i] = float(Xd_richi[i])


#---------------------------------------------------
d = np.linspace(1,60,60)
plt.plot(d,Xd_giosue,color='r',marker='o')
plt.plot(d,Xd_richi,color='b',marker='o')
plt.ylim(0,5)
plt.xlim(1,6)

plt.title('plots of xd')
plt.grid()
plt.show()
#--------------------------------------------------

Xd_giosue = [i * d[i] for i in Xd_giosue]
Xd_richi = [i * d[i] for i in Xd_richi]

d = np.linspace(1,60,60)
plt.plot(d,Xd_giosue,color='r',marker='o')
plt.plot(d,Xd_richi,color='b',marker='o')
plt.ylim(0,10)
plt.xlim(1,6)

plt.title('plots of d*xd')
plt.grid()
plt.show()

#--------------------------------------------------------------

Xd_giosue = [int(number) for number in Xd_giosue]
Xd_richi = [int(number) for number in Xd_richi]

d = np.linspace(1,60,60)
plt.plot(d,Xd_giosue,color='r',marker='o')
plt.plot(d,Xd_richi,color='b',marker='o')
plt.ylim(0,10)
plt.xlim(1,6)

plt.title('plots of round(xd*d)')
plt.grid()
plt.show()

#-------------------------------------------------------
k = 100
c = 0.2
delta_d = 0.05
R = (c*np.log(k/delta_d))*np.sqrt(k)
value = k/R
print value