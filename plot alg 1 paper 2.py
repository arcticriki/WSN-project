from RSD import *
import csv


# ----------- FIGURE 3 AND 4 -------------------

eta = [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5]

y0 = np.zeros(len(eta))
y1 = np.zeros(len(eta))
y2 = np.zeros(len(eta))
y3 = np.zeros(len(eta))
y4 = np.zeros(len(eta))
y5 = np.zeros(len(eta))

i = 0
y = np.zeros((4, len(eta)))
with open('Figure 3', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')#, quotechar='|')
    for row in reader:
        y[i, :] = row
        i += 1

y0 = y[0, :]
y1 = y[1, :]
y2 = y[2, :]
y3 = y[3, :]


y = np.zeros((3,len(eta)))
i = 0
with open('Figure 4', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')#, quotechar='|')
    for row in reader:
        y[i, :] = row
        i += 1

y2 = y[0, :]
y4 = y[1, :]
y5 = y[2, :]


plt.title('Decoding performances')
x = np.linspace(1, 2.5, 16, endpoint=True)
plt.axis([1, 2.5, 0, 1])
plt.plot(x, y0, label='100 nodes and 10 sources',color='blue'   ,linewidth=2)
plt.plot(x, y1, label='100 nodes and 20 sources',color='red'    ,linewidth=2)
plt.plot(x, y2, label='200 nodes and 20 sources',color='grey'   ,linewidth=2)
plt.plot(x, y3, label='200 nodes and 40 sources',color='magenta',linewidth=2)
plt.legend(loc=4)
plt.grid()
plt.show()

plt.title('Decoding performances')
x = np.linspace(1, 2.5, 16, endpoint=True)
plt.axis([1, 2.5, 0, 1])
plt.plot(x, y2, label='200 nodes and 20 sources', color='blue', linewidth=2)
plt.plot(x, y4, label='500 nodes and 50 sources', color='red', linewidth=2)
plt.plot(x, y5, label='1000 nodes and 100 sources', color='magenta', linewidth=2)
plt.legend(loc=4)
plt.grid()
plt.show()

# ----------- FIGURE 5 -------------------

number_of_points_in_x_axis = 10
y6 = np.zeros(number_of_points_in_x_axis)
y7 = np.zeros(number_of_points_in_x_axis)

y = np.zeros((2, number_of_points_in_x_axis))
i = 0
with open('Figure 5', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')#, quotechar='|')
    for row in reader:
        y[i, :] = row
        i += 1

y6 = y[0, :]
y7 = y[1, :]

# -- Plot --
plt.title('Decoding performances')
x = np.linspace(500, 5000, number_of_points_in_x_axis, endpoint=True)
plt.axis([500, 5000, 0, 1])
plt.plot(x, y6, label='eta 1.4', color='blue', linewidth=2)
plt.plot(x, y7, label='eta 1.7', color='red', linewidth=2)
plt.legend(loc=4)
plt.grid()
plt.show()

# ----------- FIGURE 6 -------------------

number_of_points_in_x_axis = 10
y8 = np.zeros(number_of_points_in_x_axis)
y9 = np.zeros(number_of_points_in_x_axis)

y = np.zeros((2, number_of_points_in_x_axis))
i = 0
with open('Figure 6', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')  # , quotechar='|')
    for row in reader:
        y[i, :] = row
        i += 1

y8 = y[0, :]
y9 = y[1, :]

# -- Plot --
plt.title('Decoding performances')
x = np.linspace(0.5, 5, number_of_points_in_x_axis, endpoint=True)
plt.axis([0, 5, 0.5, 1])
plt.plot(x, y8, label='500 nodes and 50 souces', color='blue', linewidth=2)
plt.plot(x, y9, label='1000 nodes and 100 souces', color='red', linewidth=2)
plt.legend(loc=4)
plt.grid()
plt.show()
