import matplotlib
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('test.csv', delimiter=',')

fig = plt.figure()

column_count = 5

for column in range(1, column_count + 1):

	ax = fig.add_subplot(1, column_count, column)

	ax.imshow(data, interpolation = 'bilinear')

	ax.axis('off')

fig.subplots_adjust(wspace=0)

plt.show()

