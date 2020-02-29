import matplotlib
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('test.csv', delimiter=',')

intensity_data = [[]]
curr_layer = 0

fig = plt.figure()

column_count = 2

for column in range(1, column_count + 1):

	ax = fig.add_subplot(1, column_count, column)

	ax.imshow(data, interpolation = 'bilinear')

	ax.axis('off')

#ax2 = fig.add_subplot(1, 2, 2)

#ax2.imshow(data, interpolation='bilinear')

#ax1.axis('off')

#ax2.axis('off')

fig.subplots_adjust(wspace=0)


#cbar1 = ax[0].figure.colorbar(im1, ax=ax, cmap='YlGn')
#cbar2 = ax[0,1].figure.colorbar(im2, ax=ax, cmap='YlGn')
#cbar1.ax[0,0].set_ylabel('Vulnerability', rotation=-90, va='bottom')
plt.show()

#print(layer_data)
