import matplotlib
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import os

images = os.listdir('Images/')

#data = np.genfromtxt('test.csv', delimiter=',')

fig = plt.figure()

row_count = len(images)

for row in range(1, row_count):

	ax = fig.add_subplot(row_count, 1, row)

	ax.imshow(mpimg.imread('./Images/' + images[row - 1]), interpolation = 'bilinear')

	ax.axis('off')
	

fig.subplots_adjust(hspace=0)

fig.suptitle('Vulnerability Visualization')

plt.show()

