import matplotlib
import numpy as np
import matplotlib.image as mpimg 
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import os

images = os.listdir('Images/')

#data = np.genfromtxt('test.csv', delimiter=',')

chart_count = len(images)
a_min = 1
a_max = chart_count
a_init = 0

chart_list = list(range(0, chart_count))

fig = plt.figure()

#print("Number of images is " + str(len(images)) + "\n")

for chart in range(0, chart_count):

	chart_list[chart] = fig.add_subplot(chart_count, 1, chart)
	chart_list[chart].axis('off')

slider_ax = plt.axes([0.1, 0.12, 0.8, 0.05])


a_slider = Slider(slider_ax, 'Layer', a_min, a_max, valinit=a_init, valfmt='%d')


def update(layer):
	print("UPDATE RECEIVED: CURRENT LAYER IS ::" + str(layer) + " \n")
	chart_list[int(layer - 1)].imshow(mpimg.imread('./Images/' + images[int(layer) - 1]), interpolation = 'bilinear')
	chart_list[int(layer) - 1].axis('off')


a_slider.on_changed(update)


	
fig.suptitle('Vulnerability Visualization')

plt.show()

