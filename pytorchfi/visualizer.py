import matplotlib
import numpy as np
import matplotlib.image as mpimg 
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import os

images = os.listdir('Images/')

#data = np.genfromtxt('test.csv', delimiter=',')

chart_count = len(images)
a_min = 0
a_max = chart_count - 1
a_init = 0
slider_position = 0

chart_list = list(range(0, chart_count))

fig = plt.figure()

for chart in range(0, chart_count):



	image = mpimg.imread('Images/' + images[chart])
	# print("This is the " + str(chart) + "th time this loop has happened")
	# print("For loop (1) Chart file is: " + str(chart_list[chart]) + "\n");
	# print("(1)      Chart is: " + str(chart) + "\n")
	# print("(1)      image is: " + images[chart] + "\n")

	chart_list[chart] = fig.add_subplot(1, chart_count, chart)
	chart_list[chart].set_position([0.25, 0.3, 0.5, 0.5])
	chart_list[chart].axis('off')
	chart_list[chart] = plt.imshow(image)

	# print("For loop (2) Chart file is: " + str(chart_list[chart]) + "\n");
	# print("(2)      Chart is: " + str(chart) + "\n")
	# print("(2)      image is: " + images[chart] + "\n")
	# print("-------------------------------------------------\n")
	# print("\n")

	chart_list[chart].set_visible(False)

chart_list[0].set_visible(True)

slider_ax = plt.axes([0.1, 0.12, 0.8, 0.05])

a_slider = Slider(slider_ax, 'Layer', a_min, a_max, valinit=a_init, valfmt='%d')

def update(layer):
	global slider_position
	chart_list[slider_position].set_visible(False)
	chart_list[layer].set_visible(True)
	slider_position = layer

def change_check(number):
	number = int(number)
	global slider_position
	if(number != slider_position):
		update(number)
	
a_slider.on_changed(change_check)

fig.suptitle('Vulnerability Visualization')

plt.show()

