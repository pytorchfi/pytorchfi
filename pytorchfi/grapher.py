import os


import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Slider

plt.rcParams["toolbar"] = "None"

images = os.listdir("Images/")


chart_count = len(images)
a_min = 0
a_max = chart_count - 1
a_init = 0
slider_position = 0

index_list = list(range(0, chart_count))
chart_list = list(range(0, chart_count))

fig = plt.figure()


for chart in range(0, chart_count):

    image = mpimg.imread("Images/" + images[chart])
    chart_list[chart] = fig.add_subplot(1, chart_count, chart)

    chart_list[chart].set_position([0.25, 0.3, 0.5, 0.5])
    chart_list[chart].axis("off")
    chart_list[chart] = plt.imshow(image)
    index_list[chart] = chart

    chart_list[chart].set_visible(False)

chart_list[0].set_visible(True)

slider_ax = plt.axes([0.1, 0.12, 0.8, 0.05])

a_slider = Slider(slider_ax, "Layer", a_min, a_max, a_init, "%d")
slider_ax.xaxis.set_visible(True)
slider_ax.set_xticks(index_list)


def update(layer):
    global slider_position

    chart_list[layer].set_visible(True)
    chart_list[slider_position].set_visible(False)

    slider_position = layer


def change_check(number):
    number = int(number)
    global slider_position
    if number != slider_position:
        update(number)


a_slider.on_changed(change_check)

fig.set_facecolor("#808080")

text = fig.suptitle("Layered Vulnerability Visualization", size="xx-large")
text.set_fontweight("black")
plt.show()
