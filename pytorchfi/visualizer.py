import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def save_image(numpy_data, layer_count, last):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.axis('off')
    im = ax.imshow(np.array(graph_data), interpolation='bilinear')
    if last:
        cbar = ax.figure.colorbar(im, ax=ax, cmap='YlGn')
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.savefig("map" + str(layer_count))

try:
    input_file = open("alexnet.csv")
    name_line = input_file.readline()
    csv_line = input_file.readline()
    layer_count = 0
    graph_data = [[]]
    while csv_line != "":
        csv_arr = csv_line.split(",")
        if int(csv_arr[1]) != layer_count:
            save_image(np.array(graph_data), layer_count, False)
            graph_data = [[]]
            layer_count += 1
        graph_data[0].append(float(csv_arr[3]))
        csv_line = input_file.readline()
    save_image(np.array(graph_data), layer_count, True)
finally:
    input_file.close()
