import os
import sys


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Slider


# Creates a heatmap of the vulnerabilities for a single layer
def heat_map(numpy_data, layer_count, output_dir):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.axis("off")
    im = ax.imshow(numpy_data, interpolation="bilinear")
    try:
        plt.savefig(output_dir + "/Layer-" + str(layer_count) + "-Heat Map")
    except FileNotFoundError:
        os.mkdir("Graphs")
        plt.savefig(output_dir + "/Layer-" + str(layer_count) + "-Heat Map")


# Creates a bar graph of vulnerabilites per feature
def sequential_bar_graph(numpy_data, layer_count, output_dir):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    data = list(numpy_data)[0]
    cols = list(range(1, len(data) + 1))
    ax.set_ylabel("Level of Vulnerability")
    ax.set_title("Layer " + str(layer_count + 1) + " Vulnerability per Feature Frame")
    ax.bar(cols, data)
    try:
        plt.savefig(output_dir + "/Layer-" + str(layer_count + 1) + "-Seq-Bar-Graph")
    except FileNotFoundError:
        os.mkdir("Graphs")
        plt.savefig(output_dir + "/Layer-" + str(layer_count + 1) + "-Seq-Bar-Graph")


# Creates bar graph of vulnerabilites per feature, sorted by vulnerability
def nonsequential_bar_graph(numpy_data, layer_count, output_dir):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    data = list(numpy_data)[0]
    cols = list(range(1, len(data) + 1))
    quicksort(data, 0, len(data) - 1)
    ax.set_ylabel("Level of Vulnerability")
    ax.set_title("Layer " + str(layer_count) + " Vulnerability per Feature Frame")
    ax.bar(cols, data)
    plt.savefig(output_dir + "/Layer-" + str(layer_count) + "-NonSeq-Bar-Graph")
    try:
        plt.savefig(output_dir + "/Layer-" + str(layer_count) + "-NonSeq-Bar-Graph")
    except FileNotFoundError:
        os.mkdir("Graphs")
        plt.savefig("Graphs/Layer-" + str(layer_count) + "-NonSeq-Bar-Graph")


# Quick sort functions
def partition(arr, low, high):
    i = low - 1
    pivot = arr[high]

    for j in range(low, high):
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def quicksort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quicksort(arr, low, pi - 1)
        quicksort(arr, pi + 1, high)


# Gets the data points of each layer, then calls the generator
# to output the graphs of each layer
def generate_graph(file_name, generator, output_dir):
    try:
        input_file = open(file_name)
        name_line = input_file.readline()
        csv_line = input_file.readline()
        layer_count = 0
        graph_data = [[]]
        while csv_line != "":
            csv_arr = csv_line.split(",")
            if int(csv_arr[1]) != layer_count:
                generator(np.array(graph_data), layer_count, output_dir)
                graph_data = [[]]
                layer_count += 1
            graph_data[0].append(float(csv_arr[3]))
            csv_line = input_file.readline()
        generator(np.array(graph_data), layer_count, output_dir)
    except FileNotFoundError:
        print("Input file does not exist/does not have read permissions")
    finally:
        input_file.close()


"""
Parameters
"""
# Graph types: seq-bar, non-seq-bar, heat-map
graph_type_param = "type="
file_name_param = "file="
output_dir_param = "dir="


if __name__ == "__main__":
    graph_type = ""
    file_name = ""
    output_dir = ""
    for arg in sys.argv[1:]:
        if graph_type_param in arg:
            graph_type = arg[len(graph_type_param) :]
        elif file_name_param in arg:
            file_name = arg[len(file_name_param) :]
        elif output_dir in arg:
            output_dir = arg[len(output_dir_param) :]
        else:
            print("Invalid parameter")

    if graph_type == "seq-bar":
        generate_graph(file_name, sequential_bar_graph, output_dir)
    elif graph_type == "non-seq-bar":
        generate_graph(file_name, nonsequential_bar_graph, output_dir)
    elif graph_type == "heat-map":
        generate_graph(file_name, heat_map, output_dir)
    else:
        print("Invalid graph type")
