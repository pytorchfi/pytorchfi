import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os

'''
Creates heatmap of the vulnerabilities for a single layer
'''
def heat_map(numpy_data, layer_count, output_dir):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.axis('off')
    im = ax.imshow(numpy_data, interpolation='bilinear')
    try:
        plt.savefig(output_dir + '/Layer-' + str(layer_count) + '-Heat Map')
    except FileNotFoundError:
        os.mkdir('Graphs')
        plt.savefig(output_dir + '/Layer-' + str(layer_count) + '-Heat Map')

'''
Creates bar graph sorted by layer number
'''
def sequential_bar_graph(numpy_data, layer_count, output_dir):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    data = list(numpy_data)[0]
    cols = list(range(1, len(data) + 1))
    ax.set_ylabel('Level of Vulnerability')
    ax.set_title('Layer ' + str(layer_count + 1) + ' Vulnerability per Feature Frame')
    ax.bar(cols, data)
    try:
        plt.savefig(output_dir + '/Layer-' + str(layer_count + 1) + '-Seq-Bar-Graph')
    except FileNotFoundError:
        os.mkdir('Graphs')
        plt.savefig(output_dir + '/Layer-' + str(layer_count + 1) + '-Seq-Bar-Graph')

'''
Creates bar graph sorted by vulnerability
'''
def nonsequential_bar_graph(numpy_data, layer_count, output_dir):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    data = list(numpy_data)[0]
    cols = list(range(1, len(data) + 1))
    quicksort(data, 0, len(data) - 1)
    ax.set_ylabel('Level of Vulnerability')
    ax.set_title('Layer ' + str(layer_count) + ' Vulnerability per Feature Frame')
    ax.bar(cols, data)
    plt.savefig(output_dir + '/Layer-' + str(layer_count) + '-NonSeq-Bar-Graph')
    try:
        plt.savefig(output_dir + '/Layer-' + str(layer_count) + '-NonSeq-Bar-Graph')
    except FileNotFoundError:
        os.mkdir('Graphs')
        plt.savefig('Graphs/Layer-' + str(layer_count) + '-NonSeq-Bar-Graph')

'''
Quick sort functions
'''
def partition(arr, low, high):
    i = (low-1)
    pivot = arr[high]

    for j in range(low, high):
        if arr[j] <= pivot:
            i = i+1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[high] = arr[high], arr[i+1]
    return ( i+1)

def quicksort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quicksort(arr, low, pi-1)
        quicksort(arr, pi+1, high)

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
    finally:
        input_file.close()

if __name__ == "__main__":
    print("Select the graph you would like to generate:")
    print("(1) Sequential Bar Graph")
    print("(2) Non-Sequential Bar Graph")
    print("(3) Heatmap")
    selection = str(input())
    file_name = str(input("Path to the CSV File: "))
    output_dir = str(input("Output directory: "))

    if selection == "1":
          generate_graph(file_name, sequential_bar_graph, output_dir)
    elif selection == "2":
          generate_graph(file_name, nonsequential_bar_graph, output_dir)
    elif selection == "3":
          generate_graph(file_name, heat_map, output_dir)
    else:
          print("Selection not valid.")

