import matplotlib
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('test.csv', delimiter=',')

intensity_data = [[]]
curr_layer = 0
for i in range(len(data)):
    if data[i][1] != curr_layer:
        curr_layer += 1
        intensity_data.append([])
    intensity_data[curr_layer].append(data[i][3])
print(intensity_data)
graph = np.array([np.array(x) for x in intensity_data])
fig, ax = plt.subplots()
im = ax.imshow(intensity_data, interpolation='bilinear')
cbar = ax.figure.colorbar(im, ax=ax, cmap='YlGn')
cbar.ax.set_ylabel('Vulnerability', rotation=-90, va='bottom')
plt.show()

print(layer_data)
