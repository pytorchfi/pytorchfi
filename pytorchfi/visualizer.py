import matplotlib
import numpy as np
import matplotlib.pyplot as plt

layer_data = np.genfromtxt('test.csv', delimiter=',')
fig, ax = plt.subplots()
im = ax.imshow(layer_data)

plt.show()

print(layer_data)
