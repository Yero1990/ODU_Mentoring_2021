import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
 return 1/(1+ np.exp(-x))

x = np.linspace(-10, 10, 100)
# print(x)
plt.plot(x,sigmoid(x),color='r',linestyle= '--')
plt.show()

