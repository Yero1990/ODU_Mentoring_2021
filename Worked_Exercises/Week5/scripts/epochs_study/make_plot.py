import numpy as np
import matplotlib.pyplot as plt



y = 5
x_label = ('xfp_vs_xpfp')
x_pos = np.arange(len(x_label))

plt.xticks(x_pos, y)
plt.xticks(rotation=45)
plt.plot()
plt.show()
