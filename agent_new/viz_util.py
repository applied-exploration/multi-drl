# For Interactive Plotting
from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np


def interactive_plotting(y, x=None):
    #y=[0]
    #for i in range(50):
        clear_output(wait=True)
        #y.append(np.log(np.e*i))
        plt.plot(y)
        plt.show()