import matplotlib
import matplotlib.pylab as plt
import time
from math import sinh

def draw(T):
    plt.imshow(T, cmap='hot', origin='lower')
    plt.colorbar(label='Temperature')
    plt.title('Steady-State Temperature Distribution')
    plt.show()