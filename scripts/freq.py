# test the frequency averaging approaches

import cmath
import numpy as np
import matplotlib.pyplot as mplot

print("Creating a complex array of random numbers...")
realp = np.random.rand(32,1)
imagp = np.random.rand(32,1)

carray = realp + imagp * 1j

print("The complex array:")
print(carray)

arrayfft = np.fft.fft(carray)
print("Complex array fft:")
print(arrayfft)

mplot.plot(abs(arrayfft))
mplot.show()
