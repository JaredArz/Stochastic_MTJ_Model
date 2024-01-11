import matplotlib.pyplot as plt
import numpy


tf = 2.6e-9
T = numpy.array([4,35,75,110,150,295])
Ms = ( 1e-3 * numpy.array([1.51, 1.49, 1.44, 1.41, 1.37, 1.23]) ) / tf

print(T)
plt.plot(T, Ms)
plt.plot(T, list(map(lambda x: -374.9*x + 583467, T)))
plt.show()


