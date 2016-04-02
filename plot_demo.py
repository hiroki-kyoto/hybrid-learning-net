import string
import matplotlib.pyplot as plt  
import numpy as np
import sys

# show arguments
print sys.argv[1]

years = range(0,100)
price = np.random.rand(100)
plt.plot(years, price, 'b*')
plt.plot(years, price, 'r')
plt.xlabel("years(+2000)")
plt.ylabel("housing average price(*2000 yuan)")
plt.ylim(0, 15)
plt.title('line_regression & gradient decrease')
plt.legend("DEMO")
plt.show()