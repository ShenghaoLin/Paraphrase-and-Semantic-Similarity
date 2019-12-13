import matplotlib.pyplot as plt
import math
from util import *

file = open("fff.pkl", "rb")
data = pickle.load(file)

train = []
test = []

data = sorted(data.items(), key=lambda ky : ky[1], reverse=True)

for d in data:
    train.append(d[1][0])
    test.append(d[1][1])


# We can set the number of bins with the `bins` kwarg
plt.plot(train, alpha=0.5)
plt.plot(test, alpha=0.5)
plt.show()
