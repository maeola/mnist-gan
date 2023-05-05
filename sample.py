import coloredlogs, logging
coloredlogs.install(fmt='%(levelname)7s %(asctime)s %(hostname)s %(message)s')

import numpy as np
import matplotlib.pyplot as plt

# a combination of different randoms

Bn = 256
Sn = 80

A = np.concatenate((np.random.randn(Bn) * 0.2 + 8, np.random.randn(Bn) * 3 - 10, np.random.randn(Bn), np.random.rand(Bn) + 2, np.random.pareto(3.0, size=Bn)))

# plt.hist(A, 100)
# plt.show()

means = []
stds = []

for k in range(100):
    sample_idx = np.random.choice(A.shape[0], size=Sn * A.shape[0] // 100, replace=False)
    h = A[sample_idx]
    means.append(h.mean())
    stds.append(h.std())

means = np.array(means)
stds = np.array(stds)

print(means.mean(), means.std())
print(stds.mean(), stds.std())