
import numpy as np
import matplotlib.pyplot as plt
import seaborn;

X=np.random.rand(10,2)
#print(X)

seaborn.set()
plt.scatter(X[:,0],X[:,1],s = 100);
#plt.show();
print (np.shape(X[:, np.newaxis, :]))
print (np.shape(X[np.newaxis, :, :]))
differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]

sq_differences = differences ** 2
dist_sq = sq_differences.sum(-1)

K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)

for i in range(X.shape[0]):
    for j in nearest_partition[i, :K+1]:
        plt.plot(*zip(X[j], X[i]), color='black')
plt.show()
