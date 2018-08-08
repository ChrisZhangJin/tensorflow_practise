import numpy as np
import matplotlib.pyplot as plt

# dim 1
num = 100000
mu = 0
sigma = 1
s = np.random.normal(mu, sigma, num)
print(s)
#s = sigma * np.random.randn(num) + mu
# s = sigma * np.random.standard_normal(num) + mu
# plt.subplot(141)
# plt.hist(s, bins=200, normed=True)
# plt.show()

# dim multi
num2 = 40000
mean = np.array([0,0])
cov = np.array([[100,100],[1,100]])
ms = np.random.multivariate_normal(mean, cov, num2)
print(ms)
print('@@@@@@@@@@@')
# print(np.reshape(ms, [2,-1]))
print(ms[:, 0])
plt.scatter(ms[:, 0], ms[:, 1])
# plt.figure()
# plt.plot(sx, sy)
# plt.axis('equal')
plt.show()

# np.random.multivariate_normal(mean, cov, sample_per_class)

