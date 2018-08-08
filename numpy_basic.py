import numpy as np
import matplotlib.pyplot as plt

samples = 100

# x = range(samples)
# y = np.linspace(1, 100, 20)
# y = np.random.normal(3, 0.2, samples)
# y = np.random.randn(samples)
# y = np.random.standard_normal(samples)
# print("x=", x)
# print("y=", y)
# plt.plot(x, y, 'ro', label='Original data')
# plt.show()

print("######demo python-based array op ######")
z0 = [[1, 2], [3, 4]]
z1 = [[1, 1], [1, 2]]
print ("add:", z0 + z1)
# print("multiply:", z0 * z1)  # not supported

print("######demo basic array op ######")
X0 = np.array([[1, 2], [3, 4]])
X1 = np.array([[1, 1], [1, 2]])
print("add:", X0 + X1)
print("multiply:", X0 * X1)

print("######demo basic matrix op ######")
Y0 = np.matrix([[1, 2], [3, 4]])
Y1 = np.matrix([[1, 1], [1, 2]])
print("add:", Y0 + Y1)
print("multiply:", Y0 * Y1)

print("######demo stack ######")
arr = [np.random.randn(1, 2) for _ in range(3)]
print("shape:", np.shape(arr), arr)
print("stack axis 0 , shape:", np.shape(np.stack(arr, axis=0)))
print("stack axis 1 , shape:", np.shape(np.stack(arr, axis=1)), np.stack(arr, axis=1))
print("stack axis 2 , shape:", np.shape(np.stack(arr, axis=-1)))

print("######demo vstack ######")
arrv1 = [1, 2]
arrv2 = [3, 4]
print("vstack, shape", np.shape(np.vstack((arrv1, arrv2))), np.vstack((arrv1, arrv2)))

print("######demo hstack ######")
arrh1 = [1, 2]
arrh2 = [3, 4]
print("hstack, shape", np.shape(np.hstack((arrv1, arrv2))), np.hstack((arrv1, arrv2)))

print("######demo concatenate ######")
a1 = [[[1, 2, 3, 4]]]
a2 = [[[4, 5, 6, 7]]]
print("concatenate by axis 0:")
concate0 = np.concatenate((a1, a2))
print(concate0, ", shape:", np.shape(concate0))
concate1 = np.concatenate((a1, a2), axis=1)
print(concate1, ", shape:", np.shape(concate1))
concate2 = np.concatenate((a1, a2), axis=-1)
print(concate2, ", shape:", np.shape(concate2))

print("######demo multivariate_normal ######")
mean = np.array([0,1])
cov = np.eye(2)
print("multivariate_normal: ", np.random.multivariate_normal(mean, cov, 3))