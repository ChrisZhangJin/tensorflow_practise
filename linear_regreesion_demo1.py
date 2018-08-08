import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#
# 生成樣本數組
#
def generate_samples(num, mean, cov, diff):
    X0 = np.random.multivariate_normal(mean, cov, num)
    Y0 = np.zeros(num)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, num)
        Y1 = (ci + 1) * np.ones(num)

        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))

    return X0, Y0

np.random.seed(10)
num_class = 2

mean = np.random.randn(num_class)
cov = np.eye(num_class)
X, Y = generate_samples(200, mean, cov, [3])
# print("==========generate samples as followings:")
# print(X)
# print(Y)

colors = ['r' if l == 0 else 'b' for l in Y[:]]
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.xlabel("age(in year)")
plt.ylabel("size(in cm)")
# plt.show()

input_dim = 2
lab_dim = 1
input_feature = tf.placeholder(tf.float32, [None, input_dim])
input_labels = tf.placeholder(tf.float32, [None, lab_dim])

W = tf.Variable(tf.random_normal([input_dim, lab_dim]), name="weight")
b = tf.Variable(tf.zeros([lab_dim]), name="bias")

output = tf.nn.sigmoid(tf.matmul(input_feature, W) + b)
cross_entropy = -(input_labels * tf.log(tf.clip_by_value(output, 1e-10,1.0)) + (1 - input_labels) * tf.log(tf.clip_by_value(1 - output, 1e-10,1.0)))
ser = tf.square(input_labels - output)
err = tf.reduce_mean(ser)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.04).minimize(loss)

max_epochs = 50
mini_batch_size = 25

# print("len of X =", len(X))
# print("len of Y =", len(Y))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(max_epochs):
        print("===============================")
        sum_err = 0
        for i in range(np.int32(len(X) / mini_batch_size)):
            # print("+++++++++++i=", i)
            x1 = X[i * mini_batch_size:(i + 1) * mini_batch_size,:]
            y1 = np.reshape(Y[i * mini_batch_size:(i + 1) * mini_batch_size], [-1, 1])

            _, lossval, outputval, errval = sess.run([optimizer, loss, output, err], feed_dict={input_feature: x1, input_labels: y1})
            sum_err = sum_err + errval
            # print("##done! lossval:", lossval, "sum err:", sum_err)

        print("epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(lossval), "err=", sum_err/mini_batch_size)

    x = np.linspace(0, 10, 200)
    y = -x * (sess.run(W)[0] / sess.run(W)[1]) - sess.run(b) / sess.run(W)[1]
    plt.plot(x, y, label='Fitted line')
    plt.legend()
    plt.show()
