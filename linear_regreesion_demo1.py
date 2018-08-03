import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def generate_samples(sample_size, mean, cov, diff, regression):
    num_class = 2
    sample_per_class = int(sample_size / 2)
    X0 = np.random.multivariate_normal(mean, cov, sample_per_class)
    Y0 = np.zeros(sample_per_class)
    # print("normal samples X0:", X0)
    # print("zeros Y0:", Y0)

    for ci, d in enumerate(diff):
        moved_mean = mean + d
        # print(ci, "##", d, "## moved_mean=", moved_mean)
        X1 = np.random.multivariate_normal(moved_mean, cov, sample_per_class)
        Y1 = (ci + 1) * np.ones(sample_per_class)
        # print ("X1", X1)
        # print ("Y1", Y1)

        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))
    # print("X0", X0)
    # print("Y0", Y0)
    if regression == False:
        class_ind = [Y0 == class_number for class_number in range(num_class)]
        # print("class_ind", class_ind)
        Y0 = np.asarray(np.hstack(class_ind), dtype=np.float32)

    # X = np.random.shuffle(X0)
    # Y = np.random.shuffle(Y0)

    return X0, Y0


# mean = np.array([2, 0])
# cov = np.array([[1, 0], [0, 1]])
# diff = np.array([3, 1])


num_class = 2
mean = np.random.randn(num_class) + 3
cov = np.eye(num_class)
X, Y = generate_samples(200, mean, cov, [3, 3], False)
print("==========generate samples as followings:")
print(X)
print(Y)

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
cross_entropy = -(input_labels * tf.log(output) + (1 - input_labels) * tf.log(1 - output))
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.04)
train = optimizer.minimize(loss)

max_epochs = 50
mini_batch_size = 20

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(max_epochs):

        for i in range(np.int32(len(Y) / mini_batch_size)):
            x1 = X[i * mini_batch_size:(i + 1) * mini_batch_size]
            y1 = np.reshape(Y[i * mini_batch_size:(i + 1) * mini_batch_size], [-1, 1])

            # print("x1:", x1)
            # print("y1:", y1)

            _, lossval, outputval = sess.run([train, loss, output], feed_dict={input_feature: x1, input_labels: y1})
            print("##done! lossval:", lossval, "outputval:", outputval)

        print("epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(lossval))
