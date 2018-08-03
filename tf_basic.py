import tensorflow as tf
import numpy as np

# arr = np.array([2, 3])
# tensor=tf.ones_like(arr)
print("#####demo0#####")
tensor = tf.zeros([2, 3])
with tf.Session() as sess:
    print("tensor=", sess.run(tensor))
    print("shape(np):", np.shape(tensor))
    print("shape(tf):", sess.run(tf.shape(tensor)))
    print("rank(tf):", sess.run(tf.rank(tensor)))
    print("size(tf):", sess.run(tf.size(tensor)))

print("#####demo1#####")
t1 = [2, 3, 4, 5, 6, 7, 8, 9, 1]
tensor1 = tf.reshape(t1, [3, 3])
with tf.Session() as sess1:
    print("reshape:", sess1.run(tensor1))

print("#####demo2#####")
t2 = [[[1, 2]]]
print("t2: ", t2, "#shape:", np.shape(t2))
t2_a = tf.expand_dims(t2, 0)
t2_b = tf.expand_dims(t2, 1)
t2_c = tf.expand_dims(t2, -1)
with tf.Session() as sess2:
    print("t2_a: ", sess2.run(t2_a), "#shape:", np.shape(t2_a))
    print("t2_b: ", sess2.run(t2_b), "#shape:", np.shape(t2_b))
    print("t2_c: ", sess2.run(t2_c), "#shape:", np.shape(t2_c))

print("#####demo3#####")
t3 = [[[2], [1]]]  # (1, 2, 1)
print("t3: ", t3, "#shape:", np.shape(t3))
t3_a = tf.squeeze(t3, 0)
# t3_b = tf.squeeze(t3, 1) // dim must be 1 or error
t3_c = tf.squeeze(t3, -1)
with tf.Session() as sess3:
    print("t3_a: ", sess3.run(t3_a), "#shape:", np.shape(t3_a))
    # print("t3_b: ", sess2.run(t3_b), "#shape:", np.shape(t3_b))
    print("t3_c: ", sess3.run(t3_c), "#shape:", np.shape(t3_c))

print("#####demo4#####")
t4_1 = [[1, 2, 3, 4]]  # (1,4)
t4_2 = [[5, 6, 7, 8]]  # (4,)
t4_contact = tf.concat([t4_1, t4_2], 1)
with tf.Session() as sess4:
    print("concat:", sess4.run(t4_contact), "#shape:", np.shape(t4_contact))

print("#####demo5#####")
t5_1 = [[1, 2, 3], [1, 2, 3]]
t5_2 = [[4, 5, 6], [6, 7, 8]]
t5_stack = tf.stack([t5_1, t5_2], axis=-1)
with tf.Session() as sess5:
    print("stack:", sess5.run(t5_stack), "#shape:", np.shape(t5_stack))

print("#####demo6#####")
t6 = [[1, 2], [3, 4], [5, 6]]  # (3,2)
arr1 = tf.unstack(t6)
arr2 = tf.unstack(t6, axis=1)
with tf.Session() as sess6:
    print("unstack:", sess6.run(arr1), "#shape:", np.shape(arr1))
    print("unstack:", sess6.run(arr2), "#shape:", np.shape(arr2))

print("#####demo7#####")
t7 = tf.range(0, 10) * 10 + tf.constant(1, shape=[10])
t7_gather = tf.gather(t7, [1, 5, 9])
with tf.Session() as sess7:
    print(sess7.run(t7))
    print("gather:", sess7.run(t7_gather), "#shape:", np.shape(t7_gather))

print("#####demo8#####")
t8 = np.random.randint(0, 10, size=[10])
t8_one_hot = tf.one_hot(t8, 10, on_value=1, off_value=None, axis=-1)
with tf.Session() as sess8:
    print(t8)
    print("one_hot:", sess8.run(t8_one_hot), "#shape:", np.shape(t8_one_hot))
    print("nonzeros:", sess8.run(tf.count_nonzero(t8_one_hot)))

print("#####demo9#####")
t9=tf.random_normal([2,3])
with tf.Session() as sess9:
    print(sess9.run(t9))