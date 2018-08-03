import tensorflow as tf

print("#####demo softmax_cross_entropy_with_logits#####")
labels = [[0, 0, 1], [0, 1, 0]]
logits = [[2, 0.5, 6], [0.1, 0, 3]]

logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)

result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)
result3 = -tf.reduce_sum(labels * tf.log(logits_scaled), 1)

with tf.Session() as sess:
    print("scaled=", sess.run(logits_scaled))
    print("scaled2=", sess.run(logits_scaled2))

    print("result1=", sess.run(result1))
    print("result2=", sess.run(result2))
    print("result3=", sess.run(result3))

print("#####demo sparse_softmax_cross_entropy_with_logits#####")
# indicates 3 classes, 0, 1, 2
labels_sparse = [2, 1]
result_sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_sparse, logits=logits)
with tf.Session() as sess:
    print("result=", sess.run(result_sparse))
