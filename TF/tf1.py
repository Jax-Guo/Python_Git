import tensorflow as tf
import numpy as np

np.random.seed(1234)
x = np.random.rand(100).astype(np.float32)
y = x*0.1 + 0.3

b = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random_uniform([1],-1.0, 1.0))
l = w*x + b

loss = tf.reduce_mean(tf.square(y-l))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in  range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(w),sess.run(b))