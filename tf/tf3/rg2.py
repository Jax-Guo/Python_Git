import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn.preprocessing as prep

s = tf.Session()
#ops.reset_default_graph()

iris = datasets.load_iris()
x_vals = np.array([x[0] for x in iris.data])
x_vals = x_vals/(max(x_vals)-min(x_vals))
y_vals = np.array([y[3] for y in iris.data])
y_vals = y_vals/(max(y_vals)-min(y_vals))

x_data = tf.placeholder(dtype=tf.float32,shape=[None,1])
y_target = tf.placeholder(dtype=tf.float32,shape=[None,1])

W = tf.Variable(tf.fill(dims=[1,1],value=1.))
b = tf.Variable(tf.fill(dims=[1,1],value=1.))
learning_rate = 0.05
batch_size = 10

my_out = tf.add(tf.matmul(x_data,W),b)
loss = tf.reduce_mean(tf.square(y_target-my_out))

my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
s.run(init)



loss_vec = []
for i in range(5000):
    rand_indices = np.random.choice(len(x_vals),size = batch_size)
    rand_x = np.transpose([x_vals[rand_indices]])
    rand_y = np.transpose([y_vals[rand_indices]])
    s.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
    my_mse = s.run(loss, feed_dict={x_data:rand_x, y_target:rand_y})
    loss_vec.append(my_mse)
    if (i + 1) % 250 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(s.run(W)) + ' b = ' + str(s.run(b)))
        print('Loss = ' + str(my_mse))

[slope] = s.run(W)
[y_intercept] = s.run(b)
best_fit = []
for i in x_vals:
    best_fit.append(slope*i+y_intercept)
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.show()