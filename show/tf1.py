###与Spark相似，tensorflow也是构建一个有向无环的工作流，通过创建常量、变量、占位量，并定义三者之间的关系，最后放入工作流执行

import numpy as np
import tensorflow as t
#唯有将常量、变量、占位量放入session执行，才会有结果，类似于spark中的transform和show，可存放超参数
s = t.Session()

##创建常量，常量一旦创建，别无法改变

t.zeros([2,2])
t.ones([2,2])
t.fill(1,[2,2])
t.range(1.0,2,3)
t.constant(43,shape=[10,10])
c = t.constant([1,2,3,4,5,6],shape=[2,3])
s.run(c)
#random_shuffle可将矩阵的行随即调换位置
t.random_shuffle(c)

#均匀分布
t.random_uniform([3,3],1,10)
#高斯分布
t.random_normal([3,3,3])
#截断高斯，即与中心相距不超过2个标准差的区域
t.truncated_normal([2, 3])

#将np.array转为tensor.array
a = np.random.rand(3,3)
at = t.convert_to_tensor(a)

##创建变量,变量在每次迭代中都会被改变，需要由初始值和初始化，可存放训练参数和每层训练后的临时结果
#设初始值
t.Variable(c)
v1 = t.Variable(at,name="at")
#初始化
v2 = t.Variable(v1)
s.run(v2.initializer)
s.run(v2)
#全局初始化，所有变量都在初始化池中，可单个初始化，也可全部初始化
init_op = t.global_variables_initializer()
s.run(init_op)

#占位量，只需指定类型和维度，无需初始化，用于存放输出值
x = t.placeholder(t.float32,[2,2])
y = t.identity(x-1)
x_vals = np.random.rand(2,2)
s.run(y,feed_dict={x:x_vals})

##矩阵操作
#定义对角矩阵
identity_matrix = t.diag([1.0, 1.0, 1.0])

A = t.truncated_normal([2, 3])
B = t.fill([2,3], 5.0)
C = t.random_uniform([3,2])
D = t.convert_to_tensor(np.array([[1., 2., 3.],[-3., -7.,
-1.],[0., 5., -2.]]))
print(s.run(identity_matrix))

#加
s.run(A+B)
#点乘
s.run(t.matmul(B,identity_matrix*2))
#转置
s.run(t.transpose(D))
#行列式
s.run(t.matrix_determinant(D))
#求逆
s.run(t.matrix_inverse(D))
#用cholesky求特征值和特征向量
s.run(t.self_adjoint_eig(D))

#除
s.run(t.div(3,4))

sr = s.run
#将int转为float后除
sr(t.truediv(3,4))
#反转类型
sr(t.floordiv(3.0,4.0))
#求余
sr(t.mod(22.0,3))
#叉乘
sr(t.cross([1.0,0.,0.],[0.,1.,0.]))
#自定义操作
def custom_polynomial(value):
    return(t.subtract(3 * t.square(value), value) + 10)
print(s.run(custom_polynomial(11)))

#激活函数
s.run(t.nn.relu([-3,0,3]))
s.run(t.nn.relu6([-3,0,3]))
s.run(t.nn.sigmoid([1.0,2.0,3]))
s.run(t.nn.tanh([1.0,2.0,3]))
s.run(t.nn.softsign([1.0,2.0,3]))
s.run(t.nn.softplus([1.0,2.0,3]))
s.run(t.nn.elu(np.array([-1,0,1],dtype=float)))

#读入iris
from sklearn import datasets
iris = datasets.load_iris()
print(len(iris.data),iris.data[1:20])

#读入mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(len(mnist.train.images),len(mnist.test.images),len(mnist.validation.images))
mnist.train.labels[1,:]