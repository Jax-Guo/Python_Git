###与Spark相似，tensorflow也是构建一个有向无环的工作流，通过创建常量、变量、占位量，并定义三者之间的关系，最后放入工作流执行

import tensorflow as tf
import numpy as np

#唯有将常量、变量、占位量放入session执行，才会有结果，类似于spark中的transform和show，可存放超参数
s = tf.Session()
sr = s.run

##创建一个简单工作流，并用tensorboard可视化工作流
#启动tensorboard -> tensorboard --logdir =C:/Users/BFD-725/Anaconda3/Lib/site-packages/tensorboard/log， 其中logdir需要与writer中保持一致
x_vals = np.array([1.,3.,5.,7.,9.])

#指定name后，tensorboard-graph中会显示
x_data = tf.placeholder(tf.float32,name='x_data')
m_const = tf.constant(3.,name='m_const')
my_product = tf.multiply(x_data,m_const,name='my_product')
for x_val in x_vals:
    print(sr(my_product,feed_dict={x_data:x_val}))
writer = tf.summary.FileWriter("C:/Users/BFD-725/Anaconda3/Lib/site-packages/tensorboard/log",tf.get_default_graph())
writer.close()

##创建一个工作流，并用tensorboard可视化工作流
my_array = np.array([[1., 3., 5., 7., 9.],
[-2., 0., 2., 4., 6.],
[-6., -3., 0., 3., 6.]])
x_vals = np.array([my_array, my_array + 1])
x_data = tf.placeholder(tf.float32, shape=(3, 5),name="x_data")

m1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]],name='m1')
m2 = tf.constant([[2.]],name='m2')
a1 = tf.constant([[10.]],name='a1')

prod1 = tf.matmul(x_data, m1,name='prod1')
prod2 = tf.matmul(prod1, m2,name='prod2')
add1 = tf.add(prod2, a1,name='add1')

writer = tf.summary.FileWriter("C:/Users/BFD-725/Anaconda3/Lib/site-packages/tensorboard/log",tf.get_default_graph())
writer.close()
for x_val in x_vals:
    print(s.run(add1, feed_dict={x_data: x_val}))

##创建一个有卷积工作流，并用tensorboard可视化工作流，该工作流包含了一个卷基层
x_shape = [1,4,4,1]
x_val = np.random.uniform(size=x_shape)
x_data = tf.placeholder(tf.float32,shape=x_shape)
x_val

my_filter = tf.constant(0.25,shape=[2,2,1,1])
my_stride = [1,2,2,1]
sr(my_filter)

mov_avg_layer = tf.nn.conv2d(x_data,my_filter,my_stride,padding="SAME",name = "mov_avg_layer")
mov_avg_layer

def custom_layer(input_mat):
    ims = tf.squeeze(input_mat)
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    t1 = tf.matmul(A,ims)
    t2 = tf.add(t1,b)
    return(tf.sigmoid(t2))

with tf.name_scope("Custom_Layer") as  scope:
    custom_layer1 = custom_layer(mov_avg_layer)


writer = tf.summary.FileWriter("C:/Users/BFD-725/Anaconda3/Lib/site-packages/tensorboard/log",tf.get_default_graph())
writer.close()

sr(custom_layer1,feed_dict={x_data:x_val})

### conv2详细介绍
# tf.nn.conv2d是TensorFlow里面实现卷积的函数，参考文档对它的介绍并不是很详细，实际上这是搭建卷积神经网络比较核心的一个方法，非常重要
#
# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
#
# 除去name参数用以指定该操作的name，与方法有关的一共五个参数：
#
# 第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
#
# 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
#
# 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
#
# 第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
#
# 第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
#
# 结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。
#
# 那么TensorFlow的卷积具体是怎样实现的呢，用一些例子去解释它：
#
# 1.考虑一种最简单的情况，现在有一张3×3单通道的图像（对应的shape：[1，3，3，1]），用一个1×1的卷积核（对应的shape：[1，1，1，1]）去做卷积，最后会得到一张3×3的feature map
#
# 2.增加图片的通道数，使用一张3×3五通道的图像（对应的shape：[1，3，3，5]），用一个1×1的卷积核（对应的shape：[1，1，1，1]）去做卷积，仍然是一张3×3的feature map，这就相当于每一个像素点，卷积核都与该像素点的每一个通道做卷积。