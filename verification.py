import tensorflow as tf 
import numpy as np 
import cv2




class model():

   # Create the model
  def __init__(self):
    self.x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])


    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')


    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(self.x, [-1,28,28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2


    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    self.prediction = tf.nn.softmax(y_conv)

model_graph = tf.Graph()

with model_graph.as_default():

  model = model()

sess = tf.Session(graph=model_graph)

with sess.as_default():

  with model_graph.as_default():

    tf.global_variables_initializer().run()

    saver = tf.train.Saver(tf.global_variables())


try:

  saver.restore(sess, 'model.ckpt')
  print("Successfully loaded")

except:

  print("Error loading model")


#right now it's hardcoded here
image = cv2.imread('test.jpg', 0)
image = cv2.resize(image, (28,28))

numpy_input = np.resize(image, (1, 784))

prediction_mnist = sess.run(model.prediction, feed_dict={model.x : numpy_input })

result = np.argmax(prediction_mnist) 

print(result)