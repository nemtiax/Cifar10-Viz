import scipy.misc
import numpy as np

import tensorflow as tf



class convnet(object):

    def __init__(self,batch_size=64,filter_size=5):
        self.batch_size = batch_size
        self.filter_size = filter_size
        self.input_images = tf.placeholder(tf.float32,shape=[None,32,32,3])
        self.batch_labels = tf.placeholder(tf.float32,shape=[None,10])

    def conv2d(self,input,in_channels,out_channels,non_linearity=tf.nn.relu):
        filters = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size,in_channels,out_channels]))
        biases = tf.Variable(tf.truncated_normal([out_channels]))
        return non_linearity(tf.nn.conv2d(input,filters,strides=[1, 1, 1, 1], padding='SAME')+biases)

    def max_pool(self,input,size=2):
        return tf.nn.max_pool(input,ksize=[1, size, size, 1],strides=[1, size, size, 1], padding='SAME')

    def fc(self,input,in_size,out_size,non_linearity=tf.nn.sigmoid):
        weights = tf.Variable(tf.truncated_normal([in_size,out_size]))
        biases = tf.Variable(tf.truncated_normal([out_size]))
        return non_linearity(tf.matmul(input,weights)+biases)

    def build(self):
        self.h0 = self.conv2d(self.input_images,3,32) #32x32x32
        self.h0_pool = self.max_pool(self.h0) #16x16x32
        self.h1 = self.conv2d(self.h0_pool,32,64) #16x16x64
        self.h1_pool = self.max_pool(self.h1) #8x8x64
        self.h2 = self.conv2d(self.h1_pool,64,128) #8x8x128
        self.h2_pool = self.max_pool(self.h2) #4x4x128

        self.h2_flat = tf.reshape(self.h2_pool,[-1,4*4*128])
        self.fc_layer = self.fc(self.h2_flat,4*4*128,1024,tf.nn.relu)
        self.out = self.fc(self.fc_layer,1024,10,tf.identity)

    def build_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.out, self.batch_labels))
        correct_prediction = tf.equal(tf.argmax(self.out,1), tf.argmax(self.batch_labels,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def build_train_op(self):
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    #from cifar-10 site
    def unpickle(self,file):
        import cPickle
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    def load_training_data(self):
        self.images = np.empty([50000,32,32,3])
        self.labels = np.zeros([50000,10])

        for i in range(1,6):
            data_file = "cifar-10-batches-py/data_batch_" + str(i)
            data = self.unpickle(data_file)
            batch_labels = data['labels']
            batch_images = data['data'].reshape((-1,3,32,32))
            batch_images = batch_images.transpose([0,2,3,1])
            self.images[10000*(i-1):10000*i,:,:,:] = batch_images/256.0
            self.labels[np.arange(10000*(i-1),10000*i),batch_labels] = 1

        test_data_file = "cifar-10-batches-py/test_batch"
        test_data = self.unpickle(test_data_file)
        self.test_labels = np.zeros([10000,10])
        self.test_labels[np.arange(10000),test_data['labels']]=1
        self.test_images = test_data['data'].reshape((-1,3,32,32))
        self.test_images = self.test_images.transpose([0,2,3,1])

    def get_batch(self,index):
        batch_images = self.images[self.batch_size*index:self.batch_size*(index+1),:,:,:]
        batch_labels = self.labels[self.batch_size*index:self.batch_size*(index+1),:]
        return batch_images,batch_labels

    #http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    def shuffle_data(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self.images)
        np.random.set_state(rng_state)
        np.random.shuffle(self.labels)

    def train(self,epochs,sess):
        tf.initialize_all_variables().run()
        count = 0
        for ep in range(epochs):
            self.shuffle_data()
            num_batches = 50000//self.batch_size
            for batch_index in range(num_batches):
                batch_images,batch_labels = self.get_batch(batch_index)
                sess.run(self.train_op,feed_dict={self.input_images: batch_images,self.batch_labels: batch_labels})
                count+=1
                if(count%100==0):
                    accuracy,loss = sess.run([self.accuracy,self.loss],feed_dict={self.input_images: self.test_images,self.batch_labels: self.test_labels})
                    print(accuracy)
                    print(loss)


with tf.Session() as sess:
    network = convnet()
    network.build()
    network.load_training_data()
    network.build_loss()
    network.build_train_op()
    network.train(100,sess)
