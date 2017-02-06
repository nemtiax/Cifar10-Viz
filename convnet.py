import scipy.misc
import numpy as np

import tensorflow as tf



class convnet(object):

    def __init__(self,batch_size=64,filter_size=5):
        self.batch_size = batch_size
        self.filter_size = filter_size
        self.input_images = tf.placeholder(tf.float32,shape=[None,32,32,3])
        self.batch_labels = tf.placeholder(tf.float32,shape=[None,10])
        self.kp = tf.placeholder(tf.float32)

    def conv2d(self,input,in_channels,out_channels,non_linearity=tf.nn.relu):
        filters = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size,in_channels,out_channels],stddev=0.01))
        biases = tf.Variable(tf.truncated_normal([out_channels],stddev=0.01))
        return non_linearity(tf.nn.conv2d(input,filters,strides=[1, 1, 1, 1], padding='SAME')+biases)

    def max_pool(self,input,size=2):
        return tf.nn.max_pool(input,ksize=[1, size, size, 1],strides=[1, size, size, 1], padding='SAME')

    def fc(self,input,in_size,out_size,non_linearity=tf.nn.sigmoid):
        weights = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.01))
        biases = tf.Variable(tf.truncated_normal([out_size],stddev=0.01))
        return non_linearity(tf.matmul(input,weights)+biases)

    def build(self):
        self.h0 = self.conv2d(self.input_images,3,64) #32x32x64
        self.h0_pool = self.max_pool(self.h0) #16x16x64
        self.h1 = self.conv2d(self.h0_pool,64,256) #16x16x256
        self.h1_pool = self.max_pool(self.h1) #8x8x256
        self.h2 = self.conv2d(self.h1_pool,256,1024) #8x8x1024
        self.h2_pool = self.max_pool(self.h2) #4x4x1024

        self.h2_flat = tf.reshape(self.h2_pool,[-1,4*4*1024])
        self.fc_layer = tf.nn.dropout(self.fc(self.h2_flat,4*4*1024,1024,tf.nn.relu),keep_prob=self.kp)
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

    #load up the cifar-10 training data and test data, both images and labels
    def load_training_data(self):
        self.images = np.empty([50000,32,32,3])
        self.labels = np.zeros([50000,10])

        for i in range(1,6):
            data_file = "cifar-10-batches-py/data_batch_" + str(i)
            data = self.unpickle(data_file)
            #labels is a list of 0-9 values saying which class each sample is
            batch_labels = data['labels']

            #images start as a bunch of 3072-size vectors, we need to reshape them into 32x32x3 images
            #the order the data is stored in results in 3x32x32 tensors, so we need to use transpose to shuffle the dimensions
            batch_images = data['data'].reshape((-1,3,32,32))
            batch_images = batch_images.transpose([0,2,3,1])
            #divide by 256 to normalize the data to [0,1]
            self.images[10000*(i-1):10000*i,:,:,:] = batch_images/256.0
            #instead of 0-9, we want one-hot vectors, so a label of '3' becomes (0,0,0,1,0,0,0,0,0,0)
            self.labels[np.arange(10000*(i-1),10000*i),batch_labels] = 1

        #same deal to load in test data
        test_data_file = "cifar-10-batches-py/test_batch"
        test_data = self.unpickle(test_data_file)
        self.test_labels = np.zeros([10000,10])
        self.test_labels[np.arange(10000),test_data['labels']]=1
        self.test_images = test_data['data'].reshape((-1,3,32,32))
        self.test_images = self.test_images.transpose([0,2,3,1]) / 256.0

    #fetch a batch of training data, index tells us which batch we're getting
    def get_batch(self,index):
        batch_images = self.images[self.batch_size*index:self.batch_size*(index+1),:,:,:]
        batch_labels = self.labels[self.batch_size*index:self.batch_size*(index+1),:]
        return batch_images,batch_labels

    #http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    #We want to randomize the order of the training data in each epoch, but we have to be careful
    #to shuffle the labels and the samples in the same way, or else we'll just have nonsense
    def shuffle_data(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self.images)
        np.random.set_state(rng_state)
        np.random.shuffle(self.labels)

    def train(self,epochs,sess):
        tf.global_variables_initializer().run()
        count = 0
        for ep in range(epochs):
            self.shuffle_data()
            num_batches = 50000//self.batch_size
            for batch_index in range(num_batches):
                batch_images,batch_labels = self.get_batch(batch_index)
                _,loss,output = sess.run([self.train_op,self.loss,self.out],feed_dict={self.input_images: batch_images,self.batch_labels: batch_labels,self.kp: 0.5})
                count+=1
                if(count%100==0):
                    accuracy= sess.run(self.accuracy,feed_dict={self.input_images: self.test_images,self.batch_labels: self.test_labels,self.kp: 1})
                    print("Epoch {:3d}, batch {:3d} - Accuracy={:0.4f} Batch_Loss={:0.4f}".format(ep,batch_index,accuracy,loss))


with tf.Session() as sess:
    network = convnet()
    network.build()
    network.load_training_data()
    network.build_loss()
    network.build_train_op()
    network.train(100,sess)
