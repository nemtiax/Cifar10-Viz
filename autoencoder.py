import scipy.misc
import numpy as np
import math

import tensorflow as tf

class autoencoder(object):

    def __init__(self,batch_size=64,filter_size=5):
        self.batch_size = batch_size
        self.filter_size = filter_size
        self.input_images = tf.placeholder(tf.float32,shape=[self.batch_size,32,32,3])
    def build(self):
        f = 32
        self.enc0 = self.conv2d(self.input_images,3,f) #16x16x16
        self.enc1 = self.conv2d(self.enc0,f,2*f) #8x8x32
        self.enc2 = self.conv2d(self.enc1,2*f,4*f) #4x4x64
        self.enc3 = self.conv2d(self.enc2,4*f,8*f) #2x2x128
        self.dec3 = self.deconv2d(self.enc3,[self.batch_size,4,4,4*f],8*f,4*f) #4x4x64
        self.dec2 = self.deconv2d(self.dec3,[self.batch_size,8,8,2*f],4*f,2*f) #8x8x32
        self.dec1 = self.deconv2d(self.dec2,[self.batch_size,16,16,f],2*f,f) #16x16x16
        self.dec0 = self.deconv2d(self.dec1,[self.batch_size,32,32,3],f,3, non_linearity=tf.nn.tanh) #32x32x3

    def build_loss(self):
        self.reconstruction_loss = tf.reduce_mean(tf.pow(self.input_images - self.dec0,2)) + tf.reduce_mean(tf.abs(self.input_images - self.dec0))

    def build_train_op(self):
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.reconstruction_loss)

    def conv2d(self,input,in_channels,out_channels,name='NONE',non_linearity=tf.nn.relu):
        filters = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size,in_channels,out_channels],stddev=0.01))
        biases = tf.Variable(tf.truncated_normal([out_channels],stddev=0.01))
        return non_linearity(tf.nn.conv2d(input,filters,strides=[1, 2, 2, 1], padding='SAME')+biases)

    def deconv2d(self,input,output_shape,in_channels,out_channels,name='NONE',non_linearity=tf.nn.relu):
        filters = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size,out_channels,in_channels],stddev=0.01))
        biases = tf.Variable(tf.truncated_normal([out_channels],stddev=0.01))
        return non_linearity(tf.nn.conv2d_transpose(input,filters,strides=[1,2,2,1],output_shape=output_shape,padding='SAME')+biases)

    def max_pool(self,input,size=2):
        return tf.nn.max_pool(input,ksize=[1, size, size, 1],strides=[1, size, size, 1], padding='SAME')

    def fc(self,input,in_size,out_size,non_linearity=tf.nn.sigmoid):
        weights = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.01))
        biases = tf.Variable(tf.truncated_normal([out_size],stddev=0.01))
        return non_linearity(tf.matmul(input,weights)+biases)

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
            self.images[10000*(i-1):10000*i,:,:,:] = batch_images/128.0 - 1
            #instead of 0-9, we want one-hot vectors, so a label of '3' becomes (0,0,0,1,0,0,0,0,0,0)
            self.labels[np.arange(10000*(i-1),10000*i),batch_labels] = 1

        #same deal to load in test data
        test_data_file = "cifar-10-batches-py/test_batch"
        test_data = self.unpickle(test_data_file)
        self.test_labels = np.zeros([10000,10])
        self.test_labels[np.arange(10000),test_data['labels']]=1
        self.test_images = test_data['data'].reshape((-1,3,32,32))
        self.test_images = self.test_images.transpose([0,2,3,1]) / 128.0 - 1

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

        sample_batch,sample_labels = self.get_batch(0)
        sample_batch = np.copy(sample_batch)

        count = 0
        for ep in range(epochs):
            self.shuffle_data()
            num_batches = 50000//self.batch_size
            for batch_index in range(num_batches):
                batch_images,batch_labels = self.get_batch(batch_index)
                _,loss,reconstructed_images = sess.run([self.train_op,self.reconstruction_loss,self.dec0],feed_dict={self.input_images: batch_images})
                count=count+1
                if(count%1000==0):
                    print(loss)
                    self.save_samples(batch_images,reconstructed_images,count)

    def save_samples(self,batch_images,reconstructed_images,count):
        output_arr = np.full((68*8,36*8,3),-1.)

        for x in range(0,8):
            for y in range(0,8):
                output_arr[68*x+2:68*x+34,36*y+2:36*y+34,:] = batch_images[8*x+y]
                output_arr[68*x+34:68*x+66,36*y+2:36*y+34,:] = reconstructed_images[8*x+y]


        image = scipy.misc.toimage(output_arr,cmin=-1,cmax=1)
        image.save("{:05d}_results.png".format(count));

with tf.Session() as sess:
    network = autoencoder()
    network.build()
    network.load_training_data()
    network.build_loss()
    network.build_train_op()
    network.train(500,sess)
