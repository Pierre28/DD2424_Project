from Generator import *
from Discriminator import *
import tensorflow as tf
import numpy as np

class DCGAN():
    def __init__(self, input_shape, first_block_depth=1024, dim_noise=100):
        self.output_side = input_shape[0]
        self.output_depth = input_shape[2]
        self.dim_noise = dim_noise

        self.X_batch = tf.placeholder(dtype=tf.float32, shape=[None, self.output_side, self.output_side, self.output_depth], name='X')
        self.noise_batch = tf.placeholder(dtype=tf.float32, shape=[None, self.dim_noise], name='noise')
        self.proba_discriminator = tf.placeholder(dtype=tf.float32, name='proba_discriminator')
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        self.discriminator = Discriminator(input_shape, first_block_depth)
        self.generator = Generator(input_shape, first_block_depth=512) 

    def get_noise(self, batch_size, min_distri=-1, max_distri=1):
        self.noise_batch = tf.random_uniform([batch_size, self.dim_noise], min_distri, max_distri)

    def update_loss(self, real_logits, fake_logits, probability_fake_images):
        self.discriminator.update_loss(real_logits, fake_logits)
        self.generator.update_loss(probability_fake_images)

    def initialize_variables(self):
        # Variables
        self.get_noise(1)
        self.X_batch = tf.random_uniform([1, 64, 64, 3], -1, 1)
        ini_fake_image = self.generator.forward_pass(self.noise_batch)
        ini_proba_fake, ini_logits_fake = self.discriminator.forward_pass(ini_fake_image)
        #Il faut aller chercher les logits d'une vraie image.. Pour débugger je prends ceux d'une fausse
        ini_proba_real, ini_logits_real = self.discriminator.forward_pass(ini_fake_image)
        # Loss
        self.update_loss(real_logits = ini_logits_real, fake_logits = ini_logits_fake, probability_fake_images = ini_proba_fake)
        self.generator.initialize_variables()
        self.discriminator.initialize_variables()
        # Computation graph
        writer = tf.summary.FileWriter('.')
        writer.add_graph(tf.get_default_graph())

    def train(self, X, n_epochs, batch_size):
        X = tf.convert_to_tensor(X)
        X = tf.reshape(X, shape=[-1, self.output_side, self.output_side, self.output_depth])
        # Initialize variables and graph
        with tf.Session() as sess:
            self.initialize_variables()
            sess.run(tf.global_variables_initializer())
            # Train
            max_j = int(np.ceil(X.shape[1] / batch_size)) + 1
            for i in range(n_epochs):
                print("Performing epoch " + str(i+1) + "/" + str(n_epochs))
                for j in range(1, max_j):
                    print("Performing sub-epoch " + str(j) + "/" + str(max_j-1))
                    j_start = (j - 1) * batch_size
                    j_end = j * batch_size
                    self.X_batch = X[j_start:j_end]
                    self.noise_batch = self.get_noise(j_end-j_start)

                    sess.run(self.discriminator.optimizer, feed_dict={"X_real": self.X_batch, "noise": self.noise_batch})
                    sess.run(self.generator.optimizer, feed_dict={"noise": self.noise_batch})
