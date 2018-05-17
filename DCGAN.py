from Generator import *
from Discriminator import *
import tensorflow as tf
import numpy as np
#import inception_model
import os
import matplotlib.pyplot as plt


class DCGAN():
    def __init__(self, input_shape, first_block_depth=1024, dim_noise=100, model="simple"):
        # Dimension of data
        self.output_side = input_shape[0]
        self.output_depth = input_shape[2]
        self.dim_noise = dim_noise
        # Build input variables
        self.X_batch = tf.placeholder(dtype=tf.float32, shape=[None, self.output_depth*self.output_side**2], name='X')

        self.noise_batch = tf.placeholder(dtype=tf.float32, shape=[None, self.dim_noise], name='noise')
        # Build both components
        self.discriminator = Discriminator(input_shape, first_block_depth, model=model)
        self.generator = Generator(input_shape, first_block_depth=512, model=model)
        # Construct the graph
        self.build_graph()

    def get_noise(self, batch_size, min_distri=-1, max_distri=1):
        return np.random.uniform(min_distri, max_distri, [batch_size, self.dim_noise]).astype('float32')

    def set_losses(self, real_images_logits, fake_images_logits):
        self.discriminator.set_loss(real_images_logits, fake_images_logits)
        self.generator.set_loss(fake_images_logits)

    def build_graph(self):
        # Variables
        fake_images = self.generator.generate_images(self.noise_batch)
        real_images_probabilities, real_images_logits = self.discriminator.compute_probability(self.X_batch)
        fake_images_probabilities, fake_images_logits = self.discriminator.compute_probability(fake_images)
        # Loss
        self.set_losses(real_images_logits, fake_images_logits)
        self.generator.set_solver()
        self.discriminator.set_solver()

    def optimize(self, sess, X_batch_values, size_noise, j, previous_D_loss, previous_G_loss, strategy="k_steps", k=1, gap=10):
        D_curr_loss = previous_D_loss
        G_curr_loss = previous_G_loss

        # Compute loss and optimize
        if strategy=="k_steps":
            noise_batch_values = self.get_noise(size_noise)
            _, D_curr_loss = sess.run([self.discriminator.solver, self.discriminator.loss],
                                      feed_dict={self.X_batch: X_batch_values, self.noise_batch: noise_batch_values})
            noise_batch_values = self.get_noise(size_noise)
            if j == 1 or j % k == 0:  # Improving G every k steps
                _, G_curr_loss = sess.run([self.generator.solver, self.generator.loss],
                                          feed_dict={self.noise_batch: noise_batch_values})
        elif strategy=="min_gap":
            if gap*previous_D_loss < previous_G_loss:
                noise_batch_values = self.get_noise(size_noise)
                _, G_curr_loss = sess.run([self.generator.solver, self.generator.loss],
                                          feed_dict={self.noise_batch: noise_batch_values})
            elif gap*previous_G_loss < previous_D_loss:
                noise_batch_values = self.get_noise(size_noise)
                _, D_curr_loss = sess.run([self.discriminator.solver, self.discriminator.loss],
                                          feed_dict={self.X_batch: X_batch_values, self.noise_batch: noise_batch_values})
            else:
                noise_batch_values = self.get_noise(size_noise)
                _, D_curr_loss = sess.run([self.discriminator.solver, self.discriminator.loss],
                                          feed_dict={self.X_batch: X_batch_values, self.noise_batch: noise_batch_values})
                noise_batch_values = self.get_noise(size_noise)
                _, G_curr_loss = sess.run([self.generator.solver, self.generator.loss], feed_dict={self.noise_batch: noise_batch_values})

        return D_curr_loss, G_curr_loss

    def train(self, X, n_epochs, batch_size, k=1, gap=5, strategy="k_steps"):
        D_curr_loss = 0
        G_curr_loss = 0
        # Initialize variables and Tensorboard
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('output_tensorboard', sess.graph)
            # Train
            max_j = int(np.ceil(int(X.shape[0])/ batch_size)) + 1
            for i in range(n_epochs):                
                print("Performing epoch " + str(i+1) + "/" + str(n_epochs) + '\n')
                for j in range(1, max_j):
                    # Indices of batch
                    j_start = (j - 1) * batch_size
                    j_end = j * batch_size
                    # Get data
                    X_batch_values = X[j_start:j_end]  # Shape (-1, n_dim)
                    # Compute loss and optimize
                    D_curr_loss, G_curr_loss = self.optimize(sess, X_batch_values, j_end - j_start, j, D_curr_loss,
                                                             G_curr_loss, strategy=strategy, k=k, gap=gap)
                    if j%100 == 0:
                        print(str(j) + '/' + str(max_j-1) + ' : cost D=' + str(D_curr_loss) + ' - cost G=' + str(G_curr_loss) + '\n')

                #if i%10 == 0:
                self.display_generated_images(sess, i)  # Store generated images after every 10 epochs

            # Value in image to low to compute inception score
            # mean, std = self.compute_inception_score(sess)
            # print('Programm ended with Inception score', mean)

    def compute_inception_score(self, sess):
        # Raise error if model not trained ?
        noise_batch_values = self.get_noise(100)
        images = sess.run(self.generator.generate_images(self.noise_batch, is_training=False), feed_dict={self.noise_batch: noise_batch_values})
        list_images = [np.append(np.array(image*127 + 128, dtype='int32').reshape((1, 28, 28)), np.zeros((2,28,28)), axis=0) for image in images]
        return inception_model.get_inception_score(list_images)  # mean, std

    def display_generated_images(self, sess, n_epoch, n_images=16):
        print("Display fake images")
        if not os.path.exists(os.path.join('generated_img', 'MNIST')):
            os.makedirs(os.path.join('generated_img', 'MNIST'))
        noise_batch_values = self.get_noise(n_images)
        faked_images = sess.run(self.generator.generate_images(self.noise_batch, is_training=False), feed_dict={self.noise_batch: noise_batch_values})
        displayable_images = [np.array(image*127 + 128, dtype='int32').reshape((28, 28)) for image in faked_images]
        fig = self.plot(displayable_images)
        plt.savefig(os.path.join('generated_img', 'MNIST', 'Epoch' + str(n_epoch) + '.png'))
        plt.close(fig)

    @staticmethod
    def plot(samples):
        size = np.sqrt(len(samples))
        assert (np.ceil(size) == size), "change image number"
        size = int(size)
        fig, axes = plt.subplots(size, size, figsize=(7,7))
        i = 0
        for j in range(size):
            for k in range(size):
                axes[j][k].set_axis_off()
                axes[j][k].imshow(samples[i], cmap='Greys_r')
                i += 1

        return fig
