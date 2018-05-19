from Generator import *
from Discriminator import *
import tensorflow as tf
import numpy as np
from Tools import inception_model
import os
import matplotlib.pyplot as plt


class DCGAN():
    def __init__(self, input_shape, depth_layers_discriminator=[64, 128, 256, 512], depth_layers_generator=[1024, 512, 256, 128],
                 dim_noise=100, model="simple", data="MNIST", flip_discri_labels=False):
        # Global model
        self.model = model
        self.data = data
        # Dimension of data
        self.output_height = input_shape[0]
        self.output_width = input_shape[1]
        self.output_depth = input_shape[2]
        self.dim_noise = dim_noise
        # Build input variables
        self.X_batch = tf.placeholder(dtype=tf.float32, shape=[None, self.output_depth*self.output_height*self.output_width], name='X')
        self.noise_batch = tf.placeholder(dtype=tf.float32, shape=[None, self.dim_noise], name='noise')
        # Build both components
        self.discriminator = Discriminator(input_shape, depth_layers_discriminator, model=model)
        self.generator = Generator(input_shape, depth_layers=depth_layers_generator, model=model, data=data)
        # Construct the graph
        self.build_graph(flip_discri_labels=flip_discri_labels)

    def get_noise(self, batch_size, min_distri=-1, max_distri=1, distribution="uniform"):
        if distribution=="uniform":
            return np.random.uniform(min_distri, max_distri, [batch_size, self.dim_noise]).astype('float32')
        elif distribution=="gaussian":
            return np.random.normal(size=(batch_size, self.dim_noise))

    def set_losses(self, real_images_logits, fake_images_logits, flip_discri_labels=False):
        self.discriminator.set_loss(real_images_logits, fake_images_logits, flip_labels=flip_discri_labels)
        self.generator.set_loss(fake_images_logits)

    def build_graph(self, flip_discri_labels=False):
        # Variables
        fake_images = self.generator.generate_images(self.noise_batch)
        real_images_probabilities, real_images_logits = self.discriminator.compute_probability(self.X_batch)
        fake_images_probabilities, fake_images_logits = self.discriminator.compute_probability(fake_images)
        # Loss
        self.set_losses(real_images_logits, fake_images_logits, flip_discri_labels=False)
        self.generator.set_solver()
        self.discriminator.set_solver()
        # Summaries
        tf.summary.scalar("Loss Generator", self.generator.loss)
        tf.summary.scalar("Loss Discriminator", self.discriminator.loss)
        tf.summary.scalar("Probability real images", real_images_probabilities)
        tf.summary.scalar("Probability fake images", fake_images_probabilities)

    def optimize(self, sess, X_batch_values, size_noise, j, previous_D_loss, previous_G_loss, strategy="k_steps", k=1, gap=10, noise_type="uniform"):
        D_curr_loss = previous_D_loss
        G_curr_loss = previous_G_loss

        # Compute loss and optimize
        if strategy=="k_steps":
            noise_batch_values = self.get_noise(size_noise, distribution=noise_type)
            _, D_curr_loss = sess.run([self.discriminator.solver, self.discriminator.loss],
                                      feed_dict={self.X_batch: X_batch_values, self.noise_batch: noise_batch_values})
            noise_batch_values = self.get_noise(size_noise, distribution=noise_type)
            if j == 1 or j % k == 0:  # Improving G every k steps
                _, G_curr_loss = sess.run([self.generator.solver, self.generator.loss],
                                          feed_dict={self.noise_batch: noise_batch_values})
        elif strategy=="min_gap":
            if gap*previous_D_loss < previous_G_loss:
                noise_batch_values = self.get_noise(size_noise, distribution=noise_type)
                _, G_curr_loss = sess.run([self.generator.solver, self.generator.loss],
                                          feed_dict={self.noise_batch: noise_batch_values})
            elif gap*previous_G_loss < previous_D_loss:
                noise_batch_values = self.get_noise(size_noise, distribution=noise_type)
                _, D_curr_loss = sess.run([self.discriminator.solver, self.discriminator.loss],
                                          feed_dict={self.X_batch: X_batch_values, self.noise_batch: noise_batch_values})
            else:
                noise_batch_values = self.get_noise(size_noise, distribution=noise_type)
                _, D_curr_loss = sess.run([self.discriminator.solver, self.discriminator.loss],
                                          feed_dict={self.X_batch: X_batch_values, self.noise_batch: noise_batch_values})
                noise_batch_values = self.get_noise(size_noise, distribution=noise_type)
                _, G_curr_loss = sess.run([self.generator.solver, self.generator.loss], feed_dict={self.noise_batch: noise_batch_values})

        return D_curr_loss, G_curr_loss

    def train(self, X, n_epochs, batch_size, k=1, gap=5, strategy="k_steps", is_data_normalized=False,
              is_inception_score_computed=False, is_model_saved=False, noise_type="uniform"):
        # Normalize input data
        if not is_data_normalized:
            if self.model=="simple":
                X = X/255
            else:
                X = (X - 127.5) / 127.5
        # Initialize variables and Tensorboard
        D_curr_loss = 0
        G_curr_loss = 0
        inception_scores = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('output_tensorboard', sess.graph)
            merged_summaries = tf.summary.merge_all()
            # Train
            max_j = int(np.ceil(int(X.shape[0])/ batch_size)) + 1
            show_loss_every = 100
            if max_j<100:
                show_loss_every = 10
            for i in range(n_epochs):
                print("Performing epoch " + str(i+1) + "/" + str(n_epochs) + '\n')
                for j in range(1, max_j):
                    # Indices of batch
                    j_start = (j - 1) * batch_size
                    j_end = j * batch_size
                    # Get data
                    X_batch_values = X[j_start:j_end, :]  # Shape (-1, n_dim)
                    # Compute loss and optimize
                    D_curr_loss, G_curr_loss = self.optimize(sess, X_batch_values, j_end - j_start, j, D_curr_loss,
                                                             G_curr_loss, strategy=strategy, k=k, gap=gap, noise_type=noise_type)
                    if j%show_loss_every == 0:
                        print(str(j) + '/' + str(max_j-1) + ' : cost D=' + str(D_curr_loss) + ' - cost G=' + str(G_curr_loss) + '\n')

                # Store generated images after each epoch
                self.display_generated_images(sess, i+1, noise_type=noise_type)
                # Saving inception score
                if is_inception_score_computed:
                    self.save_inception_score(sess, inception_scores, noise_type, i+1)
                # Saving model
                if is_model_saved:
                    self.save_model(sess, i+1, strategy)
                
            # Value in image to low to compute inception score
            mean, std = self.compute_inception_score(sess, noise_type=noise_type)
            print('Programm ended with Inception score', mean)

    def save_inception_score(self, sess, inception_scores, noise_type, i):
        # Saving inception score
        current_inception_score = self.compute_inception_score(sess, noise_type=noise_type)
        print('\nInception score : ', current_inception_score, '\n')
        inception_scores += [current_inception_score, i]
        saving_directory = os.path.join('save', self.model, self.data, 'inception_score')
        if not os.path.exists(saving_directory):
            os.makedirs(saving_directory)
        file_path = os.path.join(saving_directory, 'incep_score_epoch' + str(i))
        np.save(file_path, inception_scores)

    def save_model(self, sess, i, strategy):
        saver = tf.train.Saver()
        # Saving model
        model_saved_location = os.path.join('save', self.model, self.data, 'model')
        if not os.path.exists(model_saved_location):
            os.makedirs(model_saved_location)

        path = saver.save(sess, os.path.join(model_saved_location, strategy + '_epoch'), global_step=i)
        print("Model saved in path: %s\n" % path)

    def compute_inception_score(self, sess, noise_type="uniform"):
        # Raise error if model not trained ?
        # Attention, fonctionne ici pour dcgan et tanh, pas pour les autres et sigmoid
        noise_batch_values = self.get_noise(10, distribution=noise_type)
        images = sess.run(self.generator.generate_images(self.noise_batch, is_training=False), feed_dict={self.noise_batch: noise_batch_values})
        if self.data == 'MNIST':
            list_images = [np.append(np.array(image*127 + 128, dtype='int32').reshape((28, 28, 1)), np.zeros((28,28,2)), axis=2) for image in images]
        if self.data == 'CIFAR10':
            list_images = [np.array(np.tranpose(image*127 + 128, (1, 2, 0)), dtype='int32') for image in images]
        if self.data == 'CelebA':
            list_images = [np.array(image*127 + 128) for image in images]
        return inception_model.get_inception_score(list_images)  # mean, std

    def display_generated_images(self, sess, n_epoch, n_images=16, noise_type="uniform"):
        print("Display generated image")
        if self.data == 'MNIST':
            if not os.path.exists(os.path.join('generated_img', 'MNIST')):
                os.makedirs(os.path.join('generated_img', 'MNIST'))
            noise_batch_values = self.get_noise(n_images, distribution=noise_type)
            faked_images = sess.run(self.generator.generate_images(self.noise_batch), feed_dict={self.noise_batch: noise_batch_values})
            if self.model == "simple":
                displayable_images = [np.array(image * 127 + 128, dtype='int32').reshape((28, 28)) for image in faked_images]
            else:
                displayable_images = [np.array(image*127.5 + 127.5, dtype='int32').reshape((28, 28)) for image in faked_images]
            fig = self.plot(displayable_images)
            plt.savefig(os.path.join('generated_img', 'MNIST', 'Epoch' + str(n_epoch) + '.png'))
            plt.close(fig)

        elif self.data == 'CIFAR10':
            if not os.path.exists(os.path.join('generated_img', 'CIFAR10')):
                os.makedirs(os.path.join('generated_img', 'CIFAR10'))
            noise_batch_values = self.get_noise(n_images, distribution=noise_type)
            faked_images = sess.run(self.generator.generate_images(self.noise_batch), feed_dict={self.noise_batch: noise_batch_values})
            if self.model == "simple":
                displayable_images = (np.reshape(faked_images, (-1, 3, 32, 32)).transpose(0, 2, 3, 1)*255).astype(int)
            else:
                displayable_images = (np.reshape(faked_images, (-1, 3, 32, 32)).transpose(0, 2, 3, 1)*127.5 + 127.5).astype(int)
            fig = self.plot(displayable_images)
            plt.savefig(os.path.join('generated_img', 'CIFAR10', 'Epoch' + str(n_epoch) + '.png'))
            plt.close(fig)
            
    @staticmethod
    def plot(samples):
        size = np.sqrt(len(samples))
        assert (np.ceil(size) == size), "change image number"
        size = int(size)
        fig, axes = plt.subplots(size, size, figsize=(7, 7))
        i = 0
        for j in range(size):
            for k in range(size):
                axes[j][k].set_axis_off()
                axes[j][k].imshow(samples[i], cmap='Greys_r')
                i += 1
        return fig
