from Generator import *
from Discriminator import *
import tensorflow as tf
import numpy as np
from Tools import inception_model
import os
import matplotlib.pyplot as plt


class DCGAN():
    def __init__(self, input_shape, depth_layers_discriminator=[64, 128, 256, 512], depth_layers_generator=[1024, 512, 256, 128],
                 dim_noise=100, model="simple", data="MNIST", flip_discri_labels=False, final_generator_activation="tanh"):
        # Global model
        self.model = model
        self.data = data
        # Dimension of data
        self.output_height = input_shape[0]
        self.output_width = input_shape[1]
        self.output_depth = input_shape[2]
        self.dim_noise = dim_noise
        # Useful variables
        self.real_images_probabilities = 0
        self.fake_images_probabilities = 0
        # Build input variables
        self.X_batch = tf.placeholder(dtype=tf.float32, shape=[None, self.output_depth*self.output_height*self.output_width], name='X')
        self.noise_batch = tf.placeholder(dtype=tf.float32, shape=[None, self.dim_noise], name='noise')
        # Build both components
        self.final_generator_activation = final_generator_activation
        self.discriminator = Discriminator(input_shape, depth_layers_discriminator, model=model)
        self.generator = Generator(input_shape, depth_layers=depth_layers_generator, model=model, data=data, final_activation=final_generator_activation)
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
        self.real_images_probabilities, real_images_logits = self.discriminator.compute_probability(self.X_batch)
        self.fake_images_probabilities, fake_images_logits = self.discriminator.compute_probability(fake_images)
        # Loss
        self.set_losses(real_images_logits, fake_images_logits, flip_discri_labels=False)
        self.generator.set_solver()
        self.discriminator.set_solver()
        # Summaries
        tf.summary.scalar("Loss Generator", self.generator.loss)
        tf.summary.scalar("Loss Discriminator", self.discriminator.loss)
        tf.summary.scalar("Probability real images", self.real_images_probabilities)
        tf.summary.scalar("Probability fake images", self.fake_images_probabilities)

    def optimize(self, sess, X_batch_values, size_noise, j, previous_D_loss, previous_G_loss, was_D_trained, was_G_trained, strategy="k_steps", k=1, gap=10, noise_type="uniform"):
        D_curr_loss = previous_D_loss
        G_curr_loss = previous_G_loss
        train_D = True  # Used only for strategy="probabilities"
        train_G = True

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

        elif strategy=="probabilities":
            # Here loss is probability
            if 0.7<D_curr_loss and was_D_trained:
                train_D = False
            elif G_curr_loss>0.7 and was_G_trained:
                train_G = False
            if train_G:
                noise_batch_values = self.get_noise(size_noise, distribution=noise_type)
                _, G_curr_loss = sess.run([self.generator.solver, self.fake_images_probabilities],
                                          feed_dict={self.noise_batch: noise_batch_values})
                G_curr_loss = np.mean(G_curr_loss)
            if train_D:
                noise_batch_values = self.get_noise(size_noise, distribution=noise_type)
                _, D_curr_loss = sess.run([self.discriminator.solver, self.real_images_probabilities],
                                          feed_dict={self.X_batch: X_batch_values,
                                                     self.noise_batch: noise_batch_values})
                D_curr_loss = np.mean(D_curr_loss)

        return D_curr_loss, G_curr_loss, train_D, train_G

    def train(self, X, n_epochs, batch_size, k=1, gap=5, strategy="k_steps", is_data_normalized=False,
              is_inception_score_computed=False, is_model_saved=False, noise_type="uniform"):
        # Normalize input data
        if not is_data_normalized:
            if self.final_generator_activation=="sigmoid":
                X = X/255
            elif self.final_generator_activation=="tanh":
                X = (X - 127.5) / 127.5
        # Initialize variables and Tensorboard
        D_curr_loss = 0
        G_curr_loss = 0
        D_trained = False
        G_trained = False
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
                    D_curr_loss, G_curr_loss, D_trained, G_trained = self.optimize(sess, X_batch_values, j_end - j_start, j, D_curr_loss,
                                                             G_curr_loss, D_trained, G_trained, strategy=strategy, k=k, gap=gap, noise_type=noise_type)
                    if j%show_loss_every == 0:
                        print(str(j) + '/' + str(max_j-1) + ' : cost D=' + str(D_curr_loss) + ' - cost G=' + str(G_curr_loss) + '\n')

                # Store generated images after each epoch
                self.display_generated_images(sess, i+1, noise_type=noise_type)
                
                # Compute inception score
                if is_inception_score_computed:
                    mean, std = self.compute_inception_score(sess, noise_type=noise_type)
                    inception_scores.append(mean)
                    print('Inception score', mean)

            if is_model_saved:
                self.save_model(sess, strategy)

            self.display_generated_images(sess, 'final_', n_images=100, noise_type=noise_type)
            mean, std = self.compute_inception_score(sess, noise_type=noise_type)
            inception_scores.append(mean)
            print('Inception score', mean)
            self.save_inception_score(inception_scores)

    def save_inception_score(self, inception_scores):
        # Saving inception score
        saving_directory = os.path.join('save', self.model, self.data, 'inception_score')
        if not os.path.exists(saving_directory):
            os.makedirs(saving_directory)
        file_path = os.path.join(saving_directory, 'incep_score_per_epoch')
        np.save(file_path, inception_scores)

    def save_model(self, sess, strategy):
        saver = tf.train.Saver()
        # Saving model
        model_saved_location = os.path.join('save', self.model, self.data, 'model')
        if not os.path.exists(model_saved_location):
            os.makedirs(model_saved_location)

        path = saver.save(sess, os.path.join(model_saved_location, strategy + '_'))
        print("Model saved in path: %s\n" % path)

    def compute_inception_score(self, sess, noise_type="uniform"):
        # Raise error if model not trained ?
        # Attention, fonctionne ici pour dcgan et tanh, pas pour les autres et sigmoid
        noise_batch_values = self.get_noise(100, distribution=noise_type)
        images = sess.run(self.generator.generate_images(self.noise_batch, is_training=False), feed_dict={self.noise_batch: noise_batch_values})
        if self.data == 'MNIST':
            list_images = [np.append(np.array(image*127 + 128, dtype='int32').reshape((28, 28, 1)), np.zeros((28,28,2)), axis=2) for image in images]
        if self.data == 'CIFAR10':
            list_images = [np.array(image*127 + 128, dtype='int32') for image in images]
        if self.data == 'CelebA':
            list_images = [np.array(image*127 + 128) for image in images]
        return inception_model.get_inception_score(list_images)  # mean, std

    def display_generated_images(self, sess, n_epoch, n_images=25, noise_type="uniform"):
        print("Display generated image")
        if self.data == 'MNIST':
            if not os.path.exists(os.path.join('generated_img', 'MNIST')):
                os.makedirs(os.path.join('generated_img', 'MNIST'))
            noise_batch_values = self.get_noise(n_images, distribution=noise_type)
            faked_images = sess.run(self.generator.generate_images(self.noise_batch), feed_dict={self.noise_batch: noise_batch_values})
            #if self.final_generator_activation=="sigmoid":
            #    displayable_images = [np.array(image * 255, dtype='int32').reshape((28, 28)) for image in faked_images]
            #elif self.final_generator_activation == "tanh":
            displayable_images = [np.array(image*127.5 + 127.5, dtype='int32').reshape((28, 28)) for image in faked_images]
            fig = self.plot(displayable_images)
            plt.savefig(os.path.join('generated_img', 'MNIST', 'Epoch' + str(n_epoch) + '.png'))
            plt.close(fig)

        elif self.data == 'CIFAR10':
            if not os.path.exists(os.path.join('generated_img', 'CIFAR10')):
                os.makedirs(os.path.join('generated_img', 'CIFAR10'))
            noise_batch_values = self.get_noise(n_images, distribution=noise_type)
            faked_images = sess.run(self.generator.generate_images(self.noise_batch), feed_dict={self.noise_batch: noise_batch_values})
            if self.final_generator_activation=="sigmoid":
                displayable_images = (np.reshape(faked_images, (-1, 3, 32, 32)).transpose(0, 2, 3, 1))
            elif self.final_generator_activation == "tanh":
                displayable_images = (np.reshape(faked_images, (-1, 3, 32, 32)).transpose(0, 2, 3, 1) * 0.5 + 0.5)
            fig = self.plot(displayable_images)
            plt.savefig(os.path.join('generated_img', 'CIFAR10', 'Epoch' + str(n_epoch) + '.png'))
            plt.close(fig)
            
    @staticmethod
    def plot(samples):
        size = np.sqrt(len(samples))
        assert (np.ceil(size) == size), "change image number"
        size = int(size)
        fig, axes = plt.subplots(size, size, figsize=(7, 7))
        plt.subplots_adjust(wspace=0, hspace=0)
        i = 0
        for j in range(size):
            for k in range(size):
                axes[j][k].set_axis_off()
                axes[j][k].imshow(samples[i], cmap='Greys_r')
                i += 1
        return fig
