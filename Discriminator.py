import tensorflow as tf


class Discriminator:
    def __init__(self, input_shape, depth_layers=[64, 128, 256, 512], model="simple"):
        """Discriminator model used by the DCGAN
        :param input_shape: format [height, width, depth]
        :param depth_layers: the depth of the different layers used by the discriminator, only for the dcgan models
        :param model: type of model to use (simple, intermediate, dcgan_custom, dcgan_vanilla)"""
        # Global model
        self.model = model
        # Dimension of data
        self.output_height = input_shape[0]
        self.output_width = input_shape[1]
        self.output_depth = input_shape[2]
        # Parameters of layer
        self.depth_layers = depth_layers
        # Necessary variables to build graph
        self.variables = []
        self.loss = 0
        self.solver = 0

    def compute_probability(self, image, reuse=tf.AUTO_REUSE, dropout_probability=0.5):
        """Compute the probability of the given batch of image of bein real."""
        # Reuse = tf.AUTO_REUSE necessary to initialize both fake and real image
        with tf.variable_scope("discriminator", reuse=reuse):
            if self.model=="simple":
                image = tf.contrib.layers.flatten(image)
                logits_of_real = tf.layers.dense(image, units=128, activation=tf.nn.relu)
                logits_of_real = tf.layers.dense(logits_of_real, units=1)
                proba_of_real = tf.nn.sigmoid(logits_of_real)

            elif self.model=="intermediate":
                image = tf.reshape(image, shape=[-1, self.output_height, self.output_width, self.output_depth])
                # Convolution 1
                logits_of_real = tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same')
                #logits_of_real = tf.layers.dropout(logits_of_real, dropout_probability, training=True)
                logits_of_real = tf.nn.leaky_relu(logits_of_real)
                # 2
                logits_of_real = tf.layers.conv2d(logits_of_real, kernel_size=5, filters=128, strides=1, padding='same')
                #logits_of_real = tf.layers.dropout(logits_of_real, dropout_probability, training=True)
                logits_of_real = tf.nn.leaky_relu(logits_of_real)
                logits_of_real = tf.layers.batch_normalization(logits_of_real, training=True)

                # Feed-forward layer 1
                logits_of_real = tf.contrib.layers.flatten(logits_of_real)
                logits_of_real = tf.layers.dense(logits_of_real, units=128, activation=tf.nn.leaky_relu)
                #logits_of_real = tf.layers.dropout(logits_of_real, dropout_probability, training=True)
                #logits_of_real = tf.layers.batch_normalization(logits_of_real, training=True)
                # 2
                logits_of_real = tf.layers.dense(logits_of_real, units=1)
                proba_of_real = tf.nn.sigmoid(logits_of_real)

            elif self.model=="dcgan_custom":
                # Convolution 1
                image = tf.reshape(image, shape=[-1, self.output_height, self.output_width, self.output_depth])
                logits_of_real = tf.layers.conv2d(image, kernel_size=5, filters=self.depth_layers[0], strides=2, padding='same', activation=tf.nn.leaky_relu)
                logits_of_real = tf.layers.dropout(logits_of_real, dropout_probability, training=True)
                # Convolution 2
                logits_of_real = tf.layers.conv2d(logits_of_real, kernel_size=5, filters=self.depth_layers[1], strides=1, padding='same', activation=tf.nn.leaky_relu)
                logits_of_real = tf.layers.dropout(logits_of_real, dropout_probability, training=True)
                #logits_of_real = tf.layers.batch_normalization(logits_of_real, training=True)
                # Convolution 3
                logits_of_real = tf.layers.conv2d(logits_of_real, kernel_size=5, filters=self.depth_layers[2], strides=1, padding='same', activation=tf.nn.leaky_relu)
                logits_of_real = tf.layers.dropout(logits_of_real, dropout_probability, training=True)
                #logits_of_real = tf.layers.batch_normalization(logits_of_real, training=True)
                # Convolution 4
                logits_of_real = tf.layers.conv2d(logits_of_real, kernel_size=5, filters=self.depth_layers[3], strides=1, padding='same', activation=tf.nn.leaky_relu)
                logits_of_real = tf.layers.dropout(logits_of_real, dropout_probability, training=True)
                #logits_of_real = tf.layers.batch_normalization(logits_of_real, training=True)
                # Classification
                logits_of_real = tf.contrib.layers.flatten(logits_of_real)
                logits_of_real = tf.layers.dense(logits_of_real, units=1)

                proba_of_real = tf.nn.sigmoid(logits_of_real)

            elif self.model=="dcgan_vanilla":
                image = tf.reshape(image, shape=[-1, self.output_height, self.output_width, self.output_depth])
                # Convolution 1
                logits_of_real = tf.layers.conv2d(image, kernel_size=5, filters=self.depth_layers[0], strides=2, padding='same')
                logits_of_real = tf.nn.leaky_relu(logits_of_real)
                # Convolution 2
                logits_of_real = tf.layers.conv2d(logits_of_real, kernel_size=5, filters=self.depth_layers[1], strides=1, padding='same')
                logits_of_real = tf.nn.leaky_relu(logits_of_real)
                logits_of_real = tf.layers.batch_normalization(logits_of_real, training=True)
                # Convolution 3
                logits_of_real = tf.layers.conv2d(logits_of_real, kernel_size=5, filters=self.depth_layers[2], strides=1, padding='same')
                logits_of_real = tf.nn.leaky_relu(logits_of_real)
                logits_of_real = tf.layers.batch_normalization(logits_of_real, training=True)
                # Convolution 4
                logits_of_real = tf.layers.conv2d(logits_of_real, kernel_size=5, filters=self.depth_layers[3], strides=1, padding='same')
                logits_of_real = tf.nn.leaky_relu(logits_of_real)
                logits_of_real = tf.layers.batch_normalization(logits_of_real, training=True)
                # Classification
                logits_of_real = tf.contrib.layers.flatten(logits_of_real)
                logits_of_real = tf.layers.dense(logits_of_real, units=1)

                proba_of_real = tf.nn.sigmoid(logits_of_real)

        return proba_of_real, logits_of_real

    def set_loss(self, real_images_logits, fake_images_logits, flip_labels=False):
        """Set the loss of the discriminator given its prediction.
        :param flip_labels: flip labels in the computation of the discrimination loss (10% flip)"""
        if flip_labels:
            labels_real = tf.where(tf.random_uniform(tf.shape(real_images_logits)) > 0.1,
                                   tf.ones_like(real_images_logits), tf.zeros_like(real_images_logits))
            labels_fake = tf.where(tf.random_uniform(tf.shape(fake_images_logits)) > 0.1,
                                   tf.zeros_like(fake_images_logits), tf.ones_like(fake_images_logits))

        else:
            labels_real = tf.ones_like(real_images_logits)
            labels_fake = tf.zeros_like(fake_images_logits)
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_images_logits,
                                                                           labels=labels_real,
                                                                           name='loss_discri_real_img'),
                                   name='gradient_discri_real_img')
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_images_logits,
                                                                           labels=labels_fake,
                                                                           name='loss_discri_fake_img'),
                                   name='gradient_discri_fake_img')
        self.loss = real_loss + fake_loss

    def set_solver(self):
        """Set the optimization algorithm (AdamOptimizer is used in the DCGAN paper with learning_rate=0.0002,
        beta1=0.5, the rest is default value)"""
        self.variables = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        if self.model == "simple":
            self.solver = tf.train.AdamOptimizer().minimize(self.loss, var_list=self.variables, name='solver_discriminator')

        elif self.model=="intermediate":
            self.solver = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5).minimize(self.loss, var_list=self.variables, name='solver_discriminator')

        elif self.model == "dcgan_custom":
            self.solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.loss,
                                                                                           var_list=self.variables,
                                                                                           name='solver_generator')
        elif self.model == "dcgan_vanilla":
            self.solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.loss,
                                                                                           var_list=self.variables,
                                                                                           name='solver_generator')
