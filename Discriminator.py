import tensorflow as tf


class Discriminator:
    def __init__(self, input_shape, first_block_depth=1024, simple_model=True):
        self.simple_model = simple_model
        # Dimension of data
        self.output_height = input_shape[0]
        self.output_width = input_shape[1]
        self.output_depth = input_shape[2]
        # Parameters of layer
        self.blocks_depth = [int(first_block_depth/2**i) for i in range(4)] + [self.output_depth]
        # Necessary variables to build graph
        self.variables = []
        self.loss = 0
        self.solver = 0

    def compute_probability(self, image, nb_filters=64, reuse=tf.AUTO_REUSE):
        # Reuse = tf.AUTO_REUSE necessary to initialize both fake and real image
        with tf.variable_scope("discriminator", reuse=reuse):
            if self.simple_model:
                image = tf.contrib.layers.flatten(image)
                logits_of_real = tf.layers.dense(image, units=128, activation=tf.nn.relu)
                logits_of_real = tf.layers.dense(logits_of_real, units=1)
                proba_of_real = tf.nn.sigmoid(logits_of_real)

            else:
                image = tf.reshape(image, shape=[-1, self.output_height, self.output_width, self.output_depth])
                logits_of_real = tf.layers.conv2d(image, kernel_size=5, filters=nb_filters, strides=2, padding='same', activation=tf.nn.leaky_relu)
                #proba_of_real = tf.layers.conv2d(proba_of_real, kernel_size=5, filters=nb_filters*2, strides=1, padding='same', activation=tf.nn.leaky_relu)
                #proba_of_real = tf.layers.conv2d(proba_of_real, kernel_size=5, filters=nb_filters*4, strides=1, padding='same', activation=tf.nn.leaky_relu)
                #proba_of_real = tf.layers.conv2d(proba_of_real, kernel_size=5, filters=nb_filters*8, strides=1, padding='same', activation=tf.nn.leaky_relu)
                logits_of_real = tf.contrib.layers.flatten(logits_of_real)
                logits_of_real = tf.layers.dense(logits_of_real, units=1)

                proba_of_real = tf.nn.sigmoid(logits_of_real)

        return proba_of_real, logits_of_real

    def set_loss(self, real_images_logits, fake_images_logits):
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_images_logits,
                                                                           labels=tf.ones_like(real_images_logits),
                                                                           name='loss_discri_real_img'),
                                   name='gradient_discri_real_img')
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_images_logits,
                                                                           labels=tf.zeros_like(fake_images_logits),
                                                                           name='loss_discri_fake_img'),
                                   name='gradient_discri_fake_img')
        self.loss = real_loss + fake_loss
        
        
        

    def set_solver(self):
        self.variables = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        self.solver = tf.train.AdamOptimizer().minimize(self.loss, var_list=self.variables, name='solver_discriminator')  # Paper: learning_rate=0.0002, beta1=0.5 in Adam
