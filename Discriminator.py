import tensorflow as tf


class Discriminator:
    def __init__(self, input_shape, first_block_depth=1024):
        self.output_side = input_shape[0]
        self.output_depth = input_shape[2]
        self.blocks_depth = [int(first_block_depth/2**i) for i in range(4)] + [self.output_depth]

        self.variables = []
        self.loss = tf.Variable([])
        self.optimizer = tf.train.AdamOptimizer()

    def forward_pass(self, image, nb_filters=64, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            logits = tf.reshape(image, shape=[-1, self.output_side, self.output_side, self.output_depth])
            logits = tf.layers.conv2d(logits, kernel_size=5, filters=nb_filters, strides=2, padding='same', activation=tf.nn.leaky_relu)
            logits = tf.layers.conv2d(logits, kernel_size=5, filters=nb_filters*2, strides=1, padding='same', activation=tf.nn.leaky_relu)
            logits = tf.layers.conv2d(logits, kernel_size=5, filters=nb_filters*4, strides=1, padding='same', activation=tf.nn.leaky_relu)
            logits = tf.layers.conv2d(logits, kernel_size=5, filters=nb_filters*8, strides=1, padding='same', activation=tf.nn.leaky_relu)

            logits = tf.contrib.layers.flatten(logits)
            logits = tf.layers.dense(logits, units=1)
            probabilities = tf.nn.sigmoid(logits)
            return probabilities, logits

    def update_loss(self, real_images, fake_images):
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_images, tf.ones_like(real_images)))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_images, tf.zeros_like(fake_images)))
        self.loss = real_loss + fake_loss

    def initialize_variables(self):
        self.variables = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.loss, var_list=self.variables)