import tensorflow as tf
import math


class Generator:
    def __init__(self, input_shape, first_block_depth=1024):
        self.output_side = input_shape[0]
        self.output_depth = input_shape[2]
        self.blocks_depth = [int(first_block_depth/2**i) for i in range(4)] + [self.output_depth]
        self.blocks_size = [int(self.output_side/2**i) for i in range(5)][::-1]

        self.variables = []
        self.loss = tf.Variable([])
        self.optimizer = tf.train.AdamOptimizer()

    def forward_pass(self, z, reuse=None):
        with tf.variable_scope("generator", reuse=reuse):
            # Projection of noise and proper reshaping
            output_gen = tf.layers.dense(z, units=self.blocks_depth[0]*(self.blocks_size[0]**2), activation=tf.nn.relu)
            output_gen = tf.reshape(output_gen, shape=[-1, self.blocks_size[0], self.blocks_size[0], self.blocks_depth[0]])

            # Fractional-strided convolutions/Deconvolutions
            output_gen = tf.layers.conv2d_transpose(output_gen, kernel_size=5, filters=self.blocks_depth[1], strides=2,
                                                    padding='same', activation=tf.nn.relu)
            output_gen = tf.layers.conv2d_transpose(output_gen, kernel_size=5, filters=self.blocks_depth[2], strides=2,
                                                    padding='same', activation=tf.nn.relu)
            output_gen = tf.layers.conv2d_transpose(output_gen, kernel_size=5, filters=self.blocks_depth[3], strides=2,
                                                    padding='same', activation=tf.nn.relu)
            output_gen = tf.layers.conv2d_transpose(output_gen, kernel_size=5, filters=self.blocks_depth[4], strides=2,
                                                    padding='same', activation=tf.nn.relu)
            return tf.nn.tanh(output_gen)

    def update_loss(self, proba_fake_images):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(proba_fake_images,
                                                                      tf.ones_like(proba_fake_images)))

    def initialize_variables(self):
        self.variables = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.loss, var_list=self.variables)