import tensorflow as tf
import math

class Generator:
    def __init__(self, input_shape, first_block_depth=1024):
        self.output_side = input_shape[0]
        self.output_depth = input_shape[2]
        self.blocks_depth = [int(first_block_depth/2**i) for i in range(4)] + [self.output_depth]

    def conv_out_size_same(self, size, stride):
        return int(math.ceil(float(size) / float(stride)))

    def forward_pass(self, z):
        with tf.variable_scope("generator", reuse=None):
            # Projection of noise and proper reshaping
            output_gen = tf.layers.dense(z, units=self.blocks_depth[0]*(self.blocks_depth[0]**2), activation=tf.nn.relu)
            output_gen = tf.reshape(output_gen, shape=[-1, self.blocks_depth[0], self.blocks_depth[0], self.blocks_depth[0]])

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
