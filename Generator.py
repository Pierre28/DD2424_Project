import tensorflow as tf
import numpy as np 

class Generator:
    def __init__(self, input_shape, first_block_depth=1024):
        self.output_side = input_shape[0]
        self.output_depth = input_shape[2]
        self.blocks_depth = [int(first_block_depth/2**i) for i in range(2)] + [self.output_depth]
        #self.blocks_size = [int(self.output_side/2**i) for i in range(3)][::-1]
        self.blocks_size = [7, 14, 28]
        self.variables = []
        self.loss = tf.Variable([])
        
        self.optimizer = tf.train.AdamOptimizer()

    def forward_pass(self, z, reuse=None):
        z = tf.convert_to_tensor(z, np.float32)
        with tf.variable_scope("generator", reuse=reuse):
            # Projection of noise and proper reshaping

            output_gen = tf.layers.dense(z, units=self.blocks_depth[0]*(self.blocks_size[0]**2), activation=tf.nn.relu)
            output_gen = tf.reshape(output_gen, shape=[-1, self.blocks_size[0], self.blocks_size[0], self.blocks_depth[0]])

            # Fractional-strided convolutions/Deconvolutions
            output_gen = tf.layers.conv2d_transpose(output_gen, kernel_size=5, filters=self.blocks_depth[1], strides=2,
                                                    padding='same', activation=tf.nn.relu)
            output_gen = tf.layers.conv2d_transpose(output_gen, kernel_size=5, filters=self.blocks_depth[2], strides=2,
                                                    padding='same', activation=tf.nn.relu)
            # output_gen = tf.layers.conv2d_transpose(output_gen, kernel_size=5, filters=self.blocks_depth[3], strides=2,
            #                                         padding='same', activation=tf.nn.relu)
            # output_gen = tf.layers.conv2d_transpose(output_gen, kernel_size=5, filters=self.blocks_depth[4], strides=2,
            #                                         padding='same', activation=tf.nn.relu)
            output_gen = tf.nn.tanh(output_gen)
        return output_gen

    def update_loss(self, proba_fake_images):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=proba_fake_images,
                                                                      labels=tf.ones_like(proba_fake_images)))

    def initialize_variables(self):
        self.variables = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        print(tf.trainable_variables())
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.loss, var_list=self.variables)
