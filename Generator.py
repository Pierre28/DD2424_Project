import tensorflow as tf
import numpy as np 

class Generator:
    def __init__(self, input_shape, first_block_depth=1024, model="simple"):
        self.model = model
        # Dimension of data
        self.output_height = input_shape[0]
        self.output_width = input_shape[1]
        self.output_depth = input_shape[2]
        # Parameters of layer
        self.blocks_depth = [int(first_block_depth/2**i) for i in range(2)] + [self.output_depth]
        #self.blocks_size = [int(self.output_height/2**i) for i in range(3)][::-1]
        self.blocks_size = [7, 14, 28]
        # Necessary variables to build graph
        self.variables = []
        self.loss = 0
        self.solver = 0

    def generate_images(self, z, reuse=tf.AUTO_REUSE, is_training=True):  # AUTO_REUSE necessary to compute inception.
        z = tf.convert_to_tensor(z, np.float32)
        with tf.variable_scope("generator", reuse=reuse):
            if self.model=="simple":
                faked_images = tf.layers.dense(z, units=128, activation=tf.nn.relu)
                faked_images = tf.layers.dense(faked_images, units=self.output_depth*self.output_height*self.output_width, activation=tf.nn.sigmoid)
                faked_images = tf.reshape(faked_images, shape=[-1, self.output_height, self.output_width, self.output_depth])

            elif self.model=="intermediate":
                activation = tf.nn.leaky_relu
                d1 = 4
                d2 = 1
                dropout_probability = 0.6
                faked_images = tf.layers.dense(z, units=d1 * d1 * d2, activation=activation)
                faked_images = tf.layers.dropout(faked_images, dropout_probability, training=is_training)
                faked_images = tf.contrib.layers.batch_norm(faked_images, is_training=is_training)
                faked_images = tf.reshape(faked_images, shape=[-1, d1, d1, d2])
                faked_images = tf.image.resize_images(faked_images, size=[7, 7])
                faked_images = tf.layers.conv2d_transpose(faked_images, kernel_size=5, filters=64, strides=2,
                                                          padding='same', activation=activation)
                faked_images = tf.layers.dropout(faked_images, dropout_probability, training=is_training)
                faked_images = tf.contrib.layers.batch_norm(faked_images, is_training=is_training)
                faked_images = tf.layers.conv2d_transpose(faked_images, kernel_size=5, filters=64, strides=2,
                                                          padding='same', activation=activation)
                faked_images = tf.layers.dropout(faked_images, dropout_probability, training=is_training)
                faked_images = tf.contrib.layers.batch_norm(faked_images, is_training=is_training)
                faked_images = tf.layers.conv2d_transpose(faked_images, kernel_size=5, filters=64, strides=1,
                                                          padding='same', activation=activation)
                faked_images = tf.layers.dropout(faked_images, dropout_probability, training=is_training)
                faked_images = tf.contrib.layers.batch_norm(faked_images, is_training=is_training)
                faked_images = tf.layers.conv2d_transpose(faked_images, kernel_size=5, filters=1, strides=1,
                                                          padding='same', activation=tf.nn.sigmoid)

            elif self.model=="dcgan":
                # Projection of noise and proper reshaping
                faked_images = tf.layers.dense(z, units=self.blocks_depth[0]*(self.blocks_size[0]**2), activation=tf.nn.relu)
                faked_images = tf.reshape(faked_images, shape=[-1, self.blocks_size[0], self.blocks_size[0], self.blocks_depth[0]])

                # Fractional-strided convolutions/Deconvolutions
                faked_images = tf.layers.conv2d_transpose(faked_images, kernel_size=5, filters=self.blocks_depth[1], strides=2,
                                                        padding='same', activation=tf.nn.relu)
                faked_images = tf.layers.conv2d_transpose(faked_images, kernel_size=5, filters=self.blocks_depth[2], strides=2,
                                                        padding='same', activation=tf.nn.relu)
                # output_gen = tf.layers.conv2d_transpose(output_gen, kernel_size=5, filters=self.blocks_depth[3], strides=2,
                #                                         padding='same', activation=tf.nn.relu)
                # output_gen = tf.layers.conv2d_transpose(output_gen, kernel_size=5, filters=self.blocks_depth[4], strides=2,
                #                                         padding='same', activation=tf.nn.relu)
                faked_images = tf.nn.tanh(faked_images)
        return faked_images

    def set_loss(self, fake_images_logits):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_images_logits,
                                                                           labels=tf.ones_like(fake_images_logits),
                                                                           name='loss_generator'), name='gradient_generator')

    def set_solver(self):
        self.variables = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        self.solver = tf.train.AdamOptimizer().minimize(self.loss, var_list=self.variables, name='solver_generator')  # Paper: learning_rate=0.0002, beta1=0.5 in Adam
