from Generator import *
from Discriminator import *
import tensorflow as tf

class DCGAN():
    def __init__(self, input_shape, first_block_depth=1024, dim_noise=100):
        self.output_side = input_shape[0]
        self.output_depth = input_shape[2]
        self.dim_noise = dim_noise

        self.X_in = tf.placeholder(dtype=tf.float32, shape=[None, self.output_side, self.output_side, 1], name='X')
        self.noise = tf.placeholder(dtype=tf.float32, shape=[None, dim_noise])
        self.proba_discriminator = tf.placeholder(dtype=tf.float32, name='proba_discriminator')
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        self.discriminator = Discriminator(input_shape, first_block_depth)
        self.generator = Generator(input_shape, first_block_depth)

    def get_noise(self, min_distri=-1, max_distri=1):
        return tf.random_uniform([1, self.dim_noise], min_distri, max_distri)

    def update_loss(self, real_images, fake_images, probability_fake_images):
        self.discriminator.update_loss(real_images, fake_images)
        self.generator.update_loss(probability_fake_images)

    def initialize_variables(self):
        z = self.get_noise()
        ini_images = self.generator.forward_pass(z)
        self.generator.initialize_variables()
        self.discriminator.forward_pass(ini_images)
        self.discriminator.initialize_variables()

    def train(self):
        pass