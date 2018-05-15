import tensorflow as tf


class Discriminator:
    def __init__(self, input_shape, first_block_depth=1024):
        self.output_side = input_shape[0]
        self.output_depth = input_shape[2]
        self.blocks_depth = [int(first_block_depth/2**i) for i in range(4)] + [self.output_depth]
 
        self.D_W1 = tf.Variable(self.xavier_init((784, 128)))
        self.D_b1 = tf.Variable(tf.zeros(shape=[128]))

        self.D_W2 = tf.Variable(self.xavier_init([128, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))

        self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

        self.variables = self.theta_D
        self.loss = tf.Variable([])
        self.optimizer = tf.train.AdamOptimizer()

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def forward_pass(self, image, nb_filters=64, reuse=tf.AUTO_REUSE):
        #Reuse = tf.AUTO_REUSE necessary to initialize both fake and real image
        # with tf.variable_scope("discriminator", reuse=reuse):
        #     proba_of_real = tf.reshape(image, shape=[-1, self.output_side, self.output_side, self.output_depth])
        #     proba_of_real = tf.layers.conv2d(proba_of_real, kernel_size=5, filters=nb_filters, strides=2, padding='same', activation=tf.nn.leaky_relu)
        #     proba_of_real = tf.layers.conv2d(proba_of_real, kernel_size=5, filters=nb_filters*2, strides=1, padding='same', activation=tf.nn.leaky_relu)
        #     proba_of_real = tf.layers.conv2d(proba_of_real, kernel_size=5, filters=nb_filters*4, strides=1, padding='same', activation=tf.nn.leaky_relu)
        #     proba_of_real = tf.layers.conv2d(proba_of_real, kernel_size=5, filters=nb_filters*8, strides=1, padding='same', activation=tf.nn.leaky_relu)

        #     proba_of_real = tf.contrib.layers.flatten(proba_of_real)
        #     proba_of_real_logit = tf.layers.dense(proba_of_real, units=1)
            
        #     proba_of_real = tf.nn.sigmoid(proba_of_real_logit)
        # return proba_of_real, proba_of_real_logit
        self.D_h1 = tf.nn.relu(tf.matmul(image, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(self.D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit


    def update_loss(self, real_logits, fake_logits):
        # real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logits), logits=real_logits))
        # fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logits), logits=fake_logits))
        # self.loss = real_loss + fake_loss
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
        self.loss = D_loss_fake + D_loss_real

    def initialize_variables(self):
        #self.variables = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(self.loss, var_list=self.variables)