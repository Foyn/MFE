#coding=utf-8
#__author__='YHR'
import tensorflow as tf
from helpFunc import onelayer_mlp,  my_cross_entropy

class Mulfeat_ce_sia():
    def __init__(self, params):
        self.orginal_dim = params.nowDim
        self.mlp_outdim = 16
        self.emb_hidden_dim = 64
        self.batch_samples = params.batch_samples

        self.lr = params.lr
        self.decay_steps = params.decay_steps
        self.decay_rate = params.decay_rate

        self.sia_gamma = 1.0

        self.batchNorm = params.batchNorm
        self.Batchtrain = tf.placeholder(tf.bool)  # feed: True or False

        if params.activation == 'leakyrelu':
            self.activation = tf.nn.leaky_relu
        elif params.activation == 'relu':
            self.activation = tf.nn.relu

        self.dropout_option = params.dropout_option
        self.dropout_rate = params.dropout_rate
        self.dropouttrain = tf.placeholder(tf.bool)

        self.train_x0 = tf.placeholder(tf.float32, [None, self.orginal_dim], name="train_x0")
        self.train_y0 = tf.placeholder(tf.int32, shape=[None], name="train_y0")

        self.train_x1 = tf.placeholder(tf.float32, [None, self.orginal_dim], name="train_x1")
        self.train_y1 = tf.placeholder(tf.int32, shape=[None], name="train_y1")

        self.x0equalx1 = tf.placeholder(tf.float32, shape=[None], name="x0equalx1")

        self.train(params)
        self.test()


    def embedding(self, inp, inp_dim, outdim, activation, batchNorm, Batchtrain):  # -64-32-16
        # 1.feature extractor
        with tf.variable_scope('Mlp_embedding', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                hidden = inp
                dim = inp_dim
                # 0
                hidden = activation(onelayer_mlp(hidden, 0, dim, batchNorm, Batchtrain))  # 64
                if self.dropout_option == 'usedropout':
                    hidden = tf.layers.dropout(hidden, rate=self.dropout_rate, training=self.dropouttrain,
                                               name='dropout')

                # 1
                hidden = activation(onelayer_mlp(hidden, 1, dim//2, batchNorm, Batchtrain))  # 32
                if self.dropout_option == 'usedropout':
                    hidden = tf.layers.dropout(hidden, rate=self.dropout_rate, training=self.dropouttrain,
                                               name='dropout')

                # 2
                hidden = onelayer_mlp(hidden, 2, outdim, batchNorm, Batchtrain)  # 16

            return hidden

    def classficationLayer(self, inp):
        # classifier   default one layer with 2 units
        with tf.variable_scope('Cls_2uints', reuse=tf.AUTO_REUSE):
            out = tf.nn.sigmoid(onelayer_mlp(inp, 0, 2, 0, False))
            return out




    def compute_ce_loss(self, pred, label):
        ce_loss = my_cross_entropy(pred, tf.one_hot(label, 2))
        return ce_loss


    def compute_Sialoss(self, embx0, embx1):
        with tf.variable_scope('SiameseLoss', reuse=tf.AUTO_REUSE):
            margin = 5.0
            labels_t = self.x0equalx1
            labels_f = tf.subtract(1.0, self.x0equalx1)

            eucd2 = tf.pow(tf.subtract(embx0, embx1), 2)
            eucd2 = tf.reduce_sum(eucd2, 1)

            eucd = tf.sqrt(eucd2 + 1e-8)
            C = tf.constant(margin)

            pos = tf.multiply(labels_t, eucd2)
            neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2))

            losses = tf.add(pos, neg)
            loss = tf.reduce_mean(losses)
            return loss


    def train(self, params):
        self.global_step = tf.train.get_or_create_global_step()
        if params.decay_flag == 1:
            self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step,
                                                        self.decay_steps, self.decay_rate,
                                                        staircase=True)
        elif params.decay_flag == 0:
            self.learning_rate = tf.cast(self.lr, tf.float32)


        embx0 = self.embedding(self.train_x0, self.emb_hidden_dim, self.mlp_outdim,
                                            self.activation, self.batchNorm, self.Batchtrain)
        embx1 = self.embedding(self.train_x1, self.emb_hidden_dim, self.mlp_outdim,
                                            self.activation, self.batchNorm, self.Batchtrain)
        embx = tf.concat([embx1, embx0], axis=0)
        train_y = tf.concat([self.train_y1, self.train_y0], axis=0)
        y_pred = self.classficationLayer(embx)

        self.ce_loss = self.compute_ce_loss(y_pred, train_y)
        self.sia_loss = self.compute_Sialoss(embx0, embx1)
        self.trainLoss = self.ce_loss + self.sia_gamma * self.sia_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.trainLoss,
                                                                                 global_step=self.global_step)

    def test(self):
        self.tstx = tf.placeholder(tf.float32, shape=[None, self.orginal_dim], name="tstx")
        self.tsty = tf.placeholder(tf.float32, shape=[None], name="tsty")

        emb = self.embedding(self.tstx, self.emb_hidden_dim, self.mlp_outdim,
                             self.activation, self.batchNorm, self.Batchtrain)

        y_pred = self.classficationLayer(emb)
        self.y_pred_prob = tf.nn.softmax(y_pred)  # softmax, [sample, nclass]


















