#coding=utf-8
#__author__='YHR'
import tensorflow as tf
from helpFunc import onelayer_mlp,  my_cross_entropy

class dn():
    def __init__(self, params):
        self.batch_samples =params.batch_samples
        self.orginal_dim = params.nowDim
        self.mlp_outdim = 16
        self.emb_hidden_dim = 64
        self.dn_outdim = 16

        self.lr = params.lr
        self.decay_steps = params.decay_steps
        self.decay_rate = params.decay_rate

        self.batchNorm = params.batchNorm
        self.Batchtrain = tf.placeholder(tf.bool, name="Batchtrain")  # feed: True or False

        if params.activation == 'leakyrelu':
            self.activation = tf.nn.leaky_relu
        elif params.activation == 'relu':
            self.activation = tf.nn.relu


        self.dropout_option = params.dropout_option
        self.dropout_rate = params.dropout_rate
        self.dropouttrain = tf.placeholder(tf.bool, name="dropouttrain")

        self.train_x = tf.placeholder(tf.float32, shape=[params.batch_samples, self.orginal_dim], name="trainx")
        self.train_label = tf.placeholder(tf.int32, [params.batch_samples], name="trainlabel")

        self.only_train_cls(params)
        self.test()




    def compute_DN_loss(self, pred, label):
        label_onehot = tf.one_hot(label, 2)
        clsloss = my_cross_entropy(pred, label_onehot)
        return clsloss

    def embedding(self, inp, inp_dim, outdim, activation, batchNorm, Batchtrain):  # -64-32-16
        #  feature extractor
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

    def DN_classify(self, inp, inp_dim, activation, batchNorm, Batchtrain, layerNum=1):
        # classifier  default one layer with 2 units
        with tf.variable_scope('DN_normal_classify', reuse=tf.AUTO_REUSE):
            hidden = inp
            for _ in range(layerNum):
                if _ == layerNum - 1:
                    dim = 2  # binary classification
                    hidden = tf.nn.sigmoid(onelayer_mlp(hidden, _, dim, 0, False ))
                else:
                    dim = inp_dim
                    hidden = activation(onelayer_mlp(hidden, _, dim, batchNorm, Batchtrain ))
                    if self.dropout_option == 'usedropout':
                        hidden = tf.layers.dropout(hidden, rate=self.dropout_rate, training=self.dropouttrain,
                                                   name='dropout')
            return hidden


    def only_train_cls(self, params):
        self.only_train_cls_glbstep = tf.train.get_or_create_global_step()
        if params.decay_flag == 1:
            self.only_train_clslearning_rate = tf.train.exponential_decay(self.lr, self.only_train_cls_glbstep,
                                                                          self.decay_steps, self.decay_rate,
                                                                          staircase=True)
        elif params.decay_flag == 0:
            self.only_train_clslearning_rate = tf.cast(self.lr, tf.float32)

        merge_emb = self.embedding(self.train_x, self.emb_hidden_dim, self.mlp_outdim,
                                                    self.activation, self.batchNorm, self.Batchtrain)
        pred = self.DN_classify(merge_emb, self.dn_outdim, self.activation, self.batchNorm, self.Batchtrain)


        self.only_train_cls_trainloss = self.compute_DN_loss(pred, self.train_label)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            '''Only train DN: frozen Feature extractor, only train the DN part'''
            finetune_var_list = [var for var in tf.trainable_variables() if 'DN_normal_classify' in var.name]

            self.cls_optimizer = tf.train.AdamOptimizer(self.only_train_clslearning_rate, name="cls_optimizer").minimize(self.only_train_cls_trainloss,
                                                                                                        var_list = finetune_var_list,
                                                                                                        global_step= self.only_train_cls_glbstep)

            '''Fine-tune : unfrozen Feature extractor, train feature extractor and DN together'''
            # self.cls_optimizer = tf.train.AdamOptimizer(self.only_train_clslearning_rate,
            #                                             name="cls_optimizer").minimize(self.only_train_cls_trainloss ,
            #                                                                            global_step=self.only_train_cls_glbstep)

    def test(self):
        self.tstx = tf.placeholder(tf.float32, shape=[None, self.orginal_dim], name="tstx")
        self.tsty = tf.placeholder(tf.float32, shape=[None], name="tsty")

        emb = self.embedding(self.tstx, self.emb_hidden_dim, self.mlp_outdim,
                             self.activation, self.batchNorm, self.Batchtrain)



        y_pred = self.DN_classify(emb, self.dn_outdim, self.activation, self.batchNorm, self.Batchtrain)
        self.y_pred_prob = tf.nn.softmax(y_pred)  # softmax,   [sample, nclass]






















