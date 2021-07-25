#coding=utf-8
#__author__='YHR'
import tensorflow as tf
import os

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path



def bn(inp, batchNorm, Batchtrain):
    with tf.variable_scope(name_or_scope='bn', reuse=tf.AUTO_REUSE):
        if batchNorm == 1:
            x = tf.layers.batch_normalization(inp, training=Batchtrain)
        elif batchNorm == 0:
            x = inp
        return x

def onelayer_mlp(inp, name, outdim, batchNorm, Batchtrain, use_bias=True, L2='noL2'):
    with tf.variable_scope('mlp_layer'+str(name), reuse=tf.AUTO_REUSE):
        if L2 == 'noL2':
            x = tf.layers.dense(inp,
                                units=outdim,
                                activation=None,
                                use_bias=use_bias,
                                kernel_initializer=tf.keras.initializers.he_normal())
            # batch norm
            x = bn(x, batchNorm, Batchtrain)
        elif L2 == 'useL2':
            x = tf.layers.dense(inp,
                                units=outdim,
                                activation=None,
                                use_bias=use_bias,
                                kernel_initializer=tf.keras.initializers.he_normal(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

            # batch norm
            x = bn(x, batchNorm, Batchtrain)
        return x

def my_cross_entropy(pred, label):
    # 只关心正确的类  -- ce 越小越好
    ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label))
    # −∑y ×log(softmax(y))
    return ce


def compute_proto_dist(s_proto, query, softmax='yes'):
    # s_proto (way, dv),   query(way x query, dv)

    _p = tf.expand_dims(s_proto, axis=0)  # (1, way, dv)
    _q = tf.expand_dims(query, axis=1)  # (way x query, 1, dv)

    embedding = tf.pow(tf.subtract(_p, _q),  2)  # (way x query, way, dv)
    dist = tf.cast(tf.reduce_mean(embedding, axis=2), tf.float32)  # (way x query, way)
    if softmax == 'yes':
        prediction = tf.nn.softmax(-dist)  # (way x query, way)
    elif softmax == 'no':
        prediction = -dist  # (way x query, way)
    return prediction

def calculate_Acc(pred, label, way, shot):
    # pred & label: [way*shot, way]  (even just [shot, way] for one class)
    # Tensorflow: bool 和 float 可以互换, tf.cast()

    # (way*shot, )
    predMax = tf.argmax(tf.nn.softmax(pred, axis=-1), -1)
    labelMax = tf.argmax(label, -1)

    judge = tf.cast(tf.equal(predMax, labelMax), tf.float32)

    acc = tf.reshape(judge, shape = [way, shot])  # (way, shot)
    acc = tf.expand_dims(tf.reduce_mean(acc, axis=-1), axis=-1)  # (way, 1)
    return acc


def calculate_RP(pred, label, way, shot,  aimId=1):
    # pred & label: [way*shot, way]
    predMax = tf.argmax(tf.nn.softmax(pred, axis=-1), -1)
    labelMax = tf.argmax(label, -1)

    judge = tf.cast(tf.equal(predMax, labelMax), tf.float32)  # (way*shot, )

    if (aimId == 1):
        TP = tf.reduce_sum(judge[shot: 2 * shot])
        FP = tf.subtract(tf.cast(shot, tf.float32), tf.reduce_sum(judge[0: shot]))
        recall = tf.div(TP, tf.cast(shot, tf.float32))
        precision = tf.div(TP, tf.add(tf.add(TP, FP), 0.00001))

        return tf.expand_dims(precision, -1), tf.expand_dims(recall,-1)

    else:
        print('calculate_RP  error....')







def focal_loss_softmaxLogit(logits, labels, alpha, batch_samples, epsilon = 1.e-7, gamma=2.0):
    # 注意，alpha是一个和你的分类类别数量相等的向量；
    # alpha=[[1], [1], [1], [1]]  # (Class ,1)

    #, multi_dim = False

    '''
    :param logits:  [batch_size, n_class]
    :param labels: [batch_size]  !!! not one-hot !!!
    :return: -alpha*(1-y)^gamma * log(y)
    它是在哪实现 1- y 的？ 通过gather选择的就是1-p,而不是通过计算实现的；
    logits soft max之后是多个类别的概率，也就是二分类时候的1-P和P；多分类的时候不是1-p了；

    怎么把alpha的权重加上去？
    通过gather把alpha选择后变成batch长度，同时达到了选择和维度变换的目的

    是否需要对logits转换后的概率值进行限制？
    需要的，避免极端情况的影响

    针对输入是 (N，P，C )和  (N，P)怎么处理？
    先把他转换为和常规的一样形状，（N*P，C） 和 （N*P,）

    bug:
    ValueError: Cannot convert an unknown Dimension to a Tensor: ?
    因为输入的尺寸有时是未知的，导致了该bug,如果batchsize是确定的，可以直接修改为batchsize

    '''


    # if multi_dim:
    #     logits = tf.reshape(logits, [-1, logits.shape[2]])
    #     labels = tf.reshape(labels, [-1])

    # (Class ,1)
    alpha = tf.constant(alpha, dtype=tf.float32)

    labels = tf.cast(labels, dtype=tf.int32)
    logits = tf.cast(logits, tf.float32)
    # (N,Class) > N*Class
    softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
    # (N,) > (N,) ,但是数值变换了，变成了每个label在N*Class中的位置
    labels_shift = tf.range(0, batch_samples) * logits.shape[1] + labels
    # (N*Class,) > (N,)
    prob = tf.gather(softmax, labels_shift)
    # 预防预测概率值为0的情况  ; (N,)
    prob = tf.clip_by_value(prob, epsilon, 1. - epsilon)
    # (Class ,1) > (N,)
    alpha_choice = tf.gather(alpha, labels)
    # (N,) > (N,)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    weight = tf.multiply(alpha_choice, weight)
    # (N,) > 1
    loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))
    return loss
