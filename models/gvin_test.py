import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

def dot(x, y, sparse=False):
    if sparse:
        return tf.sparse_tensor_dense_matmul(x, y)
    else:
        return tf.matmul(x, y)

def placeholder_inputs(batch_size, num_edge, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    edgelist_pl = tf.placeholder(tf.int64, shape=(batch_size, num_edge, 2))
    edgeemb_pl = tf.placeholder(tf.float32, shape=(batch_size, num_edge, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, edgelist_pl, edgeemb_pl, labels_pl

def _pair_dist(A):
    r = tf.reduce_sum(A*A, 2)

    # turn r into column vector
    r = tf.expand_dims(r, -1)
    D = r - 2*tf.matmul(A, tf.transpose(A, [0,2,1])) + tf.transpose(r, [0,2,1])
    return D

def _kernel_gen(edge_list, num_channels):
    x = tf.layers.dense(inputs=edge_list, units=64, 
                        use_bias=False, activation=tf.nn.relu, name="layer_1")
    x = tf.layers.dense(inputs=x, units=32, 
                        use_bias=False, activation=tf.nn.relu, name="layer_2")
    out = tf.layers.dense(inputs=x, units=num_channels,
                        use_bias=False, activation=None, name="layer_3")
    return out

def _kernel(point_cloud, num_channels, name_scope="kernel"):
    with tf.variable_scope(name_scope):
        out = tf.map_fn(lambda x: _kernel_gen(x, num_channels), point_cloud, dtype=tf.float32)
    return out

def graph_conv(input_features, channels, edge_list, edge_emb, is_training=True, scope="gvin_conv"):
    '''
    a layer graph convolution: from input_features to output_features
        input:  input_features: 3D tensor (batch_size, num_points, dim)
        output: output_features: 3D tensor (batch_size, num_points, dim)
    '''
    edge_emb = _kernel(edge_emb, channels, name_scope=scope)

    output_features = []
    [batch_size, num_points, dim] = input_features.get_shape().as_list()
    _, _, channels = edge_emb.get_shape().as_list()

    # sparse_emb = _list_to_matrix(edge_list, edge_emb)
    def channel_sparse_feature(edge):
        _edge_list = edge[0]
        _edge_emb = edge[1]
        _input_features = edge[2]
        output_features = []
        for i in range(channels):
            sparse_emb = tf.SparseTensor(indices=_edge_list, values=_edge_emb[:,i], dense_shape=tf.constant([num_points,num_points], tf.int64))
            output_features.append(dot(sparse_emb, _input_features, sparse=True))
        return tf.stack(output_features)

    # [batch_size, channels, nodesize, feature_size]
    output_features = tf.map_fn(channel_sparse_feature, (edge_list, edge_emb, input_features), dtype=(tf.float32))
    output_features = tf.layers.dense(inputs=output_features, units=64, use_bias=False, activation=tf.nn.relu, name="layer_fs")
    output_features = tf.transpose(output_features, [0,2,1,3])
    return output_features

def get_model(point_cloud, edge_list, edge_emb, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = graph_conv(point_cloud_transformed, 4, edge_list, edge_emb, is_training=is_training, scope='gvin_conv1')


    # net = tf_util.conv2d(input_image, 64, [1,3],
    #                      padding='VALID', stride=[1,1],
    #                      bn=True, is_training=is_training,
    #                      scope='conv1', bn_decay=bn_decay)
    # net = tf_util.conv2d(net, 64, [1,1],
    #                      padding='VALID', stride=[1,1],
    #                      bn=True, is_training=is_training,
    #                      scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.reduce_max(net, axis=2), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.summary.scalar('mat loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
