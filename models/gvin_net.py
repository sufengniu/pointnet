import tensorflow as tf
import numpy as np
import math
import sys
import os

def dot(x, y, sparse=False):
    if sparse:
        return tf.sparse_tensor_dense_matmul(x, y)
    else:
        return tf.matmul(x, y)

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pointclouds_edge = 
    pointclouds_
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

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

# def _list_to_matrix(edge_list, edge_emb):
#     # indices     [batch_size, num_nodes, 2] -> [batch_size, num_nodes*2]
#     # edge_emb    [batch_size, num_nodes, channels] -> []
#     indices = edge_list
#     sparse_edge = []
#     for i in range(batch_size):
#       sparse_edge.append(tf.SparseTensor(indices, edge_emb[:,:,i], dense_shape=tf.constant([], tf.int64)))


#     # output      [batch_size, nodesize, nodesize, channel]
#     return 


def graph_conv(input_features, edge_list, edge_emb):
    '''
    a layer graph convolution: from input_features to output_features
        input:  input_features: 3D tensor (batch_size, num_points, dim)
        output: output_features: 3D tensor (batch_size, num_points, dim)
    '''
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
            sparse_emb = tf.SparseTensor(indices=_edge_list, values=_edge_emb[:,i], dense_shape=tf.constant([100,100], tf.int64))
            output_features.append(dot(sparse_emb, _input_features, sparse=True))
        return tf.stack(output_features)

    # [batch_size, channels, nodesize, feature_size]
    output_features = tf.map_fn(channel_sparse_feature, (edge_list, edge_emb, input_features), dtype=(tf.float32))

    return output_features



def get_model(point_cloud, edge_emb, is_training, bn_decay=None):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    k1_emb = edge_emb
    k1 = _kernel(k1_emb, 4, name_scope="kernel_1")
    l1 = graph_conv(point_cloud, edge_list, k1)

    # k2_emb = _pair_dist(l1)
    k2_emb = edge_emb
    k2 = _kernel(k2_emb, 8, name_scope="kernel_2")
    l2 = graph_conv(l1, k2)

    # k3_emb = _pair_dist(l2)
    k3_emb = edge_emb
    k3 = _kernel(k3_emb, 8, name_scope="kernel_3")
    l3 = graph_conv(l2, k3)



    
    




def get_loss(pred, label, end_points, reg_weight=0.001):

    pass




if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
