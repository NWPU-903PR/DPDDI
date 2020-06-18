#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 10:45:18 2019
comparing three features of structions,DBP and ATC
@author: fyh
"""
from __future__ import division
from __future__ import print_function
import csv
import time
import os
import h5py
import copy
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import scipy.io as sio
import tensorflow as tf
import scipy.sparse as sp
import random
import time

from keras import models
from keras import layers
from keras  import utils
from keras import optimizers
from keras import losses
from keras import metrics
from keras.callbacks import Callback
from keras.models import Model  # 泛型模型
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

from numpy import linalg as LA

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

from gae.optimizer import OptimizerAE

from gae.model import GCNModelAE


def get_roc_score1(edges_pos, edges_neg,emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pred_probability_pos = []
    pos = []
    for e in edges_pos:
        pred_probability_pos.append(adj_rec[e[0], e[1]])
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    pred_probability_neg = []
    neg = []
    for e in edges_neg:
        pred_probability_neg.append(adj_rec[e[0], e[1]])
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    preds_probability_all = np.hstack([pred_probability_pos, pred_probability_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_probability_all)   ## preds_all
    precision, recall, pr_thresholds = precision_recall_curve(labels_all, preds_probability_all)
    aupr_score = auc(recall, precision)   

    return roc_score, aupr_score

def get_roc_score2(pre,preds_probability_all,y_test):
   
    preds_all = np.array(pre)
    preds_probability_all = np.array(pre)
    labels_all = np.array(y_test)
    roc_score = roc_auc_score(labels_all,preds_probability_all)   
    
    precision, recall, pr_thresholds = precision_recall_curve(labels_all, preds_probability_all )  ##preds_all   preds_probability_all
    aupr_score = auc(recall, precision)
#    
    all_F_measure=np.zeros(len(pr_thresholds))
    for k in range(0,len(pr_thresholds)):
        if (precision[k]+precision[k])>0:
            all_F_measure[k]=2*precision[k]*recall[k]/(precision[k]+recall[k])
        else:
            all_F_measure[k]=0
    max_index=all_F_measure.argmax()
    threshold=pr_thresholds[max_index]
#        
    fpr, tpr, auc_thresholds = roc_curve(labels_all, preds_probability_all)
    auc_score = auc(fpr, tpr)
    predicted_score=np.zeros(shape=(len(labels_all),1))
    predicted_score[preds_probability_all>threshold]=1
    confusion_matri = confusion_matrix(y_true=labels_all, y_pred=predicted_score)
    print("confusion_matrix:",confusion_matri)
        
    f=f1_score(labels_all,predicted_score)
    accuracy=accuracy_score(labels_all,predicted_score)
    precision=precision_score(labels_all,predicted_score)
    recall=recall_score(labels_all,predicted_score)

    return roc_score, aupr_score,precision, recall,accuracy,f

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


    
def mask_test_edges(adj):
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = [],[],[],[],[],[]
    adj_train = []
#    kfold = 5
#    num_train_kd = edges.shape[0]// kfold
#    num_test_kd = int(np.floor(num_train_kd * 0.2))
#    num_val_kd = int(np.floor(num_train_kd * 0.5))   #num_train_kd / 20.

    link_number = 0
    non_link_number = 0
    link_position = []
    non_link_position = []  # all non-link position
    for i in range(0, adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] == 1:
                link_number = link_number + 1
                link_position.append([i, j])
            elif adj[i,j] ==0:
                non_link_number = non_link_number +1
                non_link_position.append([i,j])


    link_position = np.array(link_position)
    non_link_position = np.array(non_link_position)
    seed = 12
    random.seed(seed)
    link_index = np.arange(0, link_number)
    non_link_index = np.arange(0,non_link_number)
    random.shuffle(link_index)
    random.shuffle(non_link_index)
    kfold = 5
    num_train_kd = link_number// kfold
    num_test_kd = int(np.floor(num_train_kd * 0.2))
    num_val_kd = int(np.floor(num_train_kd * 0.5))   #num_train_kd / 20.
    num_negtive_kd = non_link_number // kfold
    
    for i in range(kdfold):
        train_link_index = link_index[i *num_train_kd:(i+1)*num_train_kd]        
        test_link_index = train_link_index[0:num_test_kd]
        val_link_index = train_link_index[num_test_kd:num_test_kd + num_val_kd]
        train_index = train_link_index[num_test_kd + num_val_kd:num_train_kd]
        
        train_index.sort()
        val_link_index.sort()
        test_link_index.sort()       
        train_edges.append(link_position[train_index])
        val_edges.append(link_position[val_link_index])
        test_edges.append(link_position[test_link_index])
        
        fold =6
        kd_no_link_index = non_link_index[i*num_negtive_kd:(i+1)*num_negtive_kd]
        test_no_link_index = kd_no_link_index[0: fold * num_test_kd]
        val_no_link_index = kd_no_link_index[fold * num_test_kd:fold * (num_test_kd + num_val_kd)]
        train_no_link_index = kd_no_link_index[fold * (num_test_kd + num_val_kd):num_negtive_kd]
        
        train_no_index.sort()
        val_no_link_index.sort()
        test_no_link_index.sort()       
        train_edges_false.append(non_link_position[train_no_index])
        val_edges_false.append(non_link_position[val_no_link_index])
        test_edges_false.append(non_link_position[test_no_link_index])
        data = np.ones(train_edges.shape[0])
        # Re-build adj matrix
        adj_train_rebuild = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        adj_train_rebuild = adj_train_rebuild + adj_train_rebuild.T
        adj_train.append(adj_train_rebuild)
    
    return adj_train, train_edges, train_edges_false,val_edges, val_edges_false, test_edges, test_edges_false

def node_trans_edges(test_edges,test_edges_false,val_edges,val_edges_false,train_edges,train_edges_false):
    if type(test_edges) != list:
        test_edges = test_edges.tolist()
    if type(train_edges_false) != list:
        train_edges_false = train_edges_false.tolist()
    if type(train_edges) != list:
        train_edges = train_edges.tolist()
    if type(test_edges_false) != list:
        test_edges_false = test_edges_false.tolist()  
    if type(val_edges) != list:
        val_edges = val_edges.tolist() 
    if type(val_edges_false) != list:
        val_edges_false = val_edges_false.tolist() 
        
    x_train_index = train_edges + train_edges_false
    y_train = [1]*len(train_edges) + [0] * len(train_edges_false)
    x_val_index = val_edges + val_edges_false
    y_val = [1]*len(val_edges) + [0] * len(val_edges_false)
    x_test_index = test_edges + test_edges_false
    y_test = [1]*len(test_edges) + [0] * len(test_edges_false)
    x_train = []
    x_val = []
    x_test = []
    ###transform node embdding to edges feature by concat opration 
    t = time.time()
    for i in range(len(x_train_index)):
        x_train.append(np.hstack((embedding[ x_train_index[i][0]],embedding[ x_train_index[i][1]])))
    for i in range(len(x_val_index)):
        x_val.append(np.hstack((embedding[ x_val_index[i][0]],embedding[ x_val_index[i][1]])))
    for i in range(len(x_test_index)):
        x_test.append(np.hstack((embedding[ x_test_index[i][0]] ,embedding[ x_test_index[i][1]])))
    print("cost time of embedding concat", time.time()-t)
    
    y_train = utils.to_categorical(y_train, 2)
    y_test = utils.to_categorical(y_test, 2)
    y_val =  utils.to_categorical(y_val, 2)
    
    x_train = np.matrix(x_train)
    x_test = np.matrix(x_test)
    x_val = np.matrix(x_val)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)
    print("fill embedding data accomplishment")
    return x_train, x_test, x_val, y_train, y_test, y_val
    

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.x_val,self.y_val = validation_data
    def on_epoch_end(self, epoch, log={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print('\n ROC_AUC - epoch:%d - score:%.6f \n' % (epoch+1, score))



# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 700, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 256, 'Number of units in hidden layer 2. ')
#flags.DEFINE_integer('hidden3', 128, 'Number of units in hidden layer 2. ')
#flags.DEFINE_integer('hidden4', 128, 'Number of units in hidden layer 2. ')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features',0, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

###======================读入要处理的数据为dataframe格式==================
#
filename = 'D:/CODE-myself/matlab/Data/Data.mat'
data = h5py.File(filename,'r')
adj = data['Adj_binary'][:]
adj = adj.transpose()
adj_copy = copy.deepcopy(adj)
adj = sp.csr.csr_matrix(adj)
# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis,:], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

if FLAGS.features == 0:
    features = sp.identity(adj.shape[0])  # featureless

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
roc_score_arr,aupr_score_arr,precision_arr, recall_arr,accuracy_arr,f_arr = [],[],[],[],[],[]
print("split end")
CV = 5
for i in range(CV):
    adj = adj_train[i]
    adj_norm = preprocess_graph(adj)   # Some preprocessing
    model = GCNModelAE(placeholders, num_features, features_nonzero)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    with tf.name_scope('optimizer'):      # Optimizer
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
        
    sess = tf.Session()    # Initialize sessioN
    sess.run(tf.global_variables_initializer())
    cost_val = []
    acc_val = []
    val_roc_score = []
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    for epoch in range(FLAGS.epochs):# Train model   #train_loss, train_acc= [],[]
        t = time.time()
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)    # Construct feed dictionary
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]
        #    train_loss.append(avg_cost)
        #   train_acc.append(avg_accuracy)
        roc_curr, aupr_score = get_roc_score1(val_edges, val_edges_false)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_ap=", "{:.5f}".format(aupr_score),
          "time=", "{:.5f}".format(time.time() - t))
    print("Optimization Finished!")
    x_train, x_test, x_val, y_train, y_test, y_val = node_trans_edges(test_edges[i],test_edges_false[i],val_edges[i],val_edges_false[i],train_edges[i],train_edges_false[i])
    RocAuc = RocAucEvaluation(validation_data=(x_val,y_val), interval=1)
    ####=====================deep learning model to predict =============================
    ####embedding 串联的模型 得到256维的数据
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(256,)))
    model.add(layers.Dense(64, activation='relu', input_shape=(128,)))
    model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
    model.add(layers.Dense(2, activation='softmax'))
    model.compile(optimizer=optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0),   ##  optimizer='adam'
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
    print("model complie incomplishment,model fit begin")
    # 训练网络
    model.fit(x_train,y_train, batch_size=50, epochs=200,validation_data=(x_val, y_val), callbacks=[RocAuc], verbose=2)
    pre = model.predict(x_test)   ###输出的每一类的概率，如果是二分类，就是size（test）  *2
    pre_lab = np.argmax(model.predict(x_test), axis=1)
    y_test = [y_test[i][1] for i in range(len(y_test))]
    pre_probability = [pre[i][1] for i in range(len(pre))]
    
    roc_score,aupr_score,precision, recall,accuracy,f = get_roc_score2(pre_lab,pre_probability,y_test)
    roc_score_arr.append(roc_score)
    aupr_score_arr.append(aupr_score)
    precision_arr.append(precision)
    recall_arr.append(recall)
    accuacy_arr.append(accuracy)
    f_arr.append(f)
    print(roc_score,aupr_score,precision, recall,accuracy,f)

roc_score = np.mean(roc_score_arr)
aupr_score = np.mean(aupr_score_arr)
precision = np.mean(precision_arr)
recall = np.mean(recall_arr)
accuracy = np.mean(accuacy_arr)
f = np.mean(f_arr)

print( "roc_score=", "{:.5f}".format(roc_score), "aupr_score =", "{:.5f}".format(aupr_score ),
          "precision=", "{:.5f}".format(precision),  "recall=", "{:.5f}".format(recall),
            "accuracy =", "{:.5f}".format(accuracy ),  "f =", "{:.5f}".format(f))


