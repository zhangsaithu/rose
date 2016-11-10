########################
# A demo to train ROSE #
########################
import sys, os
import re, fileinput, math
import numpy as np
import random
import caffe
import h5py
from gensim.models import word2vec
import ribo_convnet
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold

# make sure that caffe is on the python path
caffe_root = '/opt/caffe-master/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
train_one_model = True #!!!!!!!!

val_file = '/home/szhang/Riboseq/r64/Pop14/model/net_architecture_cv/train_val_70.prototxt'
test_file = '/home/szhang/Riboseq/r64/Pop14/model/net_architecture_cv/train_test_70.prototxt'
solver_file = '/home/szhang/Riboseq/r64/Pop14/model/net_architecture_cv/solver_35.prototxt'
# load input!!!
model = {}
nct = ['A', 'T', 'C', 'G']
cnt = 0
for a in nct:
    for b in nct:
        for c in nct:
            ivec = np.zeros(64)
            ivec[cnt] = 1
            model[a+b+c] = ivec
            cnt += 1
w2v_len = 64

label = []
score = []
if train_one_model:
    # load cv data
    random.seed(1024)
    X_train = np.load(os.path.join('/home/szhang/Riboseq/r64/Pop14/data/train_data/cv0','X_train.npy'))
    X_val = np.load(os.path.join('/home/szhang/Riboseq/r64/Pop14/data/train_data/cv0','X_val.npy'))
    X_test = np.load(os.path.join('/home/szhang/Riboseq/r64/Pop14/data/train_data/cv0','X_test.npy'))
    y_train = np.load(os.path.join('/home/szhang/Riboseq/r64/Pop14/data/train_data/cv0','y_train.npy'))
    y_val = np.load(os.path.join('/home/szhang/Riboseq/r64/Pop14/data/train_data/cv0','y_val.npy'))
    y_test = np.load(os.path.join('/home/szhang/Riboseq/r64/Pop14/data/train_data/cv0','y_test.npy'))

    X_train = np.concatenate((X_train,X_val))
    X_train = np.concatenate((X_train,X_test))
    y_train = np.concatenate((y_train,y_val))
    y_train = np.concatenate((y_train,y_test))

    kf = KFold(n=len(X_train), n_folds=10, shuffle=True, random_state=1024)
    flag = False
    for train_index, test_index in kf:
        if flag:
            break
        flag = True
        
        X_val = X_train[test_index]
        X_train = X_train[train_index]
        y_val = y_train[test_index]
        y_train = y_train[train_index]

    X_test = np.load(os.path.join('/home/szhang/Riboseq/r64/Pop14/data/test_data','X_test.npy'))
    y_test = np.load(os.path.join('/home/szhang/Riboseq/r64/Pop14/data/test_data','y_test.npy'))

    # initilize X array
    X_train_w2v = np.zeros([len(X_train), w2v_len, 1, len(X_train[0])/3]).astype(np.float32)
    X_val_w2v = np.zeros([len(X_val), w2v_len, 1, len(X_val[0])/3]).astype(np.float32)
    X_test_w2v = np.zeros([len(X_test), w2v_len, 1, len(X_test[0])/3]).astype(np.float32)
    # construct word2vec representations
    scnt = 0
    for s in X_train:
        seq_vec = np.zeros([len(X_train[0])/3, w2v_len])
        codon = ''
        ccnt = 1
        for c in s:
            if ccnt%3 != 0:
                codon = codon + c
                ccnt += 1
            else:
                codon = codon + c
                seq_vec[ccnt/3 - 1, :] = model[codon]
                ccnt += 1
                codon = ''
        seq_vec = seq_vec.transpose()
        X_train_w2v[scnt,:, 0, :] = seq_vec
        scnt += 1
    y_train = np.array(y_train).astype(np.int)
    scnt = 0
    for s in X_val:
        seq_vec = np.zeros([len(X_val[0])/3, w2v_len])
        codon = ''
        ccnt = 1
        for c in s:
            if ccnt%3 != 0:
                codon = codon + c
                ccnt += 1
            else:
                codon = codon + c
                seq_vec[ccnt/3 - 1, :] = model[codon]
                ccnt += 1
                codon = ''
        seq_vec = seq_vec.transpose()
        X_val_w2v[scnt,:, 0, :] = seq_vec
        scnt += 1
    y_val = np.array(y_val).astype(np.int)
    scnt = 0
    for s in X_test:
        seq_vec = np.zeros([len(X_test[0])/3, w2v_len])
        codon = ''
        ccnt = 1
        for c in s:
            if ccnt%3 != 0:
                codon = codon + c
                ccnt += 1
            else:
                codon = codon + c
                seq_vec[ccnt/3 - 1, :] = model[codon]
                ccnt += 1
                codon = ''
        seq_vec = seq_vec.transpose()
        X_test_w2v[scnt,:, 0, :] = seq_vec
        scnt += 1
    y_test = np.array(y_test).astype(np.int)
      
    # random shuffling
    # train data
    indices = np.arange(len(y_train))
    np.random.shuffle(indices)
    X_train_w2v = X_train_w2v[indices]
    y_train = y_train[indices]
    # validation data
    indices = np.arange(len(y_val))
    np.random.shuffle(indices)
    X_val_w2v = X_val_w2v[indices]
    y_val = y_val[indices]
    # test data
    indices = np.arange(len(y_test))
    np.random.shuffle(indices)
    X_test_w2v = X_test_w2v[indices]
    y_test = y_test[indices]

    # construct h5py file for caffe input
    with h5py.File('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/train.h5', 'w') as f:
        f['data'] = X_train_w2v
        f['label'] = y_train
    with h5py.File('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/val.h5', 'w') as f:
        f['data'] = X_val_w2v
        f['label'] = y_val
    with h5py.File('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/test.h5', 'w') as f:
        f['data'] = X_test_w2v
        f['label'] = y_test
    with open('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/train.txt', 'w') as f:
        f.write('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/train.h5\n')
    with open('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/val.txt', 'w') as f:
        f.write('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/val.h5\n')
    with open('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/test.txt', 'w') as f:
        f.write('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/test.h5\n')

    # run cnn and return result in test data (with best validation result)
    tlabel, tscore = ribo_convnet.ribo_cnn_net1(len(y_train), len(y_val), len(y_test), val_file, test_file, solver_file)
    label.extend(tlabel)
    score.extend(tscore)
    auc = roc_auc_score(label, score)
    print('AUC is: '+str(auc))
    np.save('/home/szhang/Riboseq/r64/Pop14/data/Pop14_label_test_1', label)
    np.save('/home/szhang/Riboseq/r64/Pop14/data/Pop14_prob_test_1', score)
else:
    model_num = 64
    for mnum in range(model_num):
        # load cv data
        X_train = np.load(os.path.join('/home/szhang/Riboseq/r64/Pop14/data/train_data/cv0','X_train.npy'))
        X_val = np.load(os.path.join('/home/szhang/Riboseq/r64/Pop14/data/train_data/cv0','X_val.npy'))
        X_test = np.load(os.path.join('/home/szhang/Riboseq/r64/Pop14/data/train_data/cv0','X_test.npy'))
        y_train = np.load(os.path.join('/home/szhang/Riboseq/r64/Pop14/data/train_data/cv0','y_train.npy'))
        y_val = np.load(os.path.join('/home/szhang/Riboseq/r64/Pop14/data/train_data/cv0','y_val.npy'))
        y_test = np.load(os.path.join('/home/szhang/Riboseq/r64/Pop14/data/train_data/cv0','y_test.npy'))

        X_train = np.concatenate((X_train,X_val))
        X_train = np.concatenate((X_train,X_test))
        y_train = np.concatenate((y_train,y_val))
        y_train = np.concatenate((y_train,y_test))

        kf = KFold(n=len(X_train), n_folds=10, shuffle=True, random_state=None)
        flag = False
        for train_index, test_index in kf:
            if flag:
                break
            flag = True
            
            X_val = X_train[test_index]
            X_train = X_train[train_index]
            y_val = y_train[test_index]
            y_train = y_train[train_index]

        X_test = np.load(os.path.join('/home/szhang/Riboseq/r64/Pop14/data/test_data','X_test.npy'))
        y_test = np.load(os.path.join('/home/szhang/Riboseq/r64/Pop14/data/test_data','y_test.npy'))

        # initilize X array
        X_train_w2v = np.zeros([len(X_train), w2v_len, 1, len(X_train[0])/3]).astype(np.float32)
        X_val_w2v = np.zeros([len(X_val), w2v_len, 1, len(X_val[0])/3]).astype(np.float32)
        X_test_w2v = np.zeros([len(X_test), w2v_len, 1, len(X_test[0])/3]).astype(np.float32)
        # construct word2vec representations
        scnt = 0
        for s in X_train:
            seq_vec = np.zeros([len(X_train[0])/3, w2v_len])
            codon = ''
            ccnt = 1
            for c in s:
                if ccnt%3 != 0:
                    codon = codon + c
                    ccnt += 1
                else:
                    codon = codon + c
                    seq_vec[ccnt/3 - 1, :] = model[codon]
                    ccnt += 1
                    codon = ''
            seq_vec = seq_vec.transpose()
            X_train_w2v[scnt,:, 0, :] = seq_vec
            scnt += 1
        y_train = np.array(y_train).astype(np.int)
        scnt = 0
        for s in X_val:
            seq_vec = np.zeros([len(X_val[0])/3, w2v_len])
            codon = ''
            ccnt = 1
            for c in s:
                if ccnt%3 != 0:
                    codon = codon + c
                    ccnt += 1
                else:
                    codon = codon + c
                    seq_vec[ccnt/3 - 1, :] = model[codon]
                    ccnt += 1
                    codon = ''
            seq_vec = seq_vec.transpose()
            X_val_w2v[scnt,:, 0, :] = seq_vec
            scnt += 1
        y_val = np.array(y_val).astype(np.int)
        scnt = 0
        for s in X_test:
            seq_vec = np.zeros([len(X_test[0])/3, w2v_len])
            codon = ''
            ccnt = 1
            for c in s:
                if ccnt%3 != 0:
                    codon = codon + c
                    ccnt += 1
                else:
                    codon = codon + c
                    seq_vec[ccnt/3 - 1, :] = model[codon]
                    ccnt += 1
                    codon = ''
            seq_vec = seq_vec.transpose()
            X_test_w2v[scnt,:, 0, :] = seq_vec
            scnt += 1
        y_test = np.array(y_test).astype(np.int)
          
        # random shuffling
        # train data
        indices = np.arange(len(y_train))
        np.random.shuffle(indices)
        X_train_w2v = X_train_w2v[indices]
        y_train = y_train[indices]
        # validation data
        indices = np.arange(len(y_val))
        np.random.shuffle(indices)
        X_val_w2v = X_val_w2v[indices]
        y_val = y_val[indices]
        # test data
        indices = np.arange(len(y_test))
        np.random.shuffle(indices)
        X_test_w2v = X_test_w2v[indices]
        y_test = y_test[indices]
        print len(list(y_test))
        # construct h5py file for caffe input
        with h5py.File('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/train.h5', 'w') as f:
            f['data'] = X_train_w2v
            f['label'] = y_train
        with h5py.File('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/val.h5', 'w') as f:
            f['data'] = X_val_w2v
            f['label'] = y_val
        with h5py.File('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/test.h5', 'w') as f:
            f['data'] = X_test_w2v
            f['label'] = y_test
        with open('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/train.txt', 'w') as f:
            f.write('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/train.h5\n')
        with open('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/val.txt', 'w') as f:
            f.write('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/val.h5\n')
        with open('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/test.txt', 'w') as f:
            f.write('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/test.h5\n')

        # run cnn and return result in test data (with best validation result)
        tlabel, tscore = ribo_convnet.ribo_cnn_net2(len(y_train), len(y_val), len(y_test), val_file, test_file, solver_file, mnum)
        tlabel = np.asarray(tlabel)
        tscore = np.asarray(tscore)
        if mnum == 0:
            label = tlabel
            score = tscore
        else:
            wfile = open('/home/szhang/Riboseq/r64/Pop14/model/log.txt', 'a')
            wfile.write(str(mnum)+' '+str(sum(abs(label-tlabel)))+'\n')
            wfile.close()
            score = score + tscore

    # write best model to result file
    auc = roc_auc_score(label, score/model_num)
    print('AUC is: '+str(auc))
    np.save('/home/szhang/Riboseq/r64/Pop14/model/Pop14_label_test_64', label)
    np.save('/home/szhang/Riboseq/r64/Pop14/model/Pop14_prob_test_64', score/model_num)


