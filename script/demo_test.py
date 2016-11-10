##################################################
# A demo to predict ribosome stalling using ROSE #
##################################################
import sys, os
import re, fileinput, math
import numpy as np
import random
import caffe
import h5py
from gensim.models import word2vec
import ribo_convnet
from sklearn.metrics import roc_auc_score
import glob

# make sure that caffe is on the python path
caffe_root = '/opt/caffe-master/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')

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

# set the file paths first!

X_test = np.load('/home/szhang/Riboseq/r64/Pop14/data/test_data/X_test.npy')
y_test = np.load('/home/szhang/Riboseq/r64/Pop14/data/test_data/y_test.npy')
samsize = len(X_test)
wfile = open('/home/szhang/Riboseq/r64/Pop14/model/net_architecture_cv/train_test.prototxt','w')
lnum = 1
for line in open('/home/szhang/Riboseq/r64/Pop14/model/net_architecture_cv/train_test_backup.prototxt','r'):
    if lnum == 31:
        wfile.write('    batch_size: '+str(samsize)+'\n')
    if lnum == 30:
        wfile.write('    source: "/home/szhang/Riboseq/r64/Pop14/data/caffe_data/test.txt"\n')
    if lnum != 30 and lnum != 31:
        wfile.write(line)
    lnum += 1
wfile.close()

X_test_w2v = np.zeros([len(X_test), w2v_len, 1, len(X_test[0])/3]).astype(np.float32)
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
            if codon[0] not in ['A','T','C','G'] or codon[1] not in ['A','T','C','G'] or codon[2] not in ['A','T','C','G']:
                seq_vec[ccnt/3 - 1, :] = np.zeros(64)
            else:
                seq_vec[ccnt/3 - 1, :] = model[codon]
            ccnt += 1
            codon = ''
    seq_vec = seq_vec.transpose()
    X_test_w2v[scnt,:, 0, :] = seq_vec
    scnt += 1

with h5py.File('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/test.h5', 'w') as f:
    f['data'] = X_test_w2v
    f['label'] = y_test
with open('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/test.txt', 'w') as f:
    f.write('/home/szhang/Riboseq/r64/Pop14/data/caffe_data/test.h5\n')

for i in range(64):
    trained_model = '/home/szhang/Riboseq/r64/Pop14/model_file/best_model_'+str(i)+'.caffemodel'
    test_file = '/home/szhang/Riboseq/r64/Pop14/model/net_architecture_cv/train_test.prototxt'
    test_size = len(X_test)
    if i == 0:
        prob = ribo_convnet.cv_test_cnn(test_size, trained_model, test_file)
    else:
        prob = prob + ribo_convnet.cv_test_cnn(test_size, trained_model, test_file)

prob = prob/64
np.save('/home/szhang/Riboseq/r64/Pop14/data/test_data/y_pred',prob)
print 'AUC: '+str(roc_auc_score(y_test,prob))

