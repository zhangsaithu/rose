import os
import sys
import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
from sklearn.metrics import roc_auc_score
import random

def ribo_cnn_net1(train_size, val_size, test_size, val_file, test_file, solver_file):
    #random.seed(1024)
    os.chdir('..')
    sys.path.insert(0, './python')
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_file)
    niter = 100000
    test_interval = 500

    # losses will also be stored in the log
    train_loss = np.zeros(niter)
    test_auc = []

    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe    
        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data        
        # run a full test every so often
        # Caffe can also do this for us and write to a log, but we show here
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            prob = [] # predicted probability
            label = [] # true label
            # calculate validation auc
            for test_it in range(val_size/256+1):
                solver.test_nets[0].forward()
                output = solver.test_nets[0].blobs['ip2'].data
                output_label = solver.test_nets[0].blobs['label'].data              
                prob.extend(list(np.divide(np.exp(output[:,1]), (np.exp(output[:,0])+np.exp(output[:,1])))))
                label.extend(list(output_label))
            test_auc.append(roc_auc_score(label, prob))

    # get the best model
    maxv = max(test_auc)
    maxp = test_auc.index(maxv)
    maxi = maxp * test_interval
    best_model = '/home/szhang/Riboseq/r64/Pop14/model_file/ribo_iter_' + str(maxi) + '.caffemodel'
    net_t = caffe.Net(test_file, best_model, caffe.TEST)

    # calculate auc score of test data
    prob = []
    label = []
    for test_it in range(test_size/1902):
        net_t.forward()
        output = net_t.blobs['ip2'].data
        output_label = net_t.blobs['label'].data              
        prob.extend(list(np.divide(np.exp(output[:,1]), (np.exp(output[:,0])+np.exp(output[:,1])))))
        label.extend(list(output_label))

    # return best validation and test auc scores
    #return maxv, roc_auc_score(label, prob)
    os.rename(best_model, '/home/szhang/Riboseq/r64/Pop14/model_file/best_model.caffemodel')
    return label, prob

def ribo_cnn_net2(train_size, val_size, test_size, val_file, test_file, solver_file, mnum):
    #random.seed(1024)
    os.chdir('..')
    sys.path.insert(0, './python')
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_file)
    niter = 100000
    test_interval = 500

    # losses will also be stored in the log
    train_loss = np.zeros(niter)
    test_auc = []

    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe    
        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data        
        # run a full test every so often
        # Caffe can also do this for us and write to a log, but we show here
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            prob = [] # predicted probability
            label = [] # true label
            # calculate validation auc
            for test_it in range(val_size/256+1):
                solver.test_nets[0].forward()
                output = solver.test_nets[0].blobs['ip2'].data
                output_label = solver.test_nets[0].blobs['label'].data              
                prob.extend(list(np.divide(np.exp(output[:,1]), (np.exp(output[:,0])+np.exp(output[:,1])))))
                label.extend(list(output_label))
            test_auc.append(roc_auc_score(label, prob))

    # get the best model
    maxv = max(test_auc)
    maxp = test_auc.index(maxv)
    maxi = maxp * test_interval
    best_model = '/home/szhang/Riboseq/r64/Pop14/model_file/ribo_iter_' + str(maxi) + '.caffemodel'
    net_t = caffe.Net(test_file, best_model, caffe.TEST)

    # calculate auc score of test data
    prob = []
    label = []
    for test_it in range(test_size/1902):
        net_t.forward()
        output = net_t.blobs['ip2'].data
        output_label = net_t.blobs['label'].data              
        prob.extend(list(np.divide(np.exp(output[:,1]), (np.exp(output[:,0])+np.exp(output[:,1])))))
        label.extend(list(output_label))

    # return best validation and test auc scores
    #return maxv, roc_auc_score(label, prob)
    os.rename(best_model, '/home/szhang/Riboseq/r64/Pop14/model_file/best_model_'+str(mnum)+'.caffemodel')
    return label, prob

def cv_test_cnn(test_size, trained_model, test_file):
    #random.seed(1024)
    os.chdir('..')
    sys.path.insert(0, './python')
    caffe.set_device(0)
    caffe.set_mode_gpu()
    
    best_model = trained_model
    net_t = caffe.Net(test_file, best_model, caffe.TEST)

    # calculate auc score of test data
    prob = []
    label = []
    for test_it in range(1):
        net_t.forward()
        #output1 = net_t.blobs['conv1'].data
        #output2 = net_t.blobs['conv2'].data
        #output3 = net_t.blobs['conv3'].data
        #output_label = net_t.blobs['label'].data   
        output = net_t.blobs['ip2'].data           
        prob.extend(list(np.divide(np.exp(output[:,1]), (np.exp(output[:,0])+np.exp(output[:,1])))))
        #label.extend(list(output_label))

    #return output1, output2, output3
    return np.asarray(prob)

