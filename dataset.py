# -*- coding: utf-8 -*-
from __future__ import print_function,division

import numpy as np
from scipy import sparse
import os
import pdb

# np.random.seed(2019)

def select_from_one_class(y_train,prob_pi,label,ratio):
    # select positive and negative samples respectively
    num_sample = y_train[y_train==label].shape[0]
    all_idx = np.arange(y_train.shape[0])[y_train==label]
    label_prob_pi = prob_pi[all_idx]
    obj_sample_size = int(ratio * num_sample)

    sb_idx = None
    iteration = 0
    while True:
        rand_prob = np.random.rand(num_sample)
        iter_idx = all_idx[rand_prob < label_prob_pi]
        if sb_idx is None:
            sb_idx = iter_idx
        else:
            new_idx = np.setdiff1d(iter_idx, sb_idx)
            diff_size = obj_sample_size - sb_idx.shape[0]
            if new_idx.shape[0] < diff_size:
                sb_idx = np.union1d(iter_idx, sb_idx)
            else:
                new_idx = np.random.choice(new_idx, diff_size, replace=False)
                sb_idx = np.union1d(sb_idx, new_idx)
        iteration += 1
        if sb_idx.shape[0] >= obj_sample_size:
            sb_idx = np.random.choice(sb_idx,obj_sample_size,replace=False)
            return sb_idx

        if iteration > 100:
            diff_size = obj_sample_size - sb_idx.shape[0]
            leave_idx = np.setdiff1d(all_idx, sb_idx)
            # left samples are sorted by their IF
            # leave_idx = leave_idx[np.argsort(prob_pi[leave_idx])[-diff_size:]]
            leave_idx = np.random.choice(leave_idx,diff_size,replace=False)
            sb_idx = np.union1d(sb_idx, leave_idx)
            return sb_idx

def load_data_v1(dataset_name,va_ratio=0.1):
    """Set validation ratio to get tr va and te.
    """
    if dataset_name == "criteo1%":
        train_set = np.load("data/criteo.tr.r100.gbdt0.ffm.npy").item()
        te_set = np.load("data/criteo.va.r100.gbdt0.ffm.npy").item()
        x_train = train_set["data"].tocsr()
        y_train = np.array(train_set["label"]).flatten().astype(int)
        x_va = te_set["data"].tocsr()
        y_va = np.array(te_set["label"]).flatten().astype(int)
        two_label = np.unique(train_set["label"])
        y_train[y_train == two_label[0]] = 0
        y_train[y_train == two_label[1]] = 1
        y_va[y_va == two_label[0]] = 0
        y_va[y_va == two_label[1]] = 1

    elif dataset_name == "a1a":
        train_set = np.load("data/a1a.tr.npy").item()
        va_set = np.load("data/a1a.va.npy").item()
        x_train = train_set["data"].tocsr()
        y_train = np.array(train_set["label"]).flatten().astype(int)    
        x_va = va_set["data"].tocsr()
        y_va = np.array(va_set["label"]).flatten().astype(int)       
        two_label = np.unique(train_set["label"])
        y_train[y_train == two_label[0]] = 0
        y_train[y_train == two_label[1]] = 1
        y_va[y_va == two_label[0]] = 0
        y_va[y_va == two_label[1]] = 1
    
    elif dataset_name == "criteo":
        train_set = np.load("data/criteo.tr.r100.gbdt0.ffm.npy").item()
        va_set = np.load("data/criteo.va.r100.gbdt0.ffm.npy").item()
        x_train = train_set["data"].tocsr()
        y_train = np.array(train_set["label"]).flatten().astype(int)    
        x_va = va_set["data"].tocsr()
        y_va = np.array(va_set["label"]).flatten().astype(int)       
        two_label = np.unique(train_set["label"])
        y_train[y_train == two_label[0]] = 0
        y_train[y_train == two_label[1]] = 1
        y_va[y_va == two_label[0]] = 0
        y_va[y_va == two_label[1]] = 1

    elif dataset_name == "news20":
        train_set = np.load("data/news20.binary.npy").item()
        train_set_data = train_set["data"].tocsr()
        train_set_label = train_set["label"].flatten().astype(int)
        two_label = np.unique(train_set["label"])
        train_set_label[train_set_label==two_label[0]] = 0
        train_set_label[train_set_label==two_label[1]] = 1
        # train_index = np.random.choice(np.arange(train_set_data.shape[0]),15000,replace=False)
        train_index = np.arange(train_set_data.shape[0])[:15000]
        x_train = train_set_data[train_index]
        y_train = train_set_label[train_index]
        va_index = np.setdiff1d(np.arange(train_set_data.shape[0]),train_index)
        x_va = train_set_data[va_index]
        y_va = train_set_label[va_index]

    elif dataset_name == "covtype":
        train_set = np.load("data/covtype.libsvm.binary.scale.npy").item()
        train_set_data = train_set["data"].tocsr()
        train_set_label = train_set["label"].flatten().astype(int)
        two_label = np.unique(train_set["label"])
        train_set_label[train_set_label==two_label[0]] = 0
        train_set_label[train_set_label==two_label[1]] = 1
        # train_index = np.random.choice(np.arange(train_set_data.shape[0]),400000,replace=False)
        train_index = np.arange(train_set_data.shape[0])[:400000]
        x_train = train_set_data[train_index]
        y_train = train_set_label[train_index]
        va_index = np.setdiff1d(np.arange(train_set_data.shape[0]),train_index)
        x_va = train_set_data[va_index]
        y_va = train_set_label[va_index]

    elif dataset_name == "mnist":
        x_train,y_train,x_test,y_test = load_mnist()
        pos_class = 1
        neg_class = 7
        x_train,y_train = filter_dataset(x_train,y_train,pos_class,neg_class)
        x_va,y_va = filter_dataset(x_test,y_test,pos_class,neg_class)
        y_va = y_va.astype(int)
        y_train = y_train.astype(int)

    elif dataset_name == "cancer":
        train_set = np.load("data/breast-cancer_scale.npy").item()
        train_set_data = train_set["data"].tocsr()
        train_set_label = train_set["label"].flatten().astype(int)
        two_label = np.unique(train_set["label"])
        train_set_label[train_set_label==two_label[0]] = 0
        train_set_label[train_set_label==two_label[1]] = 1
        # train_index = np.random.choice(np.arange(train_set_data.shape[0]),500,replace=False)
        train_index = np.arange(train_set_data.shape[0])[:500]    
        x_train = train_set_data[train_index]
        y_train = train_set_label[train_index]
        va_index = np.setdiff1d(np.arange(train_set_data.shape[0]),train_index)
        x_va = train_set_data[va_index]
        y_va = train_set_label[va_index]

    elif dataset_name == "diabetes":
        train_set = np.load("data/diabetes_scale.npy").item()
        train_set_data = train_set["data"].tocsr()
        train_set_label = train_set["label"].flatten().astype(int)
        two_label = np.unique(train_set["label"])
        train_set_label[train_set_label==two_label[0]] = 0
        train_set_label[train_set_label==two_label[1]] = 1
        # train_index = np.random.choice(np.arange(train_set_data.shape[0]),500,replace=False)
        train_index = np.arange(train_set_data.shape[0])[:500]
        x_train = train_set_data[train_index]
        y_train = train_set_label[train_index]
        va_index = np.setdiff1d(np.arange(train_set_data.shape[0]),train_index)
        x_va = train_set_data[va_index]
        y_va = train_set_label[va_index]

    elif dataset_name == "skin":
        train_set = np.load("data/skin_nonskin.npy").item()
        train_set_data = train_set["data"].tocsr()
        train_set_label = train_set["label"].flatten().astype(int)
        two_label = np.unique(train_set["label"])
        train_set_label[train_set_label==two_label[0]] = 0
        train_set_label[train_set_label==two_label[1]] = 1
        # train_index = np.random.choice(np.arange(train_set_data.shape[0]),200000,replace=False)
        train_index = np.arange(train_set_data.shape[0])[:200000]
        x_train = train_set_data[train_index]
        y_train = train_set_label[train_index]
        va_index = np.setdiff1d(np.arange(train_set_data.shape[0]),train_index)
        x_va = train_set_data[va_index]
        y_va = train_set_label[va_index]

    elif dataset_name == "realsim":
        train_set = np.load("data/real-sim.npy").item()
        train_set_data = train_set["data"].tocsr()
        train_set_label = train_set["label"].flatten().astype(int)
        two_label = np.unique(train_set["label"])
        train_set_label[train_set_label==two_label[0]] = 0
        train_set_label[train_set_label==two_label[1]] = 1
        # train_index = np.random.choice(np.arange(train_set_data.shape[0]),60000,replace=False)
        train_index = np.arange(train_set_data.shape[0])[:60000]
        x_train = train_set_data[train_index]
        y_train = train_set_label[train_index]
        va_index = np.setdiff1d(np.arange(train_set_data.shape[0]),train_index)
        x_va = train_set_data[va_index]
        y_va = train_set_label[va_index]

    elif dataset_name == "phishing":
        train_set = np.load("data/phishing.npy").item()
        train_set_data = train_set["data"].tocsr()
        train_set_label = train_set["label"].flatten().astype(int)
        two_label = np.unique(train_set["label"])
        train_set_label[train_set_label==two_label[0]] = 0
        train_set_label[train_set_label==two_label[1]] = 1
        # train_index = np.random.choice(np.arange(train_set_data.shape[0]),8000,replace=False)
        train_index = np.arange(train_set_data.shape[0])[:8000]
        x_train = train_set_data[train_index]
        y_train = train_set_label[train_index]
        va_index = np.setdiff1d(np.arange(train_set_data.shape[0]),train_index)
        x_va = train_set_data[va_index]
        y_va = train_set_label[va_index]

    elif dataset_name == "cifar10":
        train_set = np.load("data/cifar10.npy").item()
        test_set = np.load("data/cifar10.t.npy").item()
        x_train = train_set["data"]
        x_train = x_train / 255.0
        y_train = train_set["label"]
        x_va = test_set["data"]
        x_va = x_va / 255.0
        y_va = test_set["label"]
        # cat : 3, dog : 5
        pos_class = 3
        neg_class = 5
        x_train,y_train = filter_dataset(x_train,y_train,pos_class,neg_class)
        x_va,y_va = filter_dataset(x_va,y_va,pos_class,neg_class)
        y_va = y_va.astype(int)
        y_train = y_train.astype(int)

    elif dataset_name == "svhn":
        train_set = np.load("data/SVHN.scale.npy").item()
        test_set = np.load("data/SVHN.scale.t.npy").item()
        x_train = train_set["data"]
        y_train = train_set["label"]
        x_va = test_set["data"]
        y_va = test_set["label"]
        pos_class = 1
        neg_class = 7
        x_train, y_train = filter_dataset(x_train, y_train, pos_class, neg_class)
        x_va, y_va = filter_dataset(x_va, y_va, pos_class, neg_class)
        y_va = y_va.astype(int)
        y_train = y_train.astype(int)

    else:
        print("Cannot find the dataset {}, quit.".format(dataset_name))
        return

    # split tr and va
    # num_va_sample = int(va_ratio * x_va.shape[0])
    # vate_idx = np.arange(x_va.shape[0])
    # va_idx = np.random.choice(vate_idx, num_va_sample, replace=False)
    # te_idx = np.setdiff1d(vate_idx, va_idx)
    # x_val = x_va[va_idx]
    # y_val = y_va[va_idx]
    # x_te = x_va[te_idx]
    # y_te = y_va[te_idx]

    num_va_sample = int((1-va_ratio) * x_train.shape[0])
    x_val = x_train[num_va_sample:]
    y_val = y_train[num_va_sample:]
    x_train = x_train[:num_va_sample]
    y_train = y_train[:num_va_sample]
    x_te = x_va
    y_te = y_va

    return x_train,y_train,x_val,y_val,x_te,y_te

def load_data(dataset_name):
    if dataset_name == "avazu_app":
        train_set = np.load("data/avazu-app.tr.npy").item()
        va_set = np.load("data/avazu-app.val.npy").item()
        x_train = train_set["data"].tocsr()
        y_train = np.array(train_set["label"]).flatten().astype(int)
        x_va = va_set["data"].tocsr()
        y_va = np.array(va_set["label"]).flatten().astype(int) 
        two_label = np.unique(train_set["label"])
        y_train[y_train == two_label[0]] = 0
        y_train[y_train == two_label[1]] = 1
        y_va[y_va == two_label[0]] = 0
        y_va[y_va == two_label[1]] = 1

    elif dataset_name == "a1a":
        train_set = np.load("data/a1a.tr.npy").item()
        va_set = np.load("data/a1a.va.npy").item()
        x_train = train_set["data"].tocsr()
        y_train = np.array(train_set["label"]).flatten().astype(int)    
        x_va = va_set["data"].tocsr()
        y_va = np.array(va_set["label"]).flatten().astype(int)       
        two_label = np.unique(train_set["label"])
        y_train[y_train == two_label[0]] = 0
        y_train[y_train == two_label[1]] = 1
        y_va[y_va == two_label[0]] = 0
        y_va[y_va == two_label[1]] = 1
    
    elif dataset_name == "criteo1%":
        train_set = np.load("data/criteo.tr.r100.gbdt0.ffm.npy").item()
        va_set = np.load("data/criteo.va.r100.gbdt0.ffm.npy").item()
        x_train = train_set["data"].tocsr()
        y_train = np.array(train_set["label"]).flatten().astype(int)    
        x_va = va_set["data"].tocsr()
        y_va = np.array(va_set["label"]).flatten().astype(int)       
        two_label = np.unique(train_set["label"])
        y_train[y_train == two_label[0]] = 0
        y_train[y_train == two_label[1]] = 1
        y_va[y_va == two_label[0]] = 0
        y_va[y_va == two_label[1]] = 1

    elif dataset_name == "news20":
        train_set = np.load("data/news20.binary.npy").item()
        train_set_data = train_set["data"].tocsr()
        train_set_label = train_set["label"].flatten().astype(int)
        two_label = np.unique(train_set["label"])
        train_set_label[train_set_label==two_label[0]] = 0
        train_set_label[train_set_label==two_label[1]] = 1
        train_index = np.random.choice(np.arange(train_set_data.shape[0]),15000,replace=False)
        x_train = train_set_data[train_index]
        y_train = train_set_label[train_index]
        va_index = np.setdiff1d(np.arange(train_set_data.shape[0]),train_index)
        x_va = train_set_data[va_index]
        y_va = train_set_label[va_index]

    elif dataset_name == "covtype":
        train_set = np.load("data/covtype.libsvm.binary.scale.npy").item()
        train_set_data = train_set["data"].tocsr()
        train_set_label = train_set["label"].flatten().astype(int)
        two_label = np.unique(train_set["label"])
        train_set_label[train_set_label==two_label[0]] = 0
        train_set_label[train_set_label==two_label[1]] = 1
        train_index = np.random.choice(np.arange(train_set_data.shape[0]),15000,replace=False)
        x_train = train_set_data[train_index]
        y_train = train_set_label[train_index]
        va_index = np.setdiff1d(np.arange(train_set_data.shape[0]),train_index)
        x_va = train_set_data[va_index]
        y_va = train_set_label[va_index]

    else:
        print("Cannot find the dataset {}, quit.".format(dataset_name))
        return

    return x_train,y_train,x_va,y_va


# tool box
def load_mnist(validation_size = 5000):
    import gzip
    def _read32(bytestream):
        dt = np.dtype(np.uint32).newbyteorder(">")
        return np.frombuffer(bytestream.read(4),dtype=dt)[0]

    def extract_images(f):
        print("Extracting",f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf,dtype=np.uint8)
            data = data.reshape(num_images,rows,cols,1)
            return data
    
    def extract_labels(f):
        print('Extracting', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            num_items = _read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return labels

    data_dir = "./data"
    TRAIN_IMAGES = os.path.join(data_dir,'train-images-idx3-ubyte.gz')
    with open(TRAIN_IMAGES,"rb") as f:
        train_images = extract_images(f)

    TRAIN_LABELS =  os.path.join(data_dir,'train-labels-idx1-ubyte.gz')
    with open(TRAIN_LABELS,"rb") as f:
        train_labels = extract_labels(f)

    TEST_IMAGES =  os.path.join(data_dir,'t10k-images-idx3-ubyte.gz')
    with open(TEST_IMAGES,"rb") as f:
        test_images = extract_images(f)

    TEST_LABELS =  os.path.join(data_dir,'t10k-labels-idx1-ubyte.gz')
    with open(TEST_LABELS,"rb") as f:
        test_labels = extract_labels(f)

    # split train and val
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    # preprocessing
    train_images = train_images.astype(np.float32) / 255
    test_images  = test_images.astype(np.float32) / 255
    
    # reshape for logistic regression
    train_images = np.reshape(train_images, [train_images.shape[0], -1])
    test_images = np.reshape(test_images, [test_images.shape[0], -1])
    return train_images,train_labels,test_images,test_labels

def filter_dataset(X, Y, pos_class, neg_class, mode=None):
    """
    Filters out elements of X and Y that aren't one of pos_class or neg_class
    then transforms labels of Y so that +1 = pos_class, -1 = neg_class.
    """
    assert(X.shape[0] == Y.shape[0])
    assert(len(Y.shape) == 1)

    Y = Y.astype(int)
    
    pos_idx = Y == pos_class
    neg_idx = Y == neg_class        
    Y[pos_idx] = 1
    Y[neg_idx] = -1
    idx_to_keep = pos_idx | neg_idx
    X = X[idx_to_keep, ...]
    Y = Y[idx_to_keep]
    if Y.min() == -1 and mode != "svm":
        Y = (Y + 1) / 2
        Y.astype(int)
    return (X, Y)