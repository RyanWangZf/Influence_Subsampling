# -*- coding: utf-8 -*-
"""Do gradient without autograd from tensorflow.
"""
import numpy as np
import pdb
from scipy import sparse

"""Logistic regression
"""
# This scale factor makes the newton cg converges faster, generally selected as lambda ~ 1 / (C*nr_sample)
# here we have nr_sample ~ 10^7, C = 0.1
# thus we have lambda ~ 1e-6

def batch_grad_logloss_lr(label,ypred,x,weight_ar=None,C=0.03,has_l2=True,scale_factor=1.0):
    """Return gradient on a batch.
    Args:
        label, ypred: an array of shape [None,]
        x: an array or sparse.csc_matrix with shape [None,n]
        has_l2: if set False, the weight_ar will be ignored.

    Return:
        batch_grad: gradient of each sample on parameters,
            has shape [None,n]
    """
    diffs = ypred - label

    if isinstance(x,np.ndarray):
        diffs = diffs.reshape(-1,1)
        batch_grad = x * diffs
    else:
        diffs = sparse.diags(diffs)
        batch_grad = x.T.dot(diffs).T

    if weight_ar is not None:
        # clip the feature index which is not seen in training set
        weight_len = weight_ar.shape[0]
        if x.shape[1] > weight_len:
            x = x[:, :weight_len]

    if has_l2:
        batch_grad = C * batch_grad + weight_ar
    else:
        batch_grad = sparse.csr_matrix(C * batch_grad)
        
    return scale_factor * batch_grad

def grad_logloss_theta_lr(label,ypred,x,weight_ar=None,C=0.03,has_l2=True,scale_factor=1.0):
    """Return d l_i / d_theta = d l_i / d_ypred * d y_pred / d theta
    Args:
        label: a scalar (one sample) or an array of shape [None,]
        ypred: a scalar (one sample) or an array of shape [None,]
        x: an array with shape [None,n], or sparse.csc_matrix object.
        weight_ar: an array with shape [n,].
        C: the parameter set in the objective function
    Return:
        grad_logloss_theta: gradient on the theta, shape: [n,]
    """
    # Complex approach
    # grad_logloss_ypred = (1 - label) / (1 - ypred + 1e-10) - label / (ypred + 1e-10)
    # grad_ypred_theta = ypred * (1 - ypred) * x
    # grad_logloss_theta = grad_logloss_ypred * grad_ypred_theta

    # if there is only one sample in this batch
    if not isinstance(label,np.ndarray) or not isinstance(ypred,np.ndarray):
        label = np.array(label).flatten()
        ypred = np.array(ypred).flatten()

    if weight_ar is not None:
        # clip the feature index which is not seen in training set
        weight_len = weight_ar.shape[0]
        if x.shape[1] > weight_len:
            x = x[:, :weight_len]

    if has_l2:
        grad_logloss_theta = weight_ar + C * x.T.dot(ypred-label)
    else:
        grad_logloss_theta = C * x.T.dot(ypred-label)

    return scale_factor * grad_logloss_theta


def grad_logloss_theta_lr_weighted(label,ypred,x,weight_ar=None,C=0.03,has_l2=True,scale_factor=1.0,weights=None):
    """Return d l_i / d_theta = d l_i / d_ypred * d y_pred / d theta
    Args:
        label: a scalar (one sample) or an array of shape [None,]
        ypred: a scalar (one sample) or an array of shape [None,]
        x: an array with shape [None,n], or sparse.csc_matrix object.
        weight_ar: an array with shape [n,].
        C: the parameter set in the objective function
    Return:
        grad_logloss_theta: gradient on the theta, shape: [n,]
    """
    # Complex approach
    # grad_logloss_ypred = (1 - label) / (1 - ypred + 1e-10) - label / (ypred + 1e-10)
    # grad_ypred_theta = ypred * (1 - ypred) * x
    # grad_logloss_theta = grad_logloss_ypred * grad_ypred_theta

    # if there is only one sample in this batch
    if not isinstance(label,np.ndarray) or not isinstance(ypred,np.ndarray):
        label = np.array(label).flatten()
        ypred = np.array(ypred).flatten()

    if weight_ar is not None:
        # clip the feature index which is not seen in training set
        weight_len = weight_ar.shape[0]
        if x.shape[1] > weight_len:
            x = x[:, :weight_len]

    diffs = ypred-label
    if weights is not None:
        diffs = weights * diffs

    if has_l2:
        grad_logloss_theta = weight_ar + C * x.T.dot(diffs)
    else:
        grad_logloss_theta = C * x.T.dot(diffs)

    return scale_factor * grad_logloss_theta


def hessian_logloss_theta_lr(label,ypred,x,C=0.03,has_l2=True,scale_factor=1.0):
    """Get hessian matrix of logloss on theta.
    Args:
        label: ground truth label of x e.g. [None,]
        ypred: predictions made by logisitic regression e.g. [None,]
        x: features, an np.array has same shape with theta, e.g. [None,n]
        l2_norm: float, if there is l2_loss in logloss term, the hessian has a `I` matrix term.
            we recommend set >0 here because it helps newton_cg method's convergence in practice.
    """
    assert C >= 0.0
    # if there is only one sample in this batch
    if not isinstance(label,np.ndarray) or not isinstance(ypred,np.ndarray):
        label = np.array(label).flatten()
        ypred = np.array(ypred).flatten()
        
    h = x.multiply((ypred * (1 - ypred)).reshape(-1,1))

    if isinstance(x, sparse.coo_matrix):
        h = h.tocsr()
        hessian = C * h.T.dot(x)
    else:
        hessian = C * np.matmul(h.T,x)

    if has_l2:
        diag_idx = np.arange(hessian.shape[0])
        hessian[diag_idx,diag_idx] += 1.0

    return scale_factor * hessian

def hessian_vector_product_lr(label,ypred,x,v,C=0.03,has_l2=True,scale_factor=1.0):
    """Get implicit hessian-vector-products without explicitly building the hessian.
    H*v = v + C *X^T*(D*(X*v))
    """
    xv = x.dot(v)
    D = ypred * (1 - ypred)
    dxv = xv * D
    if has_l2:
        hvp = C * x.T.dot(dxv) +  v
    else:
        hvp = C * x.T.dot(dxv)

    return scale_factor * hvp

if __name__ == '__main__':
    main()