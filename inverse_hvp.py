# -*- coding: utf-8 -*-
"""Do inverse hessian-vector-product.
"""
from optimize.optimize import fmin_ncg
# from scipy.optimize import fmin_ncg

import pdb
import time
import numpy as np
from grad_utils import grad_logloss_theta_lr, hessian_logloss_theta_lr, hessian_vector_product_lr

def inverse_hvp_lissa(x_train,y_train,y_pred,v,
    batch_size=100,repeat=10,max_recursion_depth=10,
    l2_norm=0.01,tol=1e-6,hessian_free=True):
    """Get inverse hessian-vector-products H^-1 * v with stochastic esimation:
    linear (time) stochastic second-order algorithm, LISSA
    this method is suitable for the large dataset and useful for broad algorithms.
    Refers to 
    `Second-order Stochastic Optimization for Machine Learning in Linear Time 2017 JMLR` .
    """
    start_time = time.time()
    inverse_hvp = None
    for r in range(repeat):
        # initialize H_0 ^ -1 * v = v begin with each repeat.
        current_estimate = v
        for j in range(max_recursion_depth):
            batch_idx = np.random.choice(np.arange(x_train.shape[0]),size=batch_size)
            if hessian_free:
                hessian_vector_val = hessian_vector_product_lr(y_train[batch_idx],
                    y_pred[batch_idx],x_train[batch_idx],
                    current_estimate,l2_norm)
            else:
                hessian_matrix = hessian_logloss_theta_lr(y_train[batch_idx],
                    y_pred[batch_idx],x_train[batch_idx],l2_norm)
                hessian_vector_val = np.dot(current_estimate,hessian_matrix)
            
            current_estimate_new = v + current_estimate - hessian_vector_val

            diffs = np.linalg.norm(current_estimate_new) - np.linalg.norm(current_estimate)
            diffs = diffs / np.linalg.norm(current_estimate)
            if diffs <= tol:
                current_estimate = current_estimate_new
                print("Break in depth {}".format(str(j)))
                break
            current_estimate = current_estimate_new

        print("Repeat at {} times: norm is {:.2f}".format(r, 
                np.linalg.norm(current_estimate)))

        if inverse_hvp is None:
            inverse_hvp = current_estimate
        else:
            inverse_hvp = inverse_hvp + current_estimate

    # average
    inverse_hvp = inverse_hvp / float(repeat)
    print("Inverse HVP took {:.1f} sec".format(time.time() - start_time))
    return inverse_hvp

def inverse_hvp_lr_newtonCG(x_train,y_train,y_pred,v,C=0.01,hessian_free=True,tol=1e-5,has_l2=True,M=None,scale_factor=1.0):
    """Get inverse hessian-vector-products H^-1 * v, this method is not suitable for
    the large dataset.
    Args:
        x_train, y_train: training data used for computing the hessian, e.g. x_train: [None,n]
        y_pred: predictions made on x_train, e.g. [None,]
        v: value vector, e.g. [n,]
        hessian_free: bool, `True` means use implicit hessian-vector-product to avoid
            building hessian directly, `False` will build hessian.
            hessian free will save memory while be slower in computation, vice versa.
            such that set `True` when cope with large dataset, and set `False` with 
            relatively small dataset.
    Return:
        H^-1 * v: shape [None,]
    """
    if not hessian_free:
        hessian_matrix = hessian_logloss_theta_lr(y_train,y_pred,x_train,C,has_l2,scale_factor)

    # build functions for newton-cg optimization
    def fmin_loss_fn(x):
        """Objective function for newton-cg.
        H^-1 * v = argmin_t {0.5 * t^T * H * t - v^T * t}
        """
        if hessian_free:
            hessian_vec_val = hessian_vector_product_lr(y_train,y_pred,x_train,x,C,has_l2,scale_factor) # [n,]
        else:
            hessian_vec_val = np.dot(x,hessian_matrix) # [n,]
        obj = 0.5 * np.dot(hessian_vec_val,x) - \
                    np.dot(x, v)

        return obj

    def fmin_grad_fn(x):
        """Gradient of the objective function w.r.t t:
        grad(obj) = H * t - v
        """
        if hessian_free:
            hessian_vec_val = hessian_vector_product_lr(y_train,y_pred,x_train,x,C,has_l2,scale_factor)
        else:
            hessian_vec_val = np.dot(x,hessian_matrix) # [n,]

        grad = hessian_vec_val - v

        return grad

    def get_fmin_hvp(x,p):
        # get H * p
        if hessian_free:
            hessian_vec_val = hessian_vector_product_lr(y_train,y_pred,x_train,p,C,has_l2,scale_factor)
        else:
            hessian_vec_val = np.dot(p,hessian_matrix)

        return hessian_vec_val

    def get_cg_callback(verbose):
        def fmin_loss_split(x):
            if hessian_free:
                hessian_vec_val = hessian_vector_product_lr(y_train,y_pred,x_train,x,C,has_l2,scale_factor)
            else:
                hessian_vec_val = np.dot(x,hessian_matrix)

            loss_1 = 0.5 * np.dot(hessian_vec_val,x)
            loss_2 = - np.dot(v, x)
            return loss_1, loss_2

        def cg_callback(x):
            # idx_to_remove = 5
            # xs = x_train[idx_to_remove]
            # label = y_train[idx_to_remove]
            # ys = y_pred[idx_to_remove]

            # train_grad_loss_val = grad_logloss_theta_lr(label,ys,xs.reshape(1,-1))
            # predicted_loss_diff = np.dot(x,train_grad_loss_val) / x_train.shape[0]

            if verbose:
                print("Function value:", fmin_loss_fn(x))
                quad, lin = fmin_loss_split(x)
                print("Split function value: {}, {}".format(quad, lin))
                # print("Predicted loss diff on train_idx {}: {}".format(idx_to_remove, predicted_loss_diff))

        return cg_callback

    start_time = time.time()
    cg_callback = get_cg_callback(verbose=True)
    fmin_results = fmin_ncg(f=fmin_loss_fn,
                           x0=v,
                           fprime=fmin_grad_fn,
                           fhess_p=get_fmin_hvp,
                           callback=cg_callback,
                           avextol=tol,
                           maxiter=100,
                           preconditioner=M)

    print("implicit hessian-vector products mean:",fmin_results.mean())
    print("implicit hessian-vector products norm:",np.linalg.norm(fmin_results))
    print("Inverse HVP took {:.1f} sec".format(time.time() - start_time))
    return fmin_results


# debug
def main():
    from load_mnist import load_mnist, filter_dataset
    from image_utils import plot_flat_colorimage, plot_top_influence_colorimage
    from print_utils import print_table
    # load raw mnist
    x_train,y_train,x_test,y_test = load_mnist()
    # filter 1 and 7
    pos_class = 1
    neg_class = 7
    num_class = 2
    test_indices = 20
    x_train,y_train = filter_dataset(x_train,y_train,pos_class,neg_class)
    x_test,y_test = filter_dataset(x_test,y_test,pos_class,neg_class)

    # train logistic regression with LBGFS
    from sklearn import linear_model
    max_iter = 1000
    C = 1.0 / (x_train.shape[0] * 0.01)
    sklearn_model = linear_model.LogisticRegression(
        C= C,
        tol = 1e-8,
        fit_intercept=False,
        solver="lbfgs",
        multi_class="auto",
        warm_start=True,
        max_iter=max_iter,
        )

    sklearn_model.fit(x_train,y_train)

    # get test gradient loss value
    test_pred = sklearn_model.predict_proba(x_test[test_indices].reshape(1,-1))[:,1]
    test_grad_loss_val = grad_logloss_theta_lr(y_test[test_indices],test_pred,x_test[test_indices].reshape(1,-1))
    print("test grad loss norm:",np.linalg.norm(test_grad_loss_val))

    y_pred = sklearn_model.predict_proba(x_train)[:,1]

    "Get inverse hvp"
    # inverse_hvp = inverse_hvp_lissa(x_train,y_train,y_pred,test_grad_loss_val,100,5,200)
    inverse_hvp = inverse_hvp_lr_newtonCG(x_train,y_train,y_pred,test_grad_loss_val,0.01,False)

    start_time = time.time()
    num_tr_sample = x_train.shape[0]
    train_idx = np.arange(num_tr_sample)

    predicted_loss_diff = []
    for idx in range(num_tr_sample):
        train_grad_loss_val = grad_logloss_theta_lr(y_train[idx],y_pred[idx],x_train[idx].reshape(1,-1))
        predicted_loss_diff.append(
            np.dot(inverse_hvp, train_grad_loss_val) / num_tr_sample
            )
    predicted_loss_diffs = np.asarray(predicted_loss_diff)
    duration = time.time() - start_time
    print("Multiplying by {} train examples took {:.1f} sec".format(num_tr_sample, duration))
    print("Attribute predicted_loss_diffs, mean {}, max {}, min {}".format(
        predicted_loss_diffs.mean(), predicted_loss_diffs.max(), predicted_loss_diffs.min())
    )
    print("Test image:")
    print(y_test[test_indices])
    plot_flat_colorimage(x_test[test_indices],y_test[test_indices],28)

    print("Top from predicted influence:")
    plot_top_influence_colorimage(x_train,y_train,predicted_loss_diffs,top_n=5,ascending=True)
    print("Top harmful from predicted influence:")
    plot_top_influence_colorimage(x_train,y_train,predicted_loss_diffs,top_n=5,ascending=False)
    columns = ["idx","label","influence"]
    rows = []
    for counter,train_idx in enumerate(np.argsort(predicted_loss_diffs)[-5:]):
        rows.append([train_idx,y_train[train_idx],predicted_loss_diffs[train_idx]])

    print_table(columns,rows)
    return

if __name__ == '__main__':
    main()









