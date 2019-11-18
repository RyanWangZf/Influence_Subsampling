import numpy as np
import pdb
from scipy import sparse

from scipy.optimize import fmin_ncg
import pdb
import time

# for hinge loss SVM
def hessian_vector_product_hinge(label, ypred, x, v, scale_factor=1.0):
    mask = np.ones_like(label)
    mask[ypred * label >= 1] = 0
    mask[ypred * label <= 0] = 0
    t = label * mask
    xv = x.dot(v)
    txv = t.dot(xv)
    hvp = txv * x.T.dot(t)
    hvp = hvp / x.shape[0] * scale_factor
    return hvp

def hessian_hingle_loss_theta(label, ypred, x):
    mask = np.ones_like(label)
    mask[ypred * label >= 1] = 0
    mask[ypred * label <= 0] = 0
    t = label * mask
    tx = x.T.dot(t.reshape(-1,1))
    txxt = tx.dot(tx.T)
    return txxt


def batch_grad_hinge_loss(label, ypred, x):
    if not isinstance(x, sparse.csr_matrix):
        x = sparse.csr_matrix(x)

    mask_1 = np.zeros_like(ypred)
    mask_1[ypred * label <= 0] = 1
    mask_1 = mask_1.astype(int)

    mask_2 = np.logical_xor(np.ones_like(mask_1), mask_1).astype(int)
    mask_3 = np.ones_like(mask_1)
    mask_3[ypred * label > 1] = 0
    mask_3 = mask_3.astype(int)

    val_1 = -x.multiply(label.reshape(-1,1))
    val_2 = -val_1.multiply((1-label*ypred).reshape(-1,1))

    # val_1 = np.asarray(val_1.mean(0)).flatten()
    # val_2 = np.asarray(val_2.mean(0)).flatten()

    val_1 = val_1.multiply(mask_1.reshape(-1,1))
    val_2 = val_2.multiply(mask_2.reshape(-1,1))

    val_1 = val_1.multiply(mask_3.reshape(-1,1))
    val_2 = val_2.multiply(mask_3.reshape(-1,1))

    grad = val_1 + val_2

    return grad

def grad_hinge_loss_theta(label, ypred, x):
    if not isinstance(x, sparse.csr_matrix):
        x = sparse.csr_matrix(x)

    mask_1 = np.zeros_like(ypred)
    mask_1[ypred * label <= 0] = 1
    mask_1 = mask_1.astype(int)

    mask_2 = np.logical_xor(np.ones_like(mask_1), mask_1).astype(int)
    mask_3 = np.ones_like(mask_1)
    mask_3[ypred * label > 1] = 0
    mask_3 = mask_3.astype(int)

    val_1 = -x.multiply(label.reshape(-1,1))
    val_2 = -val_1.multiply((1-label*ypred).reshape(-1,1))

    # val_1 = np.asarray(val_1.mean(0)).flatten()
    # val_2 = np.asarray(val_2.mean(0)).flatten()

    val_1 = val_1.multiply(mask_1.reshape(-1,1))
    val_2 = val_2.multiply(mask_2.reshape(-1,1))

    val_1 = val_1.multiply(mask_3.reshape(-1,1))
    val_2 = val_2.multiply(mask_3.reshape(-1,1))

    grad = np.asarray((val_1 + val_2).mean(0)).flatten()

    return grad


def inverse_hvp_hinge_newtonCG(x_train,y_train,y_pred,v,hessian_free=True,tol=1e-5, scale_factor=1.0):
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
        hessian_matrix = hessian_hingle_loss_theta(y_train,y_pred,x_train,scale_factor)

    # build functions for newton-cg optimization
    def fmin_loss_fn(x):
        """Objective function for newton-cg.
        H^-1 * v = argmin_t {0.5 * t^T * H * t - v^T * t}
        """
        if hessian_free:
            hessian_vec_val = hessian_vector_product_hinge(y_train,y_pred,x_train,x,scale_factor) # [n,]
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
            hessian_vec_val = hessian_vector_product_hinge(y_train,y_pred,x_train,x,scale_factor)
        else:
            hessian_vec_val = np.dot(x,hessian_matrix) # [n,]

        grad = hessian_vec_val - v

        return grad

    def get_fmin_hvp(x,p):
        # get H * p
        if hessian_free:
            hessian_vec_val = hessian_vector_product_hinge(y_train,y_pred,x_train,p,scale_factor)
        else:
            hessian_vec_val = np.dot(p,hessian_matrix)

        return hessian_vec_val

    def get_cg_callback(verbose):
        def fmin_loss_split(x):
            if hessian_free:
                hessian_vec_val = hessian_vector_product_hinge(y_train,y_pred,x_train,x,scale_factor)
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
                           maxiter=100,)

    print("implicit hessian-vector products mean:",fmin_results.mean())
    print("implicit hessian-vector products norm:",np.linalg.norm(fmin_results))
    print("Inverse HVP took {:.1f} sec".format(time.time() - start_time))
    return fmin_results