import numpy as np
import torch

from nnlib.nnlib import utils
from methods import LangevinDynamics

# Import added
import scipy.optimize as opt

def discrete_mi_est(xs, ys, nx=2, ny=2):
    prob = np.zeros((nx, ny))
    for a, b in zip(xs, ys):
        prob[a,b] += 1.0/len(xs)
    pa = np.sum(prob, axis=1)
    pb = np.sum(prob, axis=0)
    mi = 0
    for a in range(nx):
        for b in range(ny):
            if prob[a,b] < 1e-9:
                continue
            mi += prob[a,b] * np.log(prob[a,b]/(pa[a]*pb[b]))
    return max(0.0, mi)


def estimate_fcmi_bound_classification(masks, preds, num_examples, num_classes,
                                       verbose=False, return_list_of_mis=False):
    bound = 0.0
    list_of_mis = []
    for idx in range(num_examples):
        ms = [p[idx] for p in masks]
        ps = [p[2*idx:2*idx+2] for p in preds]
        for i in range(len(ps)):
            ps[i] = torch.argmax(ps[i], dim=1)
            ps[i] = num_classes * ps[i][0] + ps[i][1]
            ps[i] = ps[i].item()
        cur_mi = discrete_mi_est(ms, ps, nx=2, ny=num_classes**2)
        list_of_mis.append(cur_mi)
        bound += np.sqrt(2 * cur_mi)
        if verbose and idx < 10:
            print("ms:", ms)
            print("ps:", ps)
            print("mi:", cur_mi)
    bound *= 1/num_examples

    if return_list_of_mis:
        return bound, list_of_mis

    return bound
    
def estimate_ecmi_bound_classification(masks, L0, L1, num_examples, train_acc):
    bound, bound0, bound1, sec_moment = 0.0, 0.0, 0.0, 0.0
    I_mean, Channel = 0.0, 0.0
    sharp0, sharp1 = [], []
    L_n = 1-np.mean(train_acc)
    for idx in range(num_examples):
        ms = [p[idx] for p in masks]
        err0 = [p[idx] for p in L0]
        err1 = [p[idx] for p in L1]
        delta = (np.array(err1)-np.array(err0)+1).tolist()
        delmi = discrete_mi_est(ms, delta, nx=2, ny=3)
        posmi = discrete_mi_est(ms, err0, nx=2, ny=2)
        negmi = discrete_mi_est(ms, err1, nx=2, ny=2)
        bound += np.sqrt(2 * delmi)
        bound0 += 2*np.sqrt(2 * posmi)
        bound1 += 2*np.sqrt(2 * negmi)
        # I_mean += negsmi
        I_mean += (negmi+posmi)/2

        Channel += delmi
        # sharp0.append(np.mean(err0[np.where(np.array(ms) == 0)[0]])**2)
        # sharp1.append(np.mean(err1[np.where(np.array(ms) == 1)[0]])**2)
        # sharp0.append(np.mean(err0[ms.index(0)])**2)
        # sharp1.append(np.mean(err1[ms.index(1)])**2)
        pos_trid=[i for i, x in enumerate(ms) if x == 0]
        neg_trid=[i for i, x in enumerate(ms) if x == 1]
        pos_tr=[err0[i] for i in pos_trid]
        neg_tr=[err1[i] for i in neg_trid]
        sharp0.append(np.mean(pos_tr)**2)
        sharp1.append(np.mean(neg_tr)**2)
    print(L0[:5])
    print(L1[:5])
    print(ms[:5])
    print(delta[:5])
    print(len(ms))
    print(len(pos_trid), len(pos_tr))
    print(len(neg_trid), len(neg_tr))
    print(pos_trid[:5])
    print(neg_trid[:5])
    print(err0[:9])
    print(err1[:9])
    print(pos_tr[:5])
    print(neg_tr[:5])
    print(np.mean(pos_tr[:5]))
    print(np.mean(pos_tr[:5]) ** 2)
    print(neg_tr[:5])
    print(np.mean(neg_tr[:5]) ** 2)
    bound *= 1/num_examples
    bound0 *= 1/num_examples
    bound1 *= 1/num_examples
    I_mean *= 1/num_examples
    inpbound = 2*I_mean/np.log(2)
    Channel *= 1/num_examples
    capacity = Channel/np.log(2)
    subbound=4*I_mean+4*np.sqrt(I_mean*L_n)
    sec_moment=(sum(sharp0)+sum(sharp1))/(2*num_examples)

    fun1 = lambda x: x[0]*L_n + I_mean/x[1]
    cons1 = ({'type': 'ineq', 'fun': lambda x:  -np.exp(2*x[1]) - np.exp(-2*x[1]*(x[0]+1)) + 2})
    bnds1 = ((0, None), (0, None))

    res1 = opt.minimize(fun1, (0.5, 0.1), method='SLSQP', bounds=bnds1,
               constraints=cons1, options = {'disp':False})
    optbound = res1.fun
    
    fun2 = lambda x: x[0]*(L_n-(1-x[2]**2)*((1-np.array(train_acc))**2).mean()) + I_mean/x[1]
    cons2 = ({'type': 'ineq', 'fun': lambda x:  -np.exp(2*x[1]) - np.exp(-2*x[1]*(x[0]*(x[2]**2)+1)) + 2})
    bnds2 = ((0, None), (0, None), (0, 1))

    res2 = opt.minimize(fun2, (1, 0.1, 0.5), method='SLSQP', bounds=bnds2,
               constraints=cons2, options = {'disp':False})
    varbound = res2.fun

    fun3 = lambda x: x[0]*(L_n-(1-x[2]**2)*sec_moment) + I_mean/x[1]

    res3 = opt.minimize(fun3, (1, 0.1, 0.5), method='trust-constr', bounds=bnds2,
               constraints=cons2, options = {'disp':False})
    sharpbound = res3.fun

    print('Information', I_mean)
    print('Channel Capacity', capacity)
    print('Loss', L_n, 'forvar', ((1-np.array(train_acc))**2).mean(), 'forsharp', sec_moment)
    print('Opt', optbound, res1.x)
    print('Var', varbound, res2.x)
    print('Sharp', sharpbound, res3.x)

    return bound, bound0, bound1, optbound, subbound, inpbound, varbound, sharpbound, capacity

def kl(q,p):
    if q>0:
        return q*np.log(q/p) + (1-q)*np.log( (1-q)/(1-p) )
    else:
        return np.log( 1/(1-p) )

# Function added
def estimate_interp_bound_classification(masks, preds, num_examples, num_classes, train_acc,
                                       verbose=False, return_list_of_mis=False):
    RHS = 0.0
    list_of_mis = []
    for idx in range(num_examples):
        ms = [p[idx] for p in masks]
        ps = [p[2*idx:2*idx+2] for p in preds]
        for i in range(len(ps)):
            ps[i] = torch.argmax(ps[i], dim=1)
            ps[i] = num_classes * ps[i][0] + ps[i][1]
            ps[i] = ps[i].item()
        cur_mi = discrete_mi_est(ms, ps, nx=2, ny=num_classes**2)
        list_of_mis.append(cur_mi)
        RHS += cur_mi
        if verbose and idx < 10:
            print("ms:", ms)
            print("ps:", ps)
            print("mi:", cur_mi)
    RHS *= 1/num_examples

    Rhat = 1-train_acc
    if Rhat == 0:
        bound = RHS/np.log(2)
    else:
        bound = 1
    if return_list_of_mis:
        return bound, list_of_mis

    return bound

# Function added
def estimate_kl_bound_classification(masks, preds, num_examples, num_classes, train_acc,
                                       verbose=False, return_list_of_mis=False):
    RHS = 0.0
    list_of_mis = []
    for idx in range(num_examples):
        ms = [p[idx] for p in masks]
        ps = [p[2*idx:2*idx+2] for p in preds]
        for i in range(len(ps)):
            ps[i] = torch.argmax(ps[i], dim=1)
            ps[i] = num_classes * ps[i][0] + ps[i][1]
            ps[i] = ps[i].item()
        cur_mi = discrete_mi_est(ms, ps, nx=2, ny=num_classes**2)
        list_of_mis.append(cur_mi)
        RHS += cur_mi
        if verbose and idx < 10:
            print("ms:", ms)
            print("ps:", ps)
            print("mi:", cur_mi)
    RHS *= 1/num_examples

    Rhat = 1-train_acc
    # Constraints are expressions that should be non-negative
    # Below factors guarantee R<=1, R>=0,and bound satisfied
    def con(R):
        return (RHS-kl(Rhat,Rhat/2 + R/2))*R*(1-R)

    # Minimize -R to find biggest R that satisfies constraints
    objective = lambda R: -R
    cons = ({'type': 'ineq', 'fun' : con})
    results = opt.minimize(objective,x0=0.5,
    constraints = cons,
    options = {'disp':False})

    bound = results.x[0]
    if return_list_of_mis:
        return bound, list_of_mis

    return bound

# Function added
def estimate_lg_bound_classification(masks, preds, num_examples, num_classes, train_acc,
                                       verbose=False, return_list_of_mis=False):
    RHS = 0.0
    list_of_mis = []
    for idx in range(num_examples):
        ms = [p[idx] for p in masks]
        ps = [p[2*idx:2*idx+2] for p in preds]
        for i in range(len(ps)):
            ps[i] = torch.argmax(ps[i], dim=1)
            ps[i] = num_classes * ps[i][0] + ps[i][1]
            ps[i] = ps[i].item()
        cur_mi = discrete_mi_est(ms, ps, nx=2, ny=num_classes**2)
        list_of_mis.append(cur_mi)
        RHS += cur_mi
        if verbose and idx < 10:
            print("ms:", ms)
            print("ps:", ps)
            print("mi:", cur_mi)
    RHS *= 1/num_examples

    Rhat = 1-train_acc
    def con(x):
        return (-x[0]*(1-x[1]) - (np.exp(x[0]) - 1 - x[0] ) * ( 1 + x[1]**2 ))
    objective = lambda x: x[1]*Rhat + RHS/x[0]
    cons = ({'type': 'ineq', 'fun' : con})
    bnds = ((0, 0.37),(1,np.inf))
    results = opt.minimize(objective,x0=[3,2],
                           constraints = cons,
                           bounds = bnds,
                           options = {'disp':True})

    bound = results.x[1]*Rhat + RHS/results.x[0]

    if return_list_of_mis:
        return bound, list_of_mis

    return bound





def estimate_sgld_bound(n, batch_size, model):
    """ Computes the bound of Negrea et al. "Information-Theoretic Generalization Bounds for
    SGLD via Data-Dependent Estimates". Eq (6) of https://arxiv.org/pdf/1911.02151.pdf.
    """
    assert isinstance(model, LangevinDynamics)
    assert model.track_grad_variance
    T = len(model._grad_variance_hist)
    assert len(model._lr_hist) == T + 1
    assert len(model._beta_hist) == T + 1
    ret = 0.0
    for t in range(1, T):  # skipping the first iteration as grad_variance was not tracked for it
        ret += model._lr_hist[t] * model._beta_hist[t] / 4.0 * model._grad_variance_hist[t-1]
    ret = np.sqrt(utils.to_numpy(ret))
    ret *= np.sqrt(n / 4.0 / batch_size / (n-1) / (n-1))
    return ret
