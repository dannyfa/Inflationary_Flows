# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Contains methods needed to integrate pfODEs
for toy data. 


This includes:
    1) methods that do a 'GT' pfODE integration - 
this relies on a discrete estimate of the score term, 
which is possible for lower-dimensional data. 
    2) methods that do 'Net-based' pfODE integration - 
these rely on network score estimates to compute flow
updates. This what we used in main paper experiments.

As in rest of code "melt" == "inflation" for methods
that take pfODE direction as part of their arguments.

"""

import torch
import numpy as np
from tqdm import tqdm 
import sys
sys.path.append("..")
import dnnlib

#-----------------------------------------------------------------------------#
# Methods for toy GT/discrete pfODE integration 

def compute_score_num_den(C_inv, X, xj, eps=1e-30):
    """
    
    Computes numerator and denominator for score
    term over entire set of test pts X for a given 
    refence flow pt x_j. 
    
    Args:
    -------------------
    C_inv: torch.tensor - [dim, dim]. Inverse of current
    melting kernel covariance.
    X: torch.tensor - [bs, dim]. Tensor containing test pts
    at current time point/stage.
    xj: torch.tensor - [dim]. Tensor containing reference flow
    field point we are computing wj's w.r.t. This are NOT 
    updated as time evolves.
    eps: float. Small value we use to clip numerator exponential
    term to avoid running into underflow.
    Defaults to 1e-30.
    
    """
    X = X.unsqueeze(-1) #[bs, dim, 1]
    xj = xj.unsqueeze(0).repeat((X.shape[0], 1)).unsqueeze(-1) #[bs, dim, 1]
    X_diff = X-xj
    num_exp_term=torch.einsum('bki, ij-> bkj', \
                              torch.transpose(X_diff, 1, 2), C_inv) #[bs, 1, dim]
    num_exp_term=torch.einsum('bki, bij -> bkj', \
                              num_exp_term, X_diff) #[bs, 1, 1]
    num_exp_term=(-0.5*num_exp_term).squeeze(-1).squeeze(-1) #[bs]
    #set up den and clip terms to some min floor 
    den = torch.exp(num_exp_term)
    den = torch.clip(den, min=eps, max=None)
    num = - torch.einsum('ij, bjk -> bik', C_inv, X_diff).squeeze(-1) * den[:, None]
    return num, den


def compute_unscaled_score(C_inv, X0s, curr_xs):
    """
    
    Uses above method to compute unscaled score estimates.
    
    Args
    ---------------------
    C_inv: torch.tensor - [dim, dim]. Inverse of current
    melting kernel covariance.
    X0s: torch.tensor [bs, dim]. Original (unscaled) 
    set of data pts, to be used as flow field pts (x_j's).
    curr_xs: torch.tensor [bs, dim]. Particles at 
    current time pt. These should also be UNSCALED.
    
    """
    num = torch.zeros(curr_xs.shape).to(X0s.device) #bs, dim
    den = torch.zeros(curr_xs.shape[0]).to(X0s.device) #bs
    for j in range(X0s.shape[0]):
        xj = X0s[j, :] #dim 
        cur_num, cur_den = compute_score_num_den(C_inv, curr_xs, xj)
        num += cur_num
        den += cur_den
    score = torch.div(num, den[:, None])
    return score

def compute_gt_flow(curr_tilde_xs, curr_xs, X0s, g, t, xi_star, A0=1., gamma0=5e-4, rho=1.):
    """
    Computes actual flow (dx/dt) using discrete score estimates.
    This is equivalent to Eqn (171) in Appendix B.3.1, only substituting 
    C^{-1}(t)(D(x, t) - x) for the discrete score estimates and  using
    diagonal forms for A(t), C(t).
    
    Args
    ----
    curr_tilde_xs: torch.Tensor [bs, dim]. Current unscaled 
    evolving pts.
    curr_xs: torch.Tensor [bs, dim]. Current SCALED
    evolving pts.
    X0s: torch.tensor [bs, dim]. Original (unscaled) 
    set of data pts, to be used as flow field pts (x_j's).
    g: torch.Tensor [dim]. Constant g to be used for noise,
    scale schedules.
    t: float. Current time pt being simulated.
    xi_star: float. Highest eigenval for original data.
    A0: float. Final variance to be achieved per dim.
    gamma0: float. Minimal melting kernel covariance.
    rho: float. Exp growth cte for noise schedule.
    """
    #get s, s_dot, gamma_dot 
    s_dot = dnnlib.util.get_s_dot(xi_star, g, t, A0=A0, gamma0=gamma0, rho=rho)
    s = dnnlib.util.get_s(xi_star, g, t, A0=A0, gamma0=gamma0, rho=rho)
    gamma_dot = dnnlib.util.get_gamma_dot(g, t, gamma0=gamma0, rho=rho)
    
    #get unscaled score term 
    gamma = dnnlib.util.get_gamma(g, t, gamma0=gamma0, rho=rho)
    C_inv = torch.diag(torch.reciprocal(gamma))
    unscaled_score = compute_unscaled_score(C_inv, X0s, curr_tilde_xs)
    
    #now compute dx/dt 
    dx_dt = (s_dot/s)*curr_xs #bs, dim
    dx_dt -= 0.5*s*gamma_dot[None, :]*unscaled_score #bs, dim
    
    return dx_dt, unscaled_score


def sim_batch_discrete_ODE(batch, x0, g, data_eigs, W, int_mode='melt', n_iters=1501, h=1e-2, \
                                          gamma0=5e-4, rho=1., A0=1., save_freq=10):
    """
    Args:
    -------------
    
    batch: torch.Tensor [bs, dim]
    tensor containing batch of data to integrate pfODE over.
    
    x0: torch.Tensor[bs, dim]
    tensor containing reference data pts we use to define
    scores for pfODE integration.
    
    g: torch.Tensor [dim]. Constant g tensor to be used 
    for noise and scaling schedules.
    
    data_eigs: np.array [dim]. Eigenvals for original
    dset.
    
    W: torch.Tensor [dim, dim]. Matrix whose cols
    are eigenvectors for original data. 
    
    int_mode: str. ('melt' or 'gen'). Whether to 
    simulate ODE fwd ('melt') or backwards ('gen'). 
    "melt" == "inflation" here.

    n_iters: num of iterations in ODE int/melting.
    
    h: step size
    
    gamma0: scalar. Scaling factor for starting melting kernel diag. cov.
    
    rho: scalar. Constant factor used for exp. growth of C(t).
    
    A0: scalar. Scaling factor for ending diag. cov. we will achieve at steady state.
    
    save_freq: frequency at which to save melting results.
    
    
    Returns
    -------
    Dictionary containing updates (saved at specified save_freq)
    for following variables: 
        1) tilde_xs: unscaled batch being simulated (in image space)
        2) tilde_xs_es: unscaled batch being simulated (in eigen space)
        3) xs: scaled batch being simulated (in image space)
        4) xs_es: scaled batch being simulated (in eigen space)
        5) dxs: flow updates (in image space) for scaled variable.
        6) dxs_es: flow updates (in eigen space) for scaled variable.
        7) unscaled_scores: discrete (unscaled) scores
        8) gt_netout: GT/discrete equivalent of network output D(x, t). Computed
        using relationship between network outputs and scores presented in paper.
      
    """
    
    #init dict to save results
    res_dict = {'tilde_xs':[], 'xs':[], 'tilde_xs_es':[], 'xs_es':[], 
               'dxs':[], 'dxs_es':[], 'unscaled_scores':[], 'gt_netout':[]}
    
    #get our integration time pts 
    taus = dnnlib.util.get_disc_times('ifs', end_time=n_iters*h, \
                                      n_iters=n_iters, int_mode=int_mode)
    
    #get xi_star
    xi_star = np.amax(data_eigs)
    
    #get our X0s to ES (this is needed regardless of int_mode)
    x0 = torch.einsum('ij, bjk -> bik', W.T, x0.unsqueeze(-1)).squeeze(-1)
    
    #get s at t0, this is used to initialize our xs variables 
    init_s = dnnlib.util.get_s(xi_star, g, taus[0], A0=A0, gamma0=gamma0, rho=rho)

    #init tilde_xs_es, xs_es, tilde_xs, xs
    if int_mode=='melt':
        batch = torch.einsum('ij, bjk -> bik', W.T, \
                             batch.type(torch.float32).unsqueeze(-1)).squeeze(-1)
        curr_tilde_xs_es = batch
        curr_xs_es = init_s * curr_tilde_xs_es    
    else:
        curr_xs_es = batch 
        curr_tilde_xs_es = (1/init_s) * curr_xs_es
    
    curr_tilde_xs = torch.einsum('ij, bjk -> bik', W, curr_tilde_xs_es.unsqueeze(-1)).squeeze(-1)
    curr_xs = torch.einsum('ij, bjk -> bik', W, curr_xs_es.unsqueeze(-1)).squeeze(-1)
    
    #log these initial vals... 
    res_dict['tilde_xs'].append(curr_tilde_xs.cpu().numpy())
    res_dict['xs'].append(curr_xs.cpu().numpy())
    res_dict['tilde_xs_es'].append(curr_tilde_xs_es.cpu().numpy())
    res_dict['xs_es'].append(curr_xs_es.cpu().numpy())    
    
    
    #now integrate pfODE ... 
    for itr in tqdm(range(0, n_iters-1), total=n_iters-1, desc='GT Batch ODE Sim Progression'):
       
        #get dx_dt 
        curr_dxs_dt_es, curr_unscaled_score_es = compute_gt_flow(curr_tilde_xs_es, curr_xs_es, x0, g, \
                                                        taus[itr], xi_star, A0=A0, gamma0=gamma0, rho=rho)   
        #get net output GT equivalent 
        curr_gamma = dnnlib.util.get_gamma(g, taus[itr], gamma0=gamma0, rho=rho) #dim 
        curr_gt_netout_es = curr_gamma[None, :]*curr_unscaled_score_es 
        curr_gt_netout_es += curr_tilde_xs_es

        #take Euler step
        curr_dxs_es = curr_dxs_dt_es * (taus[itr+1]- taus[itr])
        curr_xs_es += curr_dxs_es
        
        #update unscaled variable as well 
        next_s = dnnlib.util.get_s(xi_star, g, taus[itr+1], A0=A0, gamma0=gamma0, rho=rho)
        curr_tilde_xs_es = (1/next_s)*curr_xs_es
        
        #save results after first update, after multiples of save_freq,
        #and at last update.
        
        if (itr%save_freq == 0) or (itr == (n_iters - 2)):
           #get IS versions of variables we wish to log 
           curr_tilde_xs = torch.einsum('ij, bjk -> bik', W, curr_tilde_xs_es.unsqueeze(-1)).squeeze(-1)
           curr_xs = torch.einsum('ij, bjk -> bik', W, curr_xs_es.unsqueeze(-1)).squeeze(-1)
           curr_dxs = torch.einsum('ij, bjk -> bik', W, curr_dxs_es.unsqueeze(-1)).squeeze(-1)
           
           #log variables 
           res_dict['tilde_xs'].append(curr_tilde_xs.cpu().numpy())
           res_dict['xs'].append(curr_xs.cpu().numpy())
           res_dict['tilde_xs_es'].append(curr_tilde_xs_es.cpu().numpy())
           res_dict['xs_es'].append(curr_xs_es.cpu().numpy())
           res_dict['unscaled_scores'].append(curr_unscaled_score_es.cpu().numpy())
           res_dict['gt_netout'].append(curr_gt_netout_es.cpu().numpy())
           res_dict['dxs_es'].append(curr_dxs_es.cpu().numpy())
           res_dict['dxs'].append(curr_dxs.cpu().numpy())
           
    return res_dict
           


#--------------------------------------------------------------------------------
# Methods for toy Net-based pfODE integration  

def compute_net_flow(D_tilde_xs, space, W, xs, s, s_dot, gamma_inv, gamma_dot):
    """
    
    Computes flow Equation in  eigen-space (ES)
    or image space (IS).
    
    Args 
    ---------------
    
    D_tilde_xs: torch.Tensor [bd, dim]. De-noised network outputs
    in same basis/space we want to compute flow in. 
    
    space: str. Desired space/basis we should compute flow in. Should match
    net.space attribute.
    
    W: torch.Tensor [dim, dim]. Tensor containing target data eigenvectors 
    as its columns.
    
    xs: torch.Tensor [bs, dim]. Scaled xs at current time pt in 
    desired basis/space. 
    
    s_dot: temporal derivative of scaling.
    
    gamma_inv: diagonal of inverse noise covariance matrix at current time pt.
    
    gamma_dot: diagonal of temporal derivative of noise covariance mat for current 
    time pt.
    
    """
    
    if space == 'ES':
        dx_dt = ((s_dot/s) + 0.5*gamma_dot*gamma_inv)[None, :] * xs #bs, dim 
        dx_dt -= ((0.5*s*gamma_dot*gamma_inv)[None, :]*D_tilde_xs) #bs, dim 
    else:
        dx_dt_first_term = torch.einsum('ij, bjk -> bik', W.T, xs.unsqueeze(-1)).squeeze(-1)
        dx_dt_first_term = ((s_dot/s) + 0.5*gamma_dot*gamma_inv)[None, :] * dx_dt_first_term
        dx_dt_first_term = torch.einsum('ij, bjk -> bik', W, dx_dt_first_term.unsqueeze(-1)).squeeze(-1)
        
        dx_dt_second_term = torch.einsum('ij, bjk -> bik', W.T, D_tilde_xs.unsqueeze(-1)).squeeze(-1)
        dx_dt_second_term = (0.5*s*gamma_dot*gamma_inv)[None, :] * dx_dt_second_term
        dx_dt_second_term = torch.einsum('ij, bjk -> bik', W, dx_dt_second_term.unsqueeze(-1)).squeeze(-1)
        
        dx_dt = dx_dt_first_term - dx_dt_second_term 
        
    return dx_dt

def get_netout_unscld_scores(net, curr_tilde_xs_es, ts):
    
    """
    Gets network output (D(x, t)) for a given set of unscaled, noised
    inputs in ES and a given time pt. 
    Additionally, computes equivalent unscaled scores using network
    outputs --> unscld_scores = C^{-1}(t)(D(x, t) - x). 
    
    Args
    -------------------
    net: trained instance of IfsToyPreCond class. 
    curr_tilde_xs_es: torch.Tensor [bs, dim]. Contains batch
    of noised inputs in ES to be passed as inputs to net.
    These should be in ES regardless of space network was trained on.
    ts: float. Time pt we wish to query network at. This will
    be properly shaped and passed to net as noise conditioning input.
    
    """
    net_ts = (torch.ones(curr_tilde_xs_es.shape[0]) * ts).to(curr_tilde_xs_es.device)
    with torch.no_grad(): 
        D_tilde_xs, _ = net(curr_tilde_xs_es, net_ts) #this will be in net.space basis
    
    #compute equivalent unscld scores
    curr_gamma = dnnlib.util.get_gamma(net.g, ts, gamma0=net.gamma0, rho=net.rho) #dim
    curr_gamma_inv = torch.reciprocal(curr_gamma) #dim 
        
    if net.space=='ES': 
        #can apply formula directly
        unscaled_score = curr_gamma_inv[None, :]*(D_tilde_xs - curr_tilde_xs_es)
        
    else:
        #need change of basis prior to applying gamma_inv
        curr_tilde_xs = torch.einsum('ij, bjk -> bik', net.W, curr_tilde_xs_es.unsqueeze(-1)).squeeze(-1)
        unscaled_score = torch.einsum('ij, bjk -> bik', net.W.T, (D_tilde_xs - curr_tilde_xs).unsqueeze(-1)).squeeze(-1)
        unscaled_score *= curr_gamma_inv[None, :]
        unscaled_score = torch.einsum('ij, bjk -> bik', net.W, unscaled_score.unsqueeze(-1)).squeeze(-1)
    
    return D_tilde_xs, unscaled_score


def take_Euler_step(D_tilde_xs, xs_es, space, W, s, s_dot, gamma_inv, gamma_dot, t, t_plus_one):
    """
    
    CTakes Euler update in appropriate space/basis we are simulating 
    out flows in.
    
    Args
    -----
    D_tilde_xs: torch.Tensor [bs, dim]. Tensor with de-noised network outputs
    (in same space as net.space).
    xs_es: torch.Tensor [bs, dim]. Tensor with current scaled variables 
    in eigen space (ES). This might be appropriately passed to IS prior to updating variable (if needed).
    space: 'str'. Space/basis network was trained on. This matches space for D_tilde_xs.
    W: torch.Tensor [dim, dim]. Tensor containing target data eigenvectors as its columns.
    s: float. Scale at time (t).
    s_dot: float. Time derivative of s(t)
    gamma_inv: torch.Tensor [dim]. Diagonal of C^{-1}(t).
    gamma_dot: torch.Tensor [dim]. Temporal derivative of gamma == diag(C(t)).
    t: float. Current time pt
    t_plus_one: float. Next time pt.
    """
    if space == 'ES':
        dxs_dt_es = compute_net_flow(D_tilde_xs, space, W, xs_es, s, \
                                       s_dot, gamma_inv, gamma_dot) #flow in ES
        dxs_es = dxs_dt_es * (t_plus_one - t)
        xs_es += dxs_es
    
    else:
        xs = torch.einsum('ij, bjk -> bik', W, xs_es.unsqueeze(-1)).squeeze(-1)
        dxs_dt = compute_net_flow(D_tilde_xs, space, W, xs, s, \
                                       s_dot, gamma_inv, gamma_dot) #flow in IS 
        dxs = dxs_dt * (t_plus_one - t)
        xs += dxs 
        xs_es = torch.einsum('ij, bjk -> bik', W.T, xs.unsqueeze(-1)).squeeze(-1)
        dxs_es = torch.einsum('ij, bjk -> bik', W.T, dxs.unsqueeze(-1)).squeeze(-1)
    
    return xs_es, dxs_es
        
        
    
def sim_batch_net_ODE(batch, net, int_mode='melt', n_iters=1501, h=1e-2, A0=1., save_freq=10):
    """
    Integrates pfODE for a batch of toy data using network D(x, t) estimates. 
    
    
    Args:
    ------------
    batch: torch.Tensor [bs, dim]
    tensor containing batch of data we will be integrating pfODE over.
    
    net: trained instance of IfsToyPreCond class.

    int_mode: str. Integration mode - whether we want to simulate ODE fwd in time
    (melt/inflation) or backwards (generation).

    n_iters: num of iterations in ODE int/melting.
    
    h: step size
        
    A0: scalar. Scaling factor.
    
    save_freq: frequency at which to save melting results.
    
    Returns
    -------
    Dictionary containing updates (saved at specified save_freq)
    for following variables: 
        1) tilde_xs: unscaled batch being simulated (in image space)
        2) tilde_xs_es: unscaled batch being simulated (in eigen space)
        3) xs: scaled batch being simulated (in image space)
        4) xs_es: scaled batch being simulated (in eigen space)
        5) dxs: flow updates (in image space) for scaled variable.
        6) dxs_es: flow updates (in eigen space) for scaled variable.
        7) unscaled_scores: discrete (unscaled) scores
        8) net_outs: network outputs D(x, t) 

    """
    
    #setup results dict 
    res_dict = {'tilde_xs':[], 'xs':[], 'tilde_xs_es':[], 'xs_es':[], 
               'dxs':[], 'dxs_es':[], 'net_outs':[], 'unscaled_scores':[]}
    
    
    #load attributes we need from net
    g = torch.from_numpy(net.g.cpu().numpy()).type(torch.float32).to(batch.device)
    data_eigs = torch.from_numpy(net.data_eigs).to(batch.device)
    xi_star = np.amax(data_eigs.cpu().numpy()) 
    W = torch.from_numpy(net.W.cpu().numpy()).type(torch.float32).to(batch.device)
    gamma0 = net.gamma0
    rho=net.rho
    
    #make sure net attributes are on same device
    net.device=batch.device
    net.g = g
    net.W = W
    
    #pass these to net_kwargs dict
    net_kwargs = {'g':g, 'W':W, 'rho':rho, 'gamma0':gamma0, 'A0':A0, 'xi_star':xi_star}

    #get our integration time pts 
    taus = dnnlib.util.get_disc_times('ifs', end_time=n_iters*h, \
                                      n_iters=n_iters, int_mode=int_mode)
    
    #get s at t0, this is used to initialize our xs variables 
    init_s = dnnlib.util.get_s(xi_star, g, taus[0], A0=A0, gamma0=gamma0, rho=rho)

    #init tilde_xs_es, xs_es, tilde_xs, xs
    if int_mode=='melt':
        batch = torch.einsum('ij, bjk -> bik', W.T, \
                             batch.type(torch.float32).unsqueeze(-1)).squeeze(-1)
        curr_tilde_xs_es = batch
        curr_xs_es = init_s * curr_tilde_xs_es    
    else:
        curr_xs_es = batch 
        curr_tilde_xs_es = (1/init_s) * curr_xs_es
    
    curr_tilde_xs = torch.einsum('ij, bjk -> bik', W, curr_tilde_xs_es.unsqueeze(-1)).squeeze(-1)
    curr_xs = torch.einsum('ij, bjk -> bik', W, curr_xs_es.unsqueeze(-1)).squeeze(-1)
    
    #log these initial vals... 
    res_dict['tilde_xs'].append(curr_tilde_xs.cpu().numpy())
    res_dict['xs'].append(curr_xs.cpu().numpy())
    res_dict['tilde_xs_es'].append(curr_tilde_xs_es.cpu().numpy())
    res_dict['xs_es'].append(curr_xs_es.cpu().numpy()) 
    
    #now integrate pfODE ... 
    for itr in tqdm(range(0, n_iters-1), total=n_iters-1, desc='GT Batch ODE Sim Progression'):
       
        #get netoutputs, unscaled scores
        curr_D_tilde_xs, curr_unscaled_score = get_netout_unscld_scores(net, curr_tilde_xs_es, taus[itr])
        curr_s, curr_s_dot, curr_gamma_inv, curr_gamma_dot = dnnlib.util.get_dx_dt_params(taus[itr], **net_kwargs)
        
        #now take Euler step
        curr_xs_es, curr_dxs_es = take_Euler_step(curr_D_tilde_xs, curr_xs_es, net.space, \
                                       net.W, curr_s, curr_s_dot, curr_gamma_inv, \
                                           curr_gamma_dot, taus[itr], taus[itr+1])

        
        #update unscaled variable as well 
        next_s = dnnlib.util.get_s(xi_star, g, taus[itr+1], A0=A0, gamma0=gamma0, rho=rho)
        curr_tilde_xs_es = (1/next_s)*curr_xs_es
        
        #save results after first update, after multiples of save_freq,
        #and at last update.
        if (itr%save_freq == 0) or (itr == (n_iters - 2)):
           #get IS versions of variables we wish to log 
           curr_tilde_xs = torch.einsum('ij, bjk -> bik', W, curr_tilde_xs_es.unsqueeze(-1)).squeeze(-1)
           curr_xs = torch.einsum('ij, bjk -> bik', W, curr_xs_es.unsqueeze(-1)).squeeze(-1)
           curr_dxs = torch.einsum('ij, bjk -> bik', W, curr_dxs_es.unsqueeze(-1)).squeeze(-1)
           
           #log variables 
           res_dict['tilde_xs'].append(curr_tilde_xs.cpu().numpy())
           res_dict['xs'].append(curr_xs.cpu().numpy())
           res_dict['tilde_xs_es'].append(curr_tilde_xs_es.cpu().numpy())
           res_dict['xs_es'].append(curr_xs_es.cpu().numpy())
           res_dict['unscaled_scores'].append(curr_unscaled_score.cpu().numpy()) #in net.space
           res_dict['net_outs'].append(curr_D_tilde_xs.cpu().numpy()) #in net.space
           res_dict['dxs_es'].append(curr_dxs_es.cpu().numpy())
           res_dict['dxs'].append(curr_dxs.cpu().numpy())
           
    return res_dict    
