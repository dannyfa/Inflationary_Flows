#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Contains methods needed to run High-Dimensional Melt, 
Rdtrp and Gen simulations.

As in rest of code "melt" == "inflation" for methods
that take pfODE direction as part of their arguments.

"""
import numpy as np
import torch
from tqdm import tqdm 
import sys
sys.path.append("..")
import dnnlib

#-------------------------------------------------------------#
# Methods to run IFs Net-based melt/inflation and gen 
#-------------------------------------------------------------#


#Method to compute net outputs (D_x) using IFs nets 
def compute_ifs_netout(net, tilde_xs, shape, t):
    """
    Uses trained IFs network to compute a de-noised output D_x
    for a given noisy input (tilde_xs).
    This de-noise output is then used to compute our proposed
    flow.
    """
    #pass inputs to ES -- all IFs nets expect inputs to be in ES 
    tilde_xs_es = torch.einsum('ij, bjk -> bik', net.W.T, tilde_xs.unsqueeze(-1)).squeeze(-1)
    
    ts = (torch.ones(tilde_xs_es.shape[0])*t).to(tilde_xs_es.device)
    
    with torch.no_grad():
        #note that output here will already be in appropriate space - ES or IS
        #depending on net.space
        #it will also already be flattened up - [bs, dim]
        D_tilde_xs, _ = net(tilde_xs_es.reshape(shape), ts)
    return D_tilde_xs


# Method to compute flow using Ifs net outputs (D_x)
def compute_net_flow(D_tilde_xs, space, W, xs, s, s_dot, gamma_inv, gamma_dot):
    """
    Computes flow Equation in image space (IS) or eigenspace (ES)
    
    D_tilde_xs: torch.Tensor [bs, dim]. Net output. Should be in desired/correct 
    space flow is being computed in.
    
    space: str. Desired space we should compute flow in. ('ES' or 'IS')
    
    W: torch.Tensor [dim, dim]. Tensor containing data eigenvectors 
    as its columns.
    
    xs: torch.Tensor [bs, dim]. Scaled xs at current time pt in 
    desired space. 
    
    s: float. Scaling at current time pt.
    
    s_dot: float. Derivative of scaling at current time pt.
    
    gamma_inv: torch.tensor [dim]. Diagonal of inverse kernel covariance 
    matrix for current time pt.
    
    gamma_dot: torch..tensor[dim]. Diagonal of derivative of kernel 
    covariance mat for current time pt.
    
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

#method to run Euler step update
def do_Euler_update(get_netout, net, xs_vars_dict, t, t_plus_one, shape, net_kwargs):
    """
    
    Does an Euler update/step. 
    
    Args
    ----
    get_netout: method to calc network denoised output. Takes network and its xs, and t
    inputs as arguments. Xs inputs should be in images space, regardless of net.space. 
    t is jsut a float. 
    net: trained instance of IFsPreCond. 
    xs_vars_dict: dictionary containing our cureent 'xs' variables, namely: 
        1) 'xs' : scaled, in IS
        2) 'xs_es': scaled, in ES
        3) 'tilde_xs':unscaled, in IS
        4) 'tilde_xs_es':unscaled, in ES 
    t: float. Current time pt.
    t_plus_one: float. Next time pt.
    shape: list (B, C, H, W). List with original image shapes.
    net_kwargs: dictionary with additional arguments used to construct s, s_dot,
    gamma_inv, and gamma_dot needed for flow calculation.
    
    Returns
    --------
    Dictionary containing Euler updated/next 'xs' variables (see above) along with:
        1) 'net_outputs': D(x, t) network outputs for current time pt.
        2) 'unscaled scores': C^{-1}(t) (D(x, t) - x), for current time pt.
        3) 'dxs': flow updates for current time pt (in IS)
        4) 'dxs_es': flow updates for current time (in ES).
        5) 'dx_dt': flow update without time step applied (in IS).
        6) 'dx_dt_es': flow update without time step applied (in ES).
    
    """
    #get net output 
    D_tilde_xs = get_netout(net, xs_vars_dict['curr_tilde_xs'], shape, t)
    #get orther args for flow calc
    s, s_dot, gamma_inv, gamma_dot = dnnlib.util.get_dx_dt_params(t, **net_kwargs)
    if net.space == 'ES':
        #get unscaled scores 
        unscaled_score = gamma_inv[None, :]*(D_tilde_xs - xs_vars_dict['curr_tilde_xs_es']) 
        
        curr_dxs_dt_es = compute_net_flow(D_tilde_xs, net.space, net_kwargs['W'], xs_vars_dict['curr_xs_es'], \
                                          s, s_dot, gamma_inv, gamma_dot)
            
        curr_dxs_dt = torch.einsum('ij, bjk -> bik', net_kwargs['W'], curr_dxs_dt_es.unsqueeze(-1)).squeeze(-1)   
        
        dxs_es = curr_dxs_dt_es * (t_plus_one-t)
        dxs = torch.einsum('ij, bjk -> bik', net_kwargs['W'], dxs_es.unsqueeze(-1)).squeeze(-1)   
        
        #get curr_xs_es, curr_xs
        next_xs_es = xs_vars_dict['curr_xs_es'] + dxs_es
        next_xs = torch.einsum('ij, bjk -> bik', net_kwargs['W'], next_xs_es.unsqueeze(-1)).squeeze(-1)

    else: 
        unscaled_score = torch.einsum('ij, bjk -> bik', net_kwargs['W'].T, \
                                      (D_tilde_xs - xs_vars_dict['curr_tilde_xs']).unsqueeze(-1)).squeeze(-1)
        unscaled_score *= gamma_inv[None, :]
        unscaled_score = torch.einsum('ij, bjk -> bik', net_kwargs['W'], unscaled_score.unsqueeze(-1)).squeeze(-1)
            
        #get dx, dx_es
        curr_dxs_dt = compute_net_flow(D_tilde_xs, net.space, net_kwargs['W'], xs_vars_dict['curr_xs'], \
                                       s, s_dot, gamma_inv, gamma_dot)
        curr_dxs_dt_es = torch.einsum('ij, bjk -> bik', net_kwargs['W'].T, curr_dxs_dt.unsqueeze(-1)).squeeze(-1) 
            
        dxs = curr_dxs_dt * (t_plus_one-t) 
        dxs_es = torch.einsum('ij, bjk -> bik', net_kwargs['W'].T, dxs.unsqueeze(-1)).squeeze(-1)     
            
        #get curr_xs_es, curr_xs
        next_xs = xs_vars_dict['curr_xs'] + dxs
        next_xs_es = torch.einsum('ij, bjk -> bik', net_kwargs['W'].T, next_xs.unsqueeze(-1)).squeeze(-1) 
    
    #now get updates unscaled xs variables
    next_s = dnnlib.util.get_s(net_kwargs['xi_star'], net_kwargs['g'], t_plus_one, \
                               A0=net_kwargs['A0'], gamma0=net_kwargs['gamma0'], rho=net_kwargs['rho'])
    #get euler update versions of tilde_xs_es, tilde_xs 
    next_tilde_xs_es = (1/next_s)*next_xs_es
    next_tilde_xs = torch.einsum('ij, bjk -> bik', net_kwargs['W'], next_tilde_xs_es.unsqueeze(-1)).squeeze(-1)
    
    return {'next_xs':next_xs, 'next_xs_es':next_xs_es, 'next_tilde_xs':next_tilde_xs, 
            'next_tilde_xs_es':next_tilde_xs_es, 'unscaled_scores':unscaled_score, 
            'net_outs':D_tilde_xs, 'dxs':dxs, 'dxs_es':dxs_es, 
            'dx_dt':curr_dxs_dt, 'dx_dt_es':curr_dxs_dt_es}

#method to run Heun update/step
def do_Heun_update(get_netout, net, euler_results_dict, xs_vars_dict, t, t_plus_one, shape, net_kwargs):
    """
   
    Does a Heun udpate/step 
    
    Args
    -------
    get_netout: method to calc network denoised output. Takes network and its xs, and t
    inputs as arguments. Xs inputs should be in images space, regardless of net.space. 
    t is jsut a float. 
    net: trained instance of IFsPreCond. 
    euler_results_dict: dictionary containing outputs of Euler step for time "t".
    xs_vars_dict: dictionary containing our cureent 'xs' variables, at time "t" (Same as for Euler's input')
    t: float. Current time pt.
    t_plus_one: float. Next time pt.
    shape: list (B, C, H, W). List with original image shapes.
    net_kwargs: dictionary with additional arguments used to construct s, s_dot,
    gamma_inv, and gamma_dot needed for flow calculation.    
    
    Returns
    --------
    Dictionary containing Euler updated 'xs' variables (see above) along with:
        1) 'net_outputs': D(x, t) network outputs for current time pt.
        2) 'unscaled scores': C^{-1}(t) (D(x, t) - x), for current time pt.
        3) 'dxs': flow updates for current time pt (in IS)
        4) 'dxs_es': flow updates for current time (in ES).    
    
    """
    #get net output 
    D_tilde_xs = get_netout(net, euler_results_dict['next_tilde_xs'], shape, t_plus_one)
        
    #get ODE params for next_t 
    s, s_dot, gamma_inv, gamma_dot = dnnlib.util.get_dx_dt_params(t_plus_one, **net_kwargs)    
    
    if net.space=='ES':
        #get unscaled score 
        unscaled_score = gamma_inv[None, :]*(D_tilde_xs - euler_results_dict['next_tilde_xs_es'])
            
        #get dx, dx_es
        next_dxs_dt_es = compute_net_flow(D_tilde_xs, net.space, net_kwargs['W'], \
                                          euler_results_dict['next_xs_es'], \
                                              s, s_dot, gamma_inv, gamma_dot)
        dxs_es = (0.5*next_dxs_dt_es + 0.5*euler_results_dict['dx_dt_es']) * (t_plus_one-t)
        dxs = torch.einsum('ij, bjk -> bik', net_kwargs['W'], dxs_es.unsqueeze(-1)).squeeze(-1)
            
        #get curr_xs_es, curr_xs
        next_xs_es = xs_vars_dict['curr_xs_es'] + dxs_es
        next_xs = torch.einsum('ij, bjk -> bik', net_kwargs['W'], next_xs_es.unsqueeze(-1)).squeeze(-1)
            
        
    else: 
        #get unscaled score 
        unscaled_score = torch.einsum('ij, bjk -> bik', net_kwargs['W'].T, \
                                      (D_tilde_xs - euler_results_dict['next_tilde_xs']).unsqueeze(-1)).squeeze(-1)
        unscaled_score *= gamma_inv[None, :]
        unscaled_score = torch.einsum('ij, bjk -> bik', net_kwargs['W'], unscaled_score.unsqueeze(-1)).squeeze(-1)
            
        #get dx, dx_es
        next_dxs_dt = compute_net_flow(D_tilde_xs, net.space, net_kwargs['W'], euler_results_dict['next_xs'], \
                                       s, s_dot, gamma_inv, gamma_dot)
        dxs =(0.5*euler_results_dict['dx_dt'] + 0.5*next_dxs_dt) * (t_plus_one-t) 
        dxs_es = torch.einsum('ij, bjk -> bik', net_kwargs['W'].T, dxs.unsqueeze(-1)).squeeze(-1)     
            
        #get curr_xs_es, curr_xs
        next_xs = xs_vars_dict['curr_xs'] + dxs
        next_xs_es = torch.einsum('ij, bjk -> bik', net_kwargs['W'].T, next_xs.unsqueeze(-1)).squeeze(-1)  
    
    #now get tilde_xs_es, tilde_xs corresponding to this heun update
    #note that scale here is STILL for t_plus_one time pt! We just computed update to t_plus_one differently
    next_tilde_xs_es = (1/s)*next_xs_es
    next_tilde_xs = torch.einsum('ij, bjk -> bik', net_kwargs['W'], next_tilde_xs_es.unsqueeze(-1)).squeeze(-1)  
    
    return {'next_xs':next_xs, 'next_xs_es':next_xs_es, 'next_tilde_xs':next_tilde_xs, 
            'next_tilde_xs_es':next_tilde_xs_es, 'unscaled_scores':unscaled_score, 
            'net_outs':D_tilde_xs, 'dxs':dxs, 'dxs_es':dxs_es}


#method to run through one single ODE integration step 

def do_ODE_step(get_netout, net, curr_tilde_xs, curr_tilde_xs_es, curr_xs, curr_xs_es, t, t_plus_one, \
                shape, net_kwargs, solver='heun'):
    
    
    
    """
    
    Runs ODE update using either Euler or Heun solver choices. 
    
    Note that we still return D_x and unscaled_scores here --> these correspond to second step 
    if using heun solver.
    
    Returns: dict. containing updated xs_es, xs, tilde_xs_es, tilde_xs, net_ouputs, 
    unscaled_scores, dxs, and dxs_es .
    
    """
    #construct xs variables dict 
    #this has our current xs variables state 
    curr_xs_vars_dict = {'curr_tilde_xs':curr_tilde_xs, 'curr_tilde_xs_es':curr_tilde_xs_es, 
                    'curr_xs':curr_xs, 'curr_xs_es':curr_xs_es}
    
    
    #take euler step 
    curr_results_dict = do_Euler_update(get_netout, net, curr_xs_vars_dict, t, t_plus_one, shape, net_kwargs)
    
    #if using heuns method, calc heun update
    if solver == 'heun':
        curr_results_dict = do_Heun_update(get_netout, net, curr_results_dict, \
                                           curr_xs_vars_dict, t, t_plus_one, shape, net_kwargs)
    
    #return desired vals
    return {'xs_es':curr_results_dict['next_xs_es'], 'xs':curr_results_dict['next_xs'], 
            'tilde_xs_es':curr_results_dict['next_tilde_xs_es'], 'tilde_xs':curr_results_dict['next_tilde_xs'], 
            'dxs':curr_results_dict['dxs'], 'dxs_es':curr_results_dict['dxs_es'], 
            'net_outputs':curr_results_dict['net_outs'], 'unscaled_scores':curr_results_dict['unscaled_scores']}

        
    
#does ODE simulation using above helpers 
def do_ODE_sim(batch, net, shape, g, W, xi_star, get_netout, int_mode, n_iters, end_time, A0, gamma0, rho, \
               save_freq, disc='vp_ode', solver='heun', eps=1e-2):
    
    """
    Method to simulate our IFs pfODEs.
    
    Args
    ----
    batch: torch.Tensor [bs, dim]. Batch of data we are simulating ODE for.
    net: instance of IFsPreCond class (trained).
    shape: np.array or list. Shapes (B, C, H, W) for our inputs.
    g: torch.Tensor [dim]. Tensor for exponential noise schedule.
    W: torch.Tensor [dim, dim]. Tensor containing eigenvectors
    of original data as its columns.
    xi_star: float. Top eigenvalue for original data.
    get_netout: function to compute net outputs D_x. 
    int_mode: str. Indicates whether we should run ODE fwd ("melt" == "inflation")
    or backwards ("gen"). 
    n_iters:int. Num of iterations/steps to take.
    end_time: float. End melt/start gen time for ODE sim.
    A0: float. Equilibrium variance we wish to reach.
    gamma0: float. Min melting kernel variance.
    rho: float. Cte for exponential noise growth.
    save_freq:int. How often to save results from ODE sim.
    
    disc: str. Option for discretization schedule. 
    solver: str. Option for solver 
    eps: option for epsilo_s if using VP_ODE disc.
    kwargs: dict containing args for pre-trained net D_x
    methods. 

    Returns
    -------
    Dictionary containing results from ODE simulation.
    """
   
    #set up results dict 
    res_dict = {'tilde_xs':[], 'xs':[], 'tilde_xs_es':[], 'xs_es':[], 
               'dxs':[], 'dxs_es':[], 'net_outs':[], 
               'unscaled_scores':[], 'ts':[]}

    #get our time pts 
    taus = dnnlib.util.get_disc_times(disc=disc, end_time=end_time, n_iters=n_iters, int_mode=int_mode, eps=eps)
    
    #get s at t0, this is used to initialize our xs variables 
    init_s = dnnlib.util.get_s(xi_star, g, taus[0], A0=A0, gamma0=gamma0, rho=rho)

    #init tilde_xs_es, xs_es, tilde_xs, xs
    if int_mode=='melt':
        batch = torch.einsum('ij, bjk -> bik', W.T, batch.type(torch.float32).unsqueeze(-1)).squeeze(-1)
        curr_tilde_xs_es = batch
        curr_xs_es = init_s * curr_tilde_xs_es    
    else:
        curr_xs_es = batch 
        curr_tilde_xs_es = (1/init_s) * curr_xs_es
    
    curr_tilde_xs = torch.einsum('ij, bjk -> bik', W, curr_tilde_xs_es.unsqueeze(-1)).squeeze(-1)
    curr_xs = torch.einsum('ij, bjk -> bik', W, curr_xs_es.unsqueeze(-1)).squeeze(-1)
    
    #save these initial vals... 
    res_dict['tilde_xs'].append(curr_tilde_xs.cpu().numpy())
    res_dict['xs'].append(curr_xs.cpu().numpy())
    res_dict['tilde_xs_es'].append(curr_tilde_xs_es.cpu().numpy())
    res_dict['xs_es'].append(curr_xs_es.cpu().numpy())    
    res_dict['ts'].append(taus[0])   
    
    #get neural net kwargs...
    net_kwargs = {'g':g, 'W':W, 'rho':rho, 'gamma0':gamma0, 'A0':A0, 'xi_star':xi_star}


    for itr in tqdm(range(0, n_iters-1), total=n_iters-1, desc='Batch ODE Sim Progression'):
        #get updates using chosen solver
        updates = do_ODE_step(get_netout, net, curr_tilde_xs, curr_tilde_xs_es, curr_xs, \
                              curr_xs_es, taus[itr], taus[itr+1], shape, net_kwargs, solver=solver)
        
        #carry out changes to variables we need to keep track of 
        curr_xs_es, curr_xs = updates['xs_es'], updates['xs']
        curr_tilde_xs_es, curr_tilde_xs = updates['tilde_xs_es'], updates['tilde_xs']
        
        
        #now save things (if on first update, multiple of save_freq, or last update)
        if (itr%save_freq==0) or (itr==n_iters-2):
            res_dict['tilde_xs'].append(curr_tilde_xs.cpu().numpy())
            res_dict['tilde_xs_es'].append(curr_tilde_xs_es.cpu().numpy())
            
            res_dict['xs'].append(curr_xs.cpu().numpy())
            res_dict['xs_es'].append(curr_xs_es.cpu().numpy())
            
            res_dict['net_outs'].append(updates['net_outputs'].cpu().numpy())
            res_dict['unscaled_scores'].append(updates['unscaled_scores'].cpu().numpy())
            
            res_dict['dxs'].append(updates['dxs'].cpu().numpy())
            res_dict['dxs_es'].append(updates['dxs_es'].cpu().numpy())
            
    return res_dict


#Wrapper method to simulate ODE for a batch of data

def sim_batch_net_ODE(batch, net, device, shape=None, int_mode='melt', n_iters=1501, end_time=15.01, \
                                          A0=1., save_freq=10, disc='vp_ode', solver='heun', eps=1e-2, endsim_imgs_only=False, **kwargs):
    """
    General wrapper to run melt/gen from trained HD networks. 
    Should be able to handle ODE simulation from either nets trained explicitly in our
    "IFs" schedule or on Karras schedule "PreTrained". 
    """
    #--------------------------------------------#
    #check what type of net we are dealing with 
    #adjust sim params acoordingly
    #--------------------------------------------#
    

    print('*'*40)
    print('Running {} from IFs net...'.format(int_mode))
    print('*'*40)

    #set/adjust args -- esp device for net attributes 
    g = torch.from_numpy(net.g.cpu().numpy()).type(torch.float32).to(device)
    data_eigs = torch.from_numpy(net.data_eigs).to(device)
    W = torch.from_numpy(net.W.cpu().numpy()).type(torch.float32).to(device) 
    gamma0 = net.gamma0
    rho=net.rho
        
    net.device=device
    net.g = g
    net.W = W 

    xi_star = np.amax(data_eigs.cpu().numpy())
    get_netout = compute_ifs_netout
        

    #--------------------------------------------------------#
    #Run actual ODE sim
    #--------------------------------------------------------#
    
    ode_res_dict = do_ODE_sim(batch, net, shape, g, W, xi_star, get_netout, int_mode, n_iters, end_time, A0, gamma0, rho, \
    save_freq, disc=disc, solver=solver, eps=eps)
    
    if endsim_imgs_only: 
        #return only unscaled imgs (in IS) at end of ODE sim 
        return np.array(ode_res_dict['tilde_xs'])[-1, :, :]
    
    return ode_res_dict







