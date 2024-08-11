# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Runs inflation (aka melt), roundtrip, and generation 
for toy datasets using either: 
    
    1) pfODE integration with discrete/"GT" score estimates
    2) pfODE integration with network-based score estimates 
    
Experiments in paper all used option (2), but option (1) is left
as a sanity check.

As in rest of code "melt" == "inflation" for methods
that take pfODE direction as part of their arguments.

This is done all at once (i.e., no batch computations)
for a fairly large sample size ~2k, ~20K samples
for our toy experiments. Using too high a sample size might yield CUDA OOM errors, 
depending on your specific system setup. 

"""

import os
import click
import json
import pickle
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from pfODE_sim.toy_ODE import sim_batch_discrete_ODE, sim_batch_net_ODE

#------------------------------------------------------------------------#

@click.group()
def main():
    """
    Integrate pfODEs for toy data using either discrete/"GT" (1)
    or network-based (2) score estimates. 
    
    Contains separate functions to:
        1)  simulate melt/inflation, roundtrip, and generation w/ our pfODEs
        using discrete/"GT" score estimates; 
        2) simulate melt/inflation, roundtrip, and generation w/ our pfODEs
        using network-based score estimates; 
    """
#-------------------------------------------------------------------------#


#----------------------------------------------------------------------------

@main.command()
@click.option('--save_dir',                help='Where to save the output of our simulations to', metavar='DIR',                                                                                       type=str, required=True)
@click.option('--data_name',               help='Which toy dset we will be running simulations for', metavar='STR',                                                                                    type=str, required=True)
@click.option('--total_samples',           help='Total number of samples to run simulations for', metavar='INT',                                                                                       type=click.IntRange(min=1000), default=2000, show_default=True)
@click.option('--augment_to',              help='Number of dimensions to embed our toy data into (linear manifold). Defaults to 0 (no embedding)', metavar='INT',                                      type=int, default=0, show_default=True)


@click.option('--steps', 'num_steps',      help='Number of ODE integration steps', metavar='INT',                                                                                                      type=click.IntRange(min=1), default=1501, show_default=True)
@click.option('--h',                       help='Step size for ODE integration', metavar='FLOAT',                                                                                                      type=click.FloatRange(max=1e-1, max_open=True), default=1e-2, show_default=True)
@click.option('--end_vars',                help='Ending variance per dim for scaing (A0)', metavar='FLOAT',                                                                                            type=float, default=1., show_default=True)
@click.option('--eps',                     help='Latent space compressed dimension variance (for PRR gen)', metavar='FLOAT',                                                                           type=float, default=1, show_default=True)
@click.option('--rho',                     help='Exponential growth/inflation constant', metavar='FLOAT',                                                                                              type=float, default=1., show_default=True)
@click.option('--gamma0',                  help='Minimum melting kernel variance', metavar='FLOAT',                                                                                                    type=float, default=5e-4, show_default=True)


@click.option('--save_freq',               help='How often to save ODE integration steps', metavar='INT',                                                                                              type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--data_dim',                help='Number of dimensions in original data.', metavar='INT',                                                                                               type=click.IntRange(min=2), default=2, show_default=True)
@click.option('--dims_to_keep',            help='Number of original data dims to keep.', metavar='INT',                                                                                                type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--g_type',                  help='Type of g construction to use.', metavar='orig|constant_inflation_gap',                                                                               type=click.Choice(['orig', 'constant_inflation_gap']), default='orig', show_default=True)
@click.option('--inflation_gap',           help='Inflation gap to use when constructing PRR schedule g, if using constant inflation gap option. Defaults to 1.0', metavar='FLOAT',                     type=float, default=1., show_default=True) 
@click.option('--device_name',             help='String indicating which device to use.', metavar='STR',                                                                                               type=str, default='cuda', show_default=True)

#-------------------------------------------------------------------------

def discrete(save_dir, data_name, total_samples, **kwargs):
    """
    
    Calls on sim_batch_discrete_ODE to run inflation/melt, roundtrip
    and generation for a large sample of a given toy dataset using 
    discrete/"GT" score estimates. 
    
    This is left as a sanity check -- not used in actual paper 
    experiments.
    
    """
    #get all other args and pass it to general opts dict
    opts = dnnlib.util.EasyDict(kwargs)
    
    #set up save dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    #setup device
    device = torch.device(opts.device_name)
    
    #setup schedule
    schedule='PRP' if opts.data_dim == opts.dims_to_keep else 'PRRto{}D'.format(opts.dims_to_keep)
    
    #get our samples 
    toy_samples = dnnlib.util.get_toy_dset(data_name, total_samples, augment_to=opts.augment_to)[0]
    #compute W, data_eigs
    data_eigs, _ , W = dnnlib.util.get_eigenvals_basis(toy_samples, n_comp=opts.data_dim)
    W = torch.from_numpy(W).type(torch.float32).to(device)
    #compute g to be used 
    g  = dnnlib.util.get_g(opts.data_dim, opts.dims_to_keep, opts.g_type, device, inflation_gap=opts.inflation_gap)
    #set up x0 (this is just original samples)
    x0 = torch.from_numpy(toy_samples).type(torch.float32).to(device)
    
    
    #run melt/inflation simulation 
    print('*'*40)
    print('Running melt simulation ... ')
    print('*'*40)
    batch = torch.from_numpy(toy_samples).type(torch.float32).to(device) #start from DS 
    melt_res = sim_batch_discrete_ODE(batch, x0, g, data_eigs, W, int_mode='melt', n_iters=opts.num_steps, h=opts.h, \
                                              gamma0=opts.gamma0, rho=opts.rho, A0=opts.end_vars, save_freq=opts.save_freq)
    np.savez(os.path.join(save_dir, '{}_{}_melt_results.npz'.format(data_name, schedule)), tilde_xs=melt_res['tilde_xs'], tilde_xs_es=melt_res['tilde_xs_es'], \
             xs=melt_res['xs'], xs_es=melt_res['xs_es'], unscaled_scores=melt_res['unscaled_scores'], gt_netout=melt_res['gt_netout'], \
                 dxs=melt_res['dxs'], dxs_es=melt_res['dxs_es'])
    
    #now run rdtrp 
    print('*'*40)
    print('Running roundtrip simulation ... ')
    print('*'*40)
    batch = torch.from_numpy(np.array(melt_res['xs_es'])[-1, :, :]).type(torch.float32).to(device)
    rdtrp_res = sim_batch_discrete_ODE(batch, x0, g, data_eigs, W, int_mode='gen', n_iters=opts.num_steps, h=opts.h, \
                                              gamma0=opts.gamma0, rho=opts.rho, A0=opts.end_vars, save_freq=opts.save_freq)    
    np.savez(os.path.join(save_dir, '{}_{}_rdtrp_results.npz'.format(data_name, schedule)), tilde_xs=rdtrp_res['tilde_xs'], tilde_xs_es=rdtrp_res['tilde_xs_es'], \
             xs=rdtrp_res['xs'], xs_es=rdtrp_res['xs_es'], unscaled_scores=rdtrp_res['unscaled_scores'], gt_netout=rdtrp_res['gt_netout'], \
                 dxs=rdtrp_res['dxs'], dxs_es=rdtrp_res['dxs_es'])

    #now run gen
    print('*'*40)
    print('Running gen simulation ... ')
    print('*'*40)
    if schedule == 'PRP': 
        batch = torch.randn(total_samples, opts.data_dim).type(torch.float32).to(device)
    else: 
        gen_cov_diag = np.append(np.ones(opts.dims_to_keep), np.ones(opts.data_dim-opts.dims_to_keep)*opts.eps)
        gen_scales = torch.from_numpy(np.sqrt(gen_cov_diag)).type(torch.float32).to(device)
        batch = torch.randn(total_samples, opts.data_dim).type(torch.float32).to(device)
        batch *= gen_scales[None, :]
    gen_res = sim_batch_discrete_ODE(batch, x0, g, data_eigs, W, int_mode='gen', n_iters=opts.num_steps, h=opts.h, \
                                              gamma0=opts.gamma0, rho=opts.rho, A0=opts.end_vars, save_freq=opts.save_freq)  
    np.savez(os.path.join(save_dir, '{}_{}_gen_results.npz'.format(data_name, schedule)), tilde_xs=gen_res['tilde_xs'], tilde_xs_es=gen_res['tilde_xs_es'], \
             xs=gen_res['xs'], xs_es=gen_res['xs_es'], unscaled_scores=gen_res['unscaled_scores'], gt_netout=gen_res['gt_netout'], \
                 dxs=gen_res['dxs'], dxs_es=gen_res['dxs_es'])
    
    #dump all sim params into a .json file 
    params_fname = '{}_{}_sim_params.json'.format(data_name, schedule)
    
    opts.update(save_dir=save_dir)
    opts.update(data_name=data_name)
    opts.update(total_samples=total_samples)    

    if dist.get_rank() == 0:
        with open(os.path.join(save_dir, params_fname), 'wt') as f:
            json.dump(opts, f, indent=2)   
    
    return 


#----------------------------------------------------------------------------

@main.command()
@click.option('--save_dir',                help='Where to save the output of our simulations to', metavar='DIR',                                                                           type=str, required=True)
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                                                                                             type=str, required=True)
@click.option('--data_name',               help='Which toy dset we will be running simulations for', metavar='STR',                                                                        type=str, required=True)
@click.option('--total_samples',           help='Total number of samples to run simulations for', metavar='INT',                                                                           type=click.IntRange(min=1000), default=2000, show_default=True)
@click.option('--augment_to',              help='Number of dimensions to embed our toy data into (linear manifold). Defaults to 0 (no embedding)', metavar='INT',                          type=int, default=0, show_default=True)


@click.option('--steps', 'num_steps',      help='Number of ODE integration steps', metavar='INT',                                                                                          type=click.IntRange(min=1), default=1501, show_default=True)
@click.option('--h',                       help='Step size for ODE integration', metavar='FLOAT',                                                                                          type=click.FloatRange(max=1e-1, max_open=True), default=1e-2, show_default=True)
@click.option('--end_vars',                help='Ending variance per dim for scaling (A0)', metavar='FLOAT',                                                                               type=float, default=1., show_default=True)
@click.option('--eps',                     help='Latent space compressed dimensions variance (for PRR gen)', metavar='FLOAT',                                                              type=float, default=1, show_default=True)


@click.option('--save_freq',               help='How often to save ODE integration steps', metavar='INT',                                                                                  type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--data_dim',                help='Number of dimensions in original data.', metavar='INT',                                                                                   type=click.IntRange(min=2), default=2, show_default=True)
@click.option('--dims_to_keep',            help='Number of original data dims to keep.', metavar='INT',                                                                                    type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--device_name',             help='String indicating which device to use.', metavar='STR',                                                                                   type=str, default='cuda', show_default=True)

#-------------------------------------------------------------------------

def net(save_dir, network_pkl, data_name, total_samples, **kwargs):
    """
    
    Calls on sim_batch_net_ODE to run inflation/melt, roundtrip
    and generation for a large sample of a given toy dataset using 
    network-based score estimates. 
    
    This is what we use for toy experiments in paper.
    
    """    
    #get all other args and pass it to general opts dict
    opts = dnnlib.util.EasyDict(kwargs)
    
    #set up save dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    #setup device
    device = torch.device(opts.device_name)
    
    #setup schedule
    schedule='PRP' if opts.data_dim == opts.dims_to_keep else 'PRRto{}D'.format(opts.dims_to_keep)    
    
    #load network
    dist.init()    
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()    


    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)
    
    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()    
        
    
    #run melt 
    dist.print0('*'*40)
    dist.print0('Running melt simulation ... ')
    dist.print0('*'*40)
    
    batch = dnnlib.util.get_toy_dset(data_name, total_samples, augment_to=opts.augment_to)[0]
    batch = torch.from_numpy(batch).type(torch.float32).to(device)
    melt_res = sim_batch_net_ODE(batch, net, int_mode='melt', n_iters=opts.num_steps, h=opts.h, A0=opts.end_vars, save_freq=opts.save_freq)
    np.savez(os.path.join(save_dir, '{}_{}_net_melt_results.npz'.format(data_name, schedule)), tilde_xs=melt_res['tilde_xs'], tilde_xs_es=melt_res['tilde_xs_es'], \
             xs=melt_res['xs'], xs_es=melt_res['xs_es'], unscaled_scores=melt_res['unscaled_scores'], gt_netout=melt_res['net_outs'], \
                 dxs=melt_res['dxs'], dxs_es=melt_res['dxs_es'])    
    
    #run rdtrp 
    dist.print0('*'*40)
    dist.print0('Running rdtrp imulation ... ')
    dist.print0('*'*40)    
    batch = torch.from_numpy(np.array(melt_res['xs_es'])[-1, :, :]).type(torch.float32).to(device)
    rdtrp_res = sim_batch_net_ODE(batch, net, int_mode='gen', n_iters=opts.num_steps, h=opts.h, A0=opts.end_vars, save_freq=opts.save_freq)
    np.savez(os.path.join(save_dir, '{}_{}_net_rdtrp_results.npz'.format(data_name, schedule)), tilde_xs=rdtrp_res['tilde_xs'], tilde_xs_es=rdtrp_res['tilde_xs_es'], \
             xs=rdtrp_res['xs'], xs_es=rdtrp_res['xs_es'], unscaled_scores=rdtrp_res['unscaled_scores'], gt_netout=rdtrp_res['net_outs'], \
                 dxs=rdtrp_res['dxs'], dxs_es=rdtrp_res['dxs_es'])   
    
    #run gen
    dist.print0('*'*40)
    dist.print0('Running gen imulation ... ')
    dist.print0('*'*40)  
    if schedule == 'PRP': 
        batch = torch.randn(total_samples, opts.data_dim).type(torch.float32).to(device)
    else: 
        gen_cov_diag = np.append(np.ones(opts.dims_to_keep), np.ones(opts.data_dim-opts.dims_to_keep)*opts.eps)
        gen_scales = torch.from_numpy(np.sqrt(gen_cov_diag)).type(torch.float32).to(device)
        batch = torch.randn(total_samples, opts.data_dim).type(torch.float32).to(device)
        batch *= gen_scales[None, :]
    gen_res = sim_batch_net_ODE(batch, net, int_mode='gen', n_iters=opts.num_steps, h=opts.h, A0=opts.end_vars, save_freq=opts.save_freq)
    np.savez(os.path.join(save_dir, '{}_{}_net_gen_results.npz'.format(data_name, schedule)), tilde_xs=gen_res['tilde_xs'], tilde_xs_es=gen_res['tilde_xs_es'], \
             xs=gen_res['xs'], xs_es=gen_res['xs_es'], unscaled_scores=gen_res['unscaled_scores'], gt_netout=gen_res['net_outs'], \
                 dxs=gen_res['dxs'], dxs_es=gen_res['dxs_es']) 
    
    #dump all sim params into a .json file 
    params_fname = '{}_{}_net_sim_params.json'.format(data_name, schedule)
    
    opts.update(save_dir=save_dir)
    opts.update(network=network_pkl) 
    opts.update(data_name=data_name)
    opts.update(total_samples=total_samples)    

    if dist.get_rank() == 0:
        with open(os.path.join(save_dir, params_fname), 'wt') as f:
            json.dump(opts, f, indent=2)
    
    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')
    
    return  

#----------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------       
    