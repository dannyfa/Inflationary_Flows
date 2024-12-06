#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs toy MCMC experiments for a given toy net.

For manuscript, we used 2D circles data networks.
"""

#general dependencies 
import torch
import torch.distributions as D 
from torchdyn.core import NeuralODE
import hamiltorch
import numpy as np
import pickle 
import os
import click 
import glob 

import dnnlib
from torch_utils import distributed as dist

#---------------------------------------------------------------------------


@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                                                                                                            type=str, required=True)
@click.option('--network',       help='Network pkl to use', metavar='STR',                                                                                                                   type=str, required=True)
@click.option('--gt_n',          help='Number of GT x samples to create.', metavar='INT',                                                                                                    type=int, default=2000)
@click.option('--gt_eps_mvn',    help='Variance of noise injected into GT samples.', metavar='FLOAT',                                                                                        type=float, default=1e-2, show_default=True) 
@click.option('--net_eps_cd',    help='Variance of compressed dimension in latent space. Should be set to 1.0 (no compression) for PRP case.', metavar='FLOAT',                              type=float, default=1.0, show_default=True) 
@click.option('--tmax',          help='Max ODE integration time. Should match max time net was trained on.', metavar='FLOAT',                                                                type=float, default=7.01, show_default=True) 
@click.option('--device_name',   help='Which GPU device to use.', metavar='STR',                                                                                                             type=str, default='cuda:0')
@click.option('--step_size',     help='Step size for MCMC NUTs sampler.', metavar='FLOAT',                                                                                                   type=float, default=1e-2, show_default=True) 
@click.option('--traj_length',   help='L param for NUTs sampler.', metavar='INT',                                                                                                            type=int, default=15)
@click.option('--burn',          help='Number of MCMC samples to burn', metavar='INT',                                                                                                       type=int, default=500)
@click.option('--mcmc_n',        help='Number of actual MCMC samples to calculate.', metavar='INT',                                                                                          type=int, default=1500)
@click.option('--acpt',          help='Desired Acceptance rate for MCMC sampler.', metavar='FLOAT',                                                                                          type=float, default=0.8, show_default=True) 
@click.option('--seed',          help='Seed to use.', metavar='INT',                                                                                                                         type=int, default=123)



def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    
    #setup device, 
    device = torch.device(opts.device_name)   
    
    #setup hamiltorch random seed
    hamiltorch.set_random_seed(opts.seed)

    #setup output dirs 
    if dist.get_rank() == 0:
        trajs_outdir = os.path.join(opts.outdir, 'trajs_outdir')
        os.makedirs(opts.outdir, exist_ok=True)
        os.makedirs(trajs_outdir, exist_ok=True)

    #load network 
    if dist.get_rank() != 0: 
        torch.distributed.barrier()
    dist.print0(f'Loading network from "{opts.network}"...') 
    with dnnlib.util.open_url(opts.network, verbose=(dist.get_rank() == 0)) as f: 
        net = pickle.load(f)['ema'].to(device)
    if dist.get_rank() == 0: 
        torch.distributed.barrier()     
        
    
    #generate GT data 
    dist.print0('Creating GT samples...')
    xs_noised, zs_is = dnnlib.util.get_gt_samples(net, opts.gt_n, device, \
                                      opts.gt_eps_mvn, opts.net_eps_cd, net.W.cpu(), opts.tmax)
    
  
    #def log_prob handle for hamiltorch... 
    def _log_prob(params, xs_samples=xs_noised, outdir=trajs_outdir,\
                 eps_cd=opts.net_eps_cd, eps_mvn=opts.gt_eps_mvn): 
        print('*'*40) 
        #use same GT GMM vals as in Table 12 
        gt_means = torch.Tensor([[0,0], [-0.05, 0], [0.05, 0]]).to(device) 
        stds = torch.sqrt(torch.Tensor([[1, eps_cd], [1, eps_cd], [1, eps_cd]])).to(device)
        stds_weights = torch.Tensor([[0.75, 0.75], [1e-1, 1], [1, 1e-1]]).to(device)
        stds *= stds_weights
    
        #set up weights from params 
        #enforce GMM weights as simplex!
        softmax = torch.nn.Softmax(dim=0)
        p_weights = softmax(params[0:3])  

        p_zs = params[3:].reshape(-1, 2) 
    
        print('Curr weights: {}'.format(p_weights))
        print('Curr pz: {}'.format(p_zs))
    
        #construct posterior GMM
        comp = D.Independent(D.Normal(gt_means, stds), 1) 
        p_mix = D.Categorical(p_weights) 
        p_gmm = D.MixtureSameFamily(p_mix, comp)
        #pass zs to ES before calculating ll1 -- GMM for z is defined in ES! 
        pzs_es = torch.einsum('ij, bjk -> bik', net.W.T, p_zs.unsqueeze(-1)).squeeze(-1)        
        ll1 = p_gmm.log_prob(pzs_es).sum() 

        #feed z's through our ODE method  
        ode_wrapper = dnnlib.util.sim_ode_wrapper(net, device, A0=1.).to(device)
        model = NeuralODE(ode_wrapper, solver='euler', sensitivity='adjoint').to(device) 
        t_span = torch.linspace(opts.tmax, 0., int(opts.tmax*100)).to(device)
        _, trajectory = model(p_zs, t_span)

        f_zs = trajectory[-1, :, :]    
        
        #recall that torch.dist defines Normal in terms of scales, NOT variances! 
        xs_obs = D.Normal(f_zs, torch.ones(f_zs.shape).to(device)*(np.sqrt(eps_mvn)))
    
        #select only one observed sample at random
        ll2 = xs_obs.log_prob(xs_samples).sum()
    
        print('Curr ll1: {}'.format(ll1))
        print('Curr ll2: {}'.format(ll2)) 
        
        #save params
        #weights are saved as their original vals - prior to Softmax 
        #this saves ENTIRE trajectories, not just samples 
        curr_params = torch.cat([params[0:3].detach(), \
                                 p_zs.detach().reshape(p_zs.shape[0]*p_zs.shape[1])], dim=0) 
        prev_files = glob.glob(os.path.join(outdir, 'curr_params_*.npz'))
        prev_file_ids = [int(x[-10:-4]) for x in prev_files if x is not None]
        curr_file_id = max(prev_file_ids, default=-1) + 1 
        np.savez(os.path.join(outdir, f'curr_params_{curr_file_id:06d}.npz'), \
                 curr_params.cpu().numpy())
            
        return (ll1 + ll2)
    
    #run HMC and collect samples + full trajectories 
    dist.print0('Running MCMC sampling...')
    
    #to speed up convergence, init zs/weights to their GT vals
    #GT vals for weights chosen below yield ~ [0.5, 0.25, 0.25] when fed to Softmax 
    gt_z = zs_is.reshape(zs_is.shape[0]*zs_is.shape[1])
    gt_weights = torch.Tensor([1.06, 0.37, 0.37]).to(device) 
    
    params_init = torch.cat([gt_weights, gt_z], dim=0)
    
    net = net.requires_grad_(True).to(device) #make sure net has require_grad()==True
    
    N_nuts = opts.burn + opts.mcmc_n
    
    #turn store_on_GPU flag to FALSE
    #avoid running out of CUDA RAM when collecting samples! 
    params_hmc_nuts = hamiltorch.sample(log_prob_func=_log_prob, params_init=params_init, \
                                        num_samples=N_nuts,step_size=opts.step_size, \
                                            num_steps_per_sample=opts.traj_length, \
                                                sampler=hamiltorch.Sampler.HMC_NUTS, burn=opts.burn, \
                                                    desired_accept_rate=opts.acpt, \
                                                        store_on_GPU = False)
    
    
    #save actual samples too -- burn ins NOT included here!
    samples = torch.cat(params_hmc_nuts).reshape(len(params_hmc_nuts), -1)
    softmax = torch.nn.Softmax(dim=1)
    sampled_weights = softmax(samples[:, 0:3])
    sampled_zs = samples[:, 3:]
    
    np.savez(os.path.join(opts.outdir, 'sampled_zs.npz'), sampled_zs.cpu().numpy())
    np.savez(os.path.join(opts.outdir, 'sampled_weights.npz'), sampled_weights.cpu().numpy())
    
    dist.print0('Done!')
    

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
