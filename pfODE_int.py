#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run melt, roundtrip, and generation 
for a batch of HD data (CF-10 for now).

Essentially this calls methods in ifs_ode_sim 
modules to run melt and generation

"""

import torch
import numpy as np
import os
import json
import pickle
from sklearn.covariance import EmpiricalCovariance
import click 
import gc 
import dnnlib
from torch_utils import misc
from torch_utils import distributed as dist 
from pfODE_sim.ODE import sim_batch_net_ODE

#-------------------------------------------------------------------------
@click.command()
@click.option('--save_dir',                help='Where to save the output results', metavar='DIR',                                                            type=str, required=True)
@click.option('--data_root',               help='Path to the dataset', metavar='ZIP|DIR',                                                                     type=str, required=True)
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                                                                type=str, required=True)
@click.option('--data_name',               help='Name of dataset we wish to run GT sims on', metavar='STR',                                                   type=str, default='CIFAR10', show_default=True)
@click.option('--sim_type',                help='Sim type to run', metavar='melt|roundtrip|gen|all',                                                          type=click.Choice(['melt', 'roundtrip', 'gen', 'all']), default='all', show_default=True)
@click.option('--disc',                    help='Discretization to use when simulating ODE', metavar='ifs|vp_ode',                                            type=click.Choice(['ifs', 'vp_ode']), default='vp_ode', show_default=True)
@click.option('--vpode_disc_eps',          help='Epsilon_s param for vp_ode discretization (if using this option).', metavar='FLOAT',                         type=float, default=1e-2, show_default=True)
@click.option('--solver',                  help='Solver to use when simulating ODE', metavar='euler|heun',                                                    type=click.Choice(['euler', 'heun']), default='heun', show_default=True)
@click.option('--bs',                      help='Maximum batch size', metavar='INT',                                                                          type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--seed',                    help='Seed for constructing gen samples', metavar='INT',                                                           type=int, default=42, show_default=True)


@click.option('--n_iters',                 help='Number of sampling steps', metavar='INT',                                                                    type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--end_time',                help='End melt/start gen time for ODE integration', metavar='FLOAT',                                               type=float, default=15.01, show_default=True)
@click.option('--end_vars',                help='Ending vars per dim', metavar='FLOAT',                                                                       type=float, default=1., show_default=True)
@click.option('--save_freq',               help='How often to save ODE sim steps', metavar='INT',                                                             type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--dims_to_keep',            help='Number of original data dims to keep.', metavar='INT',                                                       type=click.IntRange(min=1), default=3072, show_default=True)
@click.option('--device_name',             help='Name of device we wish to run simulation on.', metavar='STR',                                                type=str, default='cuda', show_default=True)
@click.option('--img_size',                help='Size of imgs being processed.', metavar='INT',                                                               type=int, default=32, show_default=True)
@click.option('--img_ch',                  help='Number of channels in imgs being processed.', metavar='INT',                                                 type=int, default=3, show_default=True)
@click.option('--eps',                     help='Variance to be used when sampling compressed dims during PRR gen.', metavar='FLOAT',                         type=float, default=1., show_default=True)
@click.option('--prev_melt',               help='Path to previous melt results to be used for rdtrp or gen (with empirical covariance', metavar='STR',        type=str, default='', show_default=True)
@click.option('--gen_source',              help='Method used to construct gen samples', metavar='diag|empirical',                                             type=click.Choice(['diag', 'empirical']), default='diag', show_default=True)


#-------------------------------------------------------------------------

def main(save_dir, data_root, network_pkl, **kwargs): 
    
    #-------------------------------------------------#
    #General SetUp
    #-------------------------------------------------#
    
    #get all other args and pass it to general opts dict
    opts = dnnlib.util.EasyDict(kwargs)
    
    #init dist mode
    dist.init()
    
    #set up device
    device = torch.device(opts.device_name)     
    
    #set up saving dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)    
    
    #set up schedule name 
    flat_dims = opts.img_ch*(opts.img_size**2)
    assert flat_dims >= opts.dims_to_keep, 'Dims to keep needs to be less or equal to total dims!'
    schedule = 'PRP' if opts.dims_to_keep==flat_dims else 'PRRto{}'.format(opts.dims_to_keep)
    
    #load network 
    if dist.get_rank() != 0:
        torch.distributed.barrier()    

    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)
    
    if dist.get_rank() == 0:
        torch.distributed.barrier()     
        
    
    #get Kimgs (for output fname) 
    pkl_base_fname = os.path.basename(network_pkl) 
    kimgs = pkl_base_fname.split('-')[-1][:-4] 
    
    #set up dset obj, dset sampler, and data_loader_kwargs 
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data_root, use_labels=False, \
                                     xflip=False, cache=True) 
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=1, prefetch_factor=2) 
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size())
    
    
    #check that previous melt is passed if running solo rdtrp or solo gen w/ empirical cov
    if opts.sim_type == 'roundtrip' or (opts.sim_type =='gen' and opts.gen_source=='empirical'):
        assert opts.prev_melt!='', 'To run solo roundtrip or gen with empirical covariance samples, a previous melt is needed!'
    
    #--------------------------------------------------#
    #Run desired sim(s)
    #--------------------------------------------------#
    
    if opts.sim_type=='all' or opts.sim_type=='melt':
        
        dist.print0('Starting {} melt for {} {} bs...'.format(schedule, opts.data_name, opts.bs))
        
        #set up data for melt
        dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler,\
                                                    batch_size=opts.bs, **data_loader_kwargs))
        
        #get batch of data 
        melt_samples = next(dataset_iterator)
        melt_samples = melt_samples[0].reshape(-1, flat_dims).type(torch.float32) #flatten imgs
        melt_samples = melt_samples/127.5 - 1 #center and scale 
        
        #run actual melt 
        melt_res = sim_batch_net_ODE(melt_samples.to(device), net, device, shape=[opts.bs, opts.img_ch, opts.img_size, opts.img_size], \
                                     int_mode='melt', n_iters=opts.n_iters, end_time=opts.end_time, \
                                         A0=opts.end_vars, save_freq=opts.save_freq, disc=opts.disc, solver=opts.solver, \
                                             eps=opts.vpode_disc_eps)
        
        dist.print0('Saving melt results ...')
        

        melt_fname = '{}_IFsNet_{}_melt_{}niters_{}endtime_{}A0_{}rho_{}gamma0_{}savefreq_{}Kimgs_{}bs.npz'.format(opts.data_name, schedule, opts.n_iters, \
                                                                                                           opts.end_time, opts.end_vars, net.rho, net.gamma0, \
                                                                                                               opts.save_freq, kimgs, opts.bs)
        np.savez(os.path.join(save_dir, melt_fname), tilde_xs=melt_res['tilde_xs'], xs=melt_res['xs'], \
                 tilde_xs_es=melt_res['tilde_xs_es'], xs_es=melt_res['xs_es'], dxs=melt_res['dxs'], \
                     dxs_es=melt_res['dxs_es'], net_outs=melt_res['net_outs'], unscaled_scores=melt_res['unscaled_scores'])
            
    if opts.sim_type=='all' or opts.sim_type=='roundtrip':
        #run roundtrip
        dist.print0('Starting {} roundtrip for {} {}bs...'.format(schedule, opts.data_name, opts.bs))
        
        if opts.sim_type == 'all':
            #get end melt from previous melt run (above)
            end_melt = np.array(melt_res['xs_es'])[-1, :, :]
        else: 
            melt_res = np.load(opts.prev_melt)
            end_melt = melt_res['xs_es'][-1, :, :]
        
        
        batch = torch.from_numpy(end_melt).type(torch.float32).to(device)
        del melt_res
        gc.collect()
        
        rdtrp_res = sim_batch_net_ODE(batch, net, device, shape=[opts.bs, opts.img_ch, opts.img_size, opts.img_size], \
                                     int_mode='gen', n_iters=opts.n_iters, end_time=opts.end_time, \
                                         A0=opts.end_vars, save_freq=opts.save_freq, disc=opts.disc, solver=opts.solver, \
                                             eps=opts.vpode_disc_eps)
        
        
        dist.print0('Saving roundtrip results ...')
        

        rdtrp_fname = '{}_IFsNet_{}_roundtrip_{}niters_{}endtime_{}A0_{}rho_{}gamma0_{}savefreq_{}Kimgs_{}bs.npz'.format(opts.data_name, schedule, opts.n_iters, \
                                                                                                           opts.end_time, opts.end_vars, net.rho, net.gamma0, \
                                                                                                               opts.save_freq, kimgs, opts.bs)
                
        np.savez(os.path.join(save_dir, rdtrp_fname), tilde_xs=rdtrp_res['tilde_xs'], xs=rdtrp_res['xs'], \
                 tilde_xs_es=rdtrp_res['tilde_xs_es'], xs_es=rdtrp_res['xs_es'], dxs=rdtrp_res['dxs'], \
                     dxs_es=rdtrp_res['dxs_es'], net_outs=rdtrp_res['net_outs'], unscaled_scores=rdtrp_res['unscaled_scores'])
        
        del rdtrp_res
        gc.collect()
        
    if opts.sim_type=='all' or opts.sim_type=='gen':
        
        #run gen 
        dist.print0('Starting {} generation for {} {}bs'.format(schedule, opts.data_name, opts.bs))
        
        if opts.gen_source=='empirical':
            if opts.sim_type=='gen':
                #get end melt from prev_melt file given 
                melt_res = np.load(opts.prev_melt)
                end_melt = melt_res['xs_es'][-1, :, :]
                
            cov=EmpiricalCovariance().fit(end_melt)
            emp_cov_diag = np.diagonal(cov.covariance_) #get diagonal only 
            batch = dnnlib.util.get_mvn_samples(np.zeros(flat_dims), np.diag(emp_cov_diag), opts.bs, random_state=opts.seed)
            
        else: 
            #sample using diag cov 
            if schedule=='PRP':
                batch = dnnlib.util.get_mvn_samples(np.zeros(flat_dims), np.eye(flat_dims), opts.bs, random_state=opts.seed)
            else: 
                cov=np.ones(opts.dims_to_keep)
                cov = np.append(cov, (np.ones(flat_dims - opts.dims_to_keep)*opts.eps))
                batch = dnnlib.util.get_mvn_samples(np.zeros(flat_dims), np.diag(cov), opts.bs, random_state=opts.seed) 
            
        batch = torch.from_numpy(batch).type(torch.float32).to(device)            
        gen_res = sim_batch_net_ODE(batch, net, device, shape=[opts.bs, opts.img_ch, opts.img_size, opts.img_size], \
                                     int_mode='gen', n_iters=opts.n_iters, end_time=opts.end_time, \
                                         A0=opts.end_vars, save_freq=opts.save_freq, disc=opts.disc, solver=opts.solver, \
                                             eps=opts.vpode_disc_eps) 

        dist.print0('Saving gen results ...')
        

        gen_fname = '{}_IFsNet_{}_gen_{}niters_{}endtime_{}A0_{}rho_{}gamma0_{}savefreq_{}Kimgs_{}bs'.format(opts.data_name, schedule, opts.n_iters, \
                                                                                                           opts.end_time, opts.end_vars, net.rho, net.gamma0, \
                                                                                                               opts.save_freq, kimgs, opts.bs)           
        if opts.gen_source=='empirical':
            gen_fname+='_EmpiricalCov.npz'
        else: 
            gen_fname+='_DiagCov_{}eps.npz'.format(opts.eps)
        
        np.savez(os.path.join(save_dir, gen_fname), tilde_xs=gen_res['tilde_xs'], xs=gen_res['xs'], \
                 tilde_xs_es=gen_res['tilde_xs_es'], xs_es=gen_res['xs_es'], dxs=gen_res['dxs'], \
                     dxs_es=gen_res['dxs_es'], net_outs=gen_res['net_outs'], unscaled_scores=gen_res['unscaled_scores'])
            

    #log params to json file 
    opts.update(save_dir=save_dir)
    opts.update(network=network_pkl)
    opts.update(data_root=data_root)
    
    #set up fname for output 
    fname = '{}_{}_HD_pfODE_int_sim_params.json'.format(opts.data_name, schedule)

    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, fname), 'wt') as f:
            json.dump(opts, f, indent=2)
                    
    return 

#---------------------------------------------------------------------#

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------#


