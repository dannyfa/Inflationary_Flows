#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Computes Roundtrip MSE for a given randomly sampled set of imgs. 

"""

import torch
import numpy as np
import os
import pickle
import click 
import json
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
@click.option('--data_name',               help='Name of dataset we wish to calc rdtrp mse for', metavar='STR',                                               type=str, default='cifar10', show_default=True)
@click.option('--disc',                    help='Discretization to use when simulating ODE', metavar='ifs|vp_ode',                                            type=click.Choice(['ifs', 'vp_ode']), default='vp_ode', show_default=True)
@click.option('--vpode_disc_eps',          help='Epsilon_s param for vp_ode discretization (if using this option).', metavar='FLOAT',                         type=float, default=1e-2, show_default=True)
@click.option('--solver',                  help='Solver to use when simulating ODE', metavar='euler|heun',                                                    type=click.Choice(['euler', 'heun']), default='heun', show_default=True)
@click.option('--bs',                      help='Batch size', metavar='INT',                                                                                  type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--total_samples',           help='Total number of samples we wish to compute rdtrp MSE over', metavar='INT',                                   type=click.IntRange(min=1), default=10000, show_default=True)
@click.option('--seed',                    help='Seed for sampling/shuffling initial imgs.', metavar='INT',                                                   type=int, default=42, show_default=True)


@click.option('--n_iters',                 help='Number of ODE integration steps', metavar='INT',                                                             type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--end_time',                help='End melt/inflation time for ODE integration', metavar='FLOAT',                                               type=float, default=15.01, show_default=True)
@click.option('--end_vars',                help='Ending vars per dim (A0)', metavar='FLOAT',                                                                  type=float, default=1., show_default=True)
@click.option('--dims_to_keep',            help='Number of original data dims to keep.', metavar='INT',                                                       type=click.IntRange(min=1), default=3072, show_default=True)
@click.option('--device_name',             help='Name of device we wish to run simulation on.', metavar='STR',                                                type=str, default='cuda', show_default=True)
@click.option('--img_size',                help='Size of imgs being processed.', metavar='INT',                                                               type=int, default=32, show_default=True)
@click.option('--img_ch',                  help='Number of channels in imgs being processed.', metavar='INT',                                                 type=int, default=3, show_default=True)

#-------------------------------------------------------------------------

def main(save_dir, data_root, network_pkl, **kwargs): 
    
    """
    
    Simulates rdtrp and computes MSE between original images
    and recovered images at end of rdtrp for a given network 
    and schedule. 
    
    
    Example: 
        \b torchrun --rdzv_endpoint=0.0.0.0:29501 calc_roundtrip_mse.py \
        --save_dir=mse-tmp --data_root=datasets/cifar10-32x32.zip \
        --network=networks/network.pkl --data_name=cifar10 --bs=1000 \
        --total_samples=10000 --seed=42 --dims_to_keep=3072
        
    """
        
    #get all other args and pass it to general opts dict
    opts = dnnlib.util.EasyDict(kwargs)
    
    #init dist mode
    dist.init()
    
    #set up device
    device = torch.device(opts.device_name)     
        
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
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=opts.seed)
    
    
    #set up dataset iterator 
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler,\
                                                    batch_size=opts.bs, **data_loader_kwargs))    
         
    
    #ok now loop through several batches, run roundtrip and compute sse for each 
    curr_nimg = 0
    curr_sse = 0.
    batch_idx = 0
    while (curr_nimg < opts.total_samples):
        #get batch for melting 
        melt_samples = next(dataset_iterator)
        melt_samples = melt_samples[0].reshape(-1, flat_dims).type(torch.float32) #flatten imgs
        melt_samples = melt_samples/127.5 - 1 #center and scale 
        dist.print0('*'*40)
        dist.print0('Starting rdtrp for batch {}...'.format(batch_idx))
        dist.print0('*'*40)
        #run melt 
        melt_res = sim_batch_net_ODE(melt_samples.to(device), net, device, shape=[opts.bs, opts.img_ch, opts.img_size, opts.img_size], \
                                     int_mode='melt', n_iters=opts.n_iters, end_time=opts.end_time, \
                                         A0=opts.end_vars, save_freq=10, disc=opts.disc, solver=opts.solver, \
                                             eps=opts.vpode_disc_eps, endsim_imgs_only=False)
        #run gen from the above 
        melted_imgs = torch.from_numpy(np.array(melt_res['xs_es'])[-1, :, :]).type(torch.float32).to(device)
        del melt_res
        gc.collect()
        
        rdtrp_imgs = sim_batch_net_ODE(melted_imgs, net, device, shape=[opts.bs, opts.img_ch, opts.img_size, opts.img_size], \
                                     int_mode='gen', n_iters=opts.n_iters, end_time=opts.end_time, \
                                         A0=opts.end_vars, save_freq=10, disc=opts.disc, solver=opts.solver, \
                                             eps=opts.vpode_disc_eps, endsim_imgs_only=True)
        
        #ok now compute SSE for this batch (note that we sum over dims and over samples here!)
        #get orig imgs again (w/out center, scaling and properly clipped)
        orig_imgs = (melt_samples*127.5 + 128).cpu().numpy()
        orig_imgs = np.clip(orig_imgs, 0, 255).astype(np.uint8)

        #get reconstructed imgs (again, w/out center, scaling and properly clipped)
        recovered_imgs = rdtrp_imgs*127.5 + 128 
        recovered_imgs = np.clip(recovered_imgs, 0, 255).astype(np.uint8)

        batch_sse = np.sum(np.sum((orig_imgs - recovered_imgs)**2, axis=-1), axis=0)
        batch_idx+= 1
        curr_sse+= batch_sse
        curr_nimg += opts.bs
    
    
    #now compute and report final mse (over all N samples)
    mse = curr_sse/(curr_nimg*flat_dims)
    dist.print0('MSE for {} Roundtrip over {} random samples: {:2f}'.format(schedule, opts.total_samples, mse))
    

    #add mse and remaining params to opts -- we will log this as a json file 
    opts.update(mse=mse)
    opts.update(network=network_pkl)
    opts.update(data_root=data_root)
    
    #set up fname for output 
    fname = '{}_Roundtrip_MSE_{}seed_{}kimgs.json'.format(schedule, opts.seed, kimgs)

    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, fname), 'wt') as f:
            json.dump(opts, f, indent=2)
            
    return 

#---------------------------------------------------------------------#

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------#
