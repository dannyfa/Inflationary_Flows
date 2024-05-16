#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script to compute Autocorrelations for 
network outputs and network residuals 
for nets trained on HD datasets. 

For now, this includes only our de novo nets.

"""

import torch
import numpy as np
import os
import pickle
import click 
from tqdm import tqdm 
import dnnlib
from torch_utils import misc
from torch_utils import distributed as dist 


#--------------------------------------------------------------------#

def calc_autocorr(X1, mu1, X2, mu2, device):
    """
    Computes(X1-mu1)@(X2-mu)^\top for a given batch.
    We then accumulate these results over batch elements
    and batches before taking expectation and applying scales.
    """
    #pass arrays to torch 
    X1 = torch.from_numpy(X1).type(torch.float32).to(device)
    mu1 = torch.from_numpy(mu1).type(torch.float32).to(device)
    
    X2 = torch.from_numpy(X2).type(torch.float32).to(device)
    mu2 = torch.from_numpy(mu2).type(torch.float32).to(device)
    
    #get diffs we want 
    X1_minus_mu1 = (X1 - mu1.unsqueeze(0).repeat(X1.shape[0], 1)).unsqueeze(-1) #bs, dim, 1
    X2_minus_mu2 = (X2 - mu2.unsqueeze(0).repeat(X2.shape[0], 1)).unsqueeze(-1) #bs, dim, 1
    
    #get inner products we want using einsum
    ac = torch.einsum('bjk, bki -> bji', X1_minus_mu1, torch.transpose(X2_minus_mu2, 1, 2)) #bs, dim, dim
    
    #collapse across batch elements
    ac = torch.sum(ac, dim=0) #dim, dim
    
    return ac.cpu().numpy()

#-----------------------------------------------------------------------#

#------------------------------------------------------------------------#

@click.group()
def main():
    """
    Calculate Scaled Cross-Correlation Matrices for De-Noiser Network Outputs
    and Residuals 
    
    Contains separate functions to:
        1) extract and save network outputs and their 
        residuals for a given number of samples and 
        given set of desired time pts; 
        2) use extracted values from (1) to compute scaled 
        cross-correlation matrices and save these. 
    """
#-------------------------------------------------------------------------#


@main.command()
@click.option('--save_dir',                help='Where to save the output results', metavar='DIR',                                                            type=str, required=True)
@click.option('--data_root',               help='Path to the dataset', metavar='ZIP|DIR',                                                                     type=str, required=True)
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                                                                type=str, required=True)
@click.option('--data_name',               help='Name of dataset we wish to use', metavar='STR',                                                              type=str, default='cifar10', show_default=True)

@click.option('--disc',                    help='Discretization to use when  constructing time pts to query', metavar='ifs|vp_ode',                           type=click.Choice(['ifs', 'vp_ode']), default='vp_ode', show_default=True)
@click.option('--vpode_disc_eps',          help='Epsilon_s param for vp_ode discretization (if using this option).', metavar='FLOAT',                         type=float, default=1e-2, show_default=True)
@click.option('--n_time_pts',              help='N for time discretization schedule', metavar='INT',                                                          type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--end_time',                help='End melt/inflation time net was trained on', metavar='FLOAT',                                                type=float, default=15.01, show_default=True)

@click.option('--n_samples',               help='Total number of samples to compute AC over', metavar='INT',                                                  type=int, default=50000, show_default=True)
@click.option('--bs',                      help='Maximum batch size', metavar='INT',                                                                          type=click.IntRange(min=1), default=500, show_default=True)

@click.option('--device_name',             help='Name of device we wish to run simulation on.', metavar='STR',                                                type=str, default='cuda', show_default=True)
@click.option('--img_size',                help='Size of imgs being processed.', metavar='INT',                                                               type=int, default=32, show_default=True)
@click.option('--img_ch',                  help='Number of channels in imgs being processed.', metavar='INT',                                                 type=int, default=3, show_default=True)
@click.option('--dims_to_keep',            help='Number of original data dims to keep.', metavar='INT',                                                       type=click.IntRange(min=1), default=3072, show_default=True)
    

def extract(save_dir, data_root, network_pkl, **kwargs):

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
        
    #make sure net g, W are on proper device
    #and update net device attribute 
    g = torch.from_numpy(net.g.cpu().numpy()).type(torch.float32).to(device)
    W = torch.from_numpy(net.W.cpu().numpy()).type(torch.float32).to(device)
    net.g = g
    net.W = W
    net.device = device 
    
    #set up dset obj, dset sampler, and data_loader_kwargs 
    # make sure loader and sampler do NOT shuffle data here... 
    
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data_root, use_labels=False, \
                                     xflip=False, cache=True) 
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=1, prefetch_factor=2, shuffle=False) 
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), \
                                           num_replicas=dist.get_world_size(), shuffle=False)
    dataset_iter = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler,\
                                                    batch_size=opts.bs, **data_loader_kwargs))
        
    #get time pts of interest 
    time_pts = dnnlib.util.get_disc_times(opts.disc, end_time=opts.end_time, n_iters=opts.n_time_pts, int_mode='melt', eps=opts.vpode_disc_eps)
    
    
    #-----------------------------------------------#
    # Calc netoutputs, scores
    #-----------------------------------------------#
 
    for t in range(time_pts.shape[0]):
        dist.print0('*'*40)
        dist.print0("Extracting Residuals for Time Pt {:.4f}".format(time_pts[t]))
        dist.print0('*'*40)
        

        curr_t_all_netoutputs = []
        curr_t_all_netresiduals = []
        curr_gamma_t = dnnlib.util.get_gamma(net.g, time_pts[t], gamma0=net.gamma0, rho=net.rho)
        
        n_samples_seen = 0
        
        while n_samples_seen < opts.n_samples:
            
            batch = next(dataset_iter)
            batch = batch[0].reshape(-1, flat_dims).type(torch.float32).to(device) #flatten dims, pass to device 
            batch = batch/127.5 - 1 #demean, standardize
            
            
            batch_es = torch.einsum('ij, bjk -> bik', net.W.T, batch.unsqueeze(-1)).squeeze(-1) #pass to ES 
            
            
            #sample noise 
            noise = torch.randn(batch.shape).type(torch.float32).to(device)
            noise *= torch.sqrt(curr_gamma_t)[None, :]
            
            #add noise to batch in ES (this is input to net)
            curr_x_es = (batch_es + noise).reshape(batch.shape[0], opts.img_ch, opts.img_size, opts.img_size)
            
            #get noise conditioning input to net 
            ts = torch.ones(batch.shape[0]).type(torch.float32).to(device) * time_pts[t] 
            
            #now get netoutputs for curr_x
            with torch.no_grad():
                curr_D_x, _ = net(curr_x_es, ts) #D_x is in IS and flattened 
            
            #ok now get network residuals 
            curr_net_res = (curr_D_x - batch)
            
            #accumulate outputs and residuals 
            curr_t_all_netoutputs.append(curr_D_x.cpu().numpy())
            curr_t_all_netresiduals.append(curr_net_res.cpu().numpy())
            
            #update samples seen
            n_samples_seen += batch.shape[0]
            
    
        #concatenate across batches 
        curr_t_all_netoutputs = np.concatenate(curr_t_all_netoutputs, axis=0)
        curr_t_all_netresiduals = np.concatenate(curr_t_all_netresiduals, axis=0)
        
        
        #save these 
        fname = '{}_{}_net_outputs_residuals_time{}.npz'.format(opts.data_name, schedule, t)
        np.savez(os.path.join(save_dir, fname), net_outputs = curr_t_all_netoutputs, \
             net_residuals = curr_t_all_netresiduals)
            
    return 


#-------------------------------------------------------------------------#

#-------------------------------------------------------------------------#


@main.command()
@click.option('--save_dir',                help='Where to save the output results', metavar='DIR',                                                            type=str, required=True)
@click.option('--data_name',               help='Name of toy dataset to use', metavar='STR',                                                                  type=str, default='circles', show_default=True)
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                                                                type=str, required=True)

@click.option('--n_time_pts',              help='N for time linearly spaced discretization schedule', metavar='INT',                                          type=click.IntRange(min=1), default=701, show_default=True)
@click.option('--h',                       help='Step size to use when constructing disc. schedule', metavar='FLOAT',                                         type=float, default=1e-2, show_default=True)

@click.option('--n_samples',               help='Total number of samples to compute AC over', metavar='INT',                                                  type=int, default=10000, show_default=True)

@click.option('--device_name',             help='Name of device we wish to run simulation on.', metavar='STR',                                                type=str, default='cuda', show_default=True)
@click.option('--data_dim',                help='Dimensionality of toy data.', metavar='INT',                                                                 type=int, default=2, show_default=True)
@click.option('--dims_to_keep',            help='Number of original data dims to keep.', metavar='INT',                                                       type=click.IntRange(min=1), default=2, show_default=True)

def extracttoys(save_dir, data_name, network_pkl, **kwargs):
    """
    Similar to extract, but adapted to toy datasets/networks.
    
    Runs network output and residual extraction for all "N" total 
    samples at once (i.e., no batch computation)
    
    Only uses linearly spaced discretization schedule. 
    """
    
    #get all other args and pass it to general opts dict
    opts = dnnlib.util.EasyDict(kwargs)
    
    #init dist mode
    dist.init()    
    
    #set up device
    device = torch.device(opts.device_name)   


    #set up saving dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)       
        
    
    #set up schedule
    schedule = 'PRP' if opts.data_dim==opts.dims_to_keep else 'PRRto{}'.format(opts.dims_to_keep)

    #load network
    if dist.get_rank() != 0:
        torch.distributed.barrier()    

    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)
    
    if dist.get_rank() == 0:
        torch.distributed.barrier()     
    
    #make sure net g, W are on proper device
    #and update net device attribute 
    g = torch.from_numpy(net.g.cpu().numpy()).type(torch.float32).to(device)
    W = torch.from_numpy(net.W.cpu().numpy()).type(torch.float32).to(device)
    net.g = g
    net.W = W
    net.device = device 
        
    
    #get disc times 
    time_pts = dnnlib.util.get_disc_times('ifs', end_time=(opts.h*opts.n_time_pts), n_iters=opts.n_time_pts, int_mode='melt')
    
    #get data samples 
    toy_data = dnnlib.util.get_toy_dset(data_name, opts.n_samples)[0]
    toy_data = torch.from_numpy(toy_data).type(torch.float32).to(device)
    
    #init variables to hold net outputs, output residuals 
    all_netouts = []
    all_net_residuals = []

    for itr in tqdm(range(0, time_pts.shape[0]), total=time_pts.shape[0], desc='Net Residuals Extraction Progression'):
        #get gamma(t)
        curr_gammat = dnnlib.util.get_gamma(net.g, time_pts[itr], gamma0=net.gamma0, rho=net.rho)
        #sample noise for this t 
        curr_noise = torch.randn(toy_data.shape[0], toy_data.shape[1]).to(device) * torch.sqrt(curr_gammat)[None, :] 
        
        #add noise to samples in ES 
        samples_es = torch.einsum('ij, bjk -> bik', net.W.T, toy_data.unsqueeze(-1)).squeeze(-1)
        samples_n_es = samples_es + curr_noise
        
        #now feed these to net to get an output 
        ts = torch.ones(toy_data.shape[0]).to(device) * time_pts[itr]
        with torch.no_grad(): 
            D_x, _ = net(samples_n_es, ts) #output in net.space 
            all_netouts.append(D_x.cpu().numpy()) 
        
        #get net output residuals 
        #account for toy nets trained in different bases 
        if net.space == 'IS': 
            net_residuals = D_x - toy_data
        else: 
            net_residuals = D_x - samples_es
        
        all_net_residuals.append(net_residuals.cpu().numpy())
        
    
    
    #save these 
    fname = '{}_{}_toy_net_outputs_residuals.npz'.format(data_name, schedule)
    np.savez(os.path.join(save_dir, fname), net_outputs = np.array(all_netouts), \
             net_residuals = np.array(all_net_residuals))    
    
    return 

#-------------------------------------------------------------------------#



@main.command()
@click.option('--save_dir',                help='Where to save the output results', metavar='DIR',                                                            type=str, required=True)
@click.option('--res_root',                help='Path to residuals extracted', metavar='DIR',                                                                 type=str, required=True)
@click.option('--data_name',               help='Name of dataset we wish to use', metavar='STR',                                                              type=str, default='CIFAR10', show_default=True)

@click.option('--n_time_pts',              help='N for time discretization schedule', metavar='INT',                                                          type=click.IntRange(min=1), default=256, show_default=True)

@click.option('--img_size',                help='Size of imgs being processed.', metavar='INT',                                                               type=int, default=32, show_default=True)
@click.option('--img_ch',                  help='Number of channels in imgs being processed.', metavar='INT',                                                 type=int, default=3, show_default=True)
@click.option('--dims_to_keep',            help='Number of original data dims to keep.', metavar='INT',                                                       type=click.IntRange(min=1), default=3072, show_default=True)
@click.option('--n_samples',               help='Total number of samples to compute AC over', metavar='INT',                                                  type=int, default=50000, show_default=True)

@click.option('--bs',                      help='Maximum batch size', metavar='INT',                                                                          type=click.IntRange(min=1), default=1000, show_default=True)
@click.option('--device_name',             help='Name of device we wish to run simulation on.', metavar='STR',                                                type=str, default='cuda', show_default=True)


def calc(save_dir, res_root, **kwargs):

    #get all other args and pass it to general opts dict
    opts = dnnlib.util.EasyDict(kwargs)
    
    #set up device
    device = torch.device(opts.device_name)   


    #set up saving dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)        
    
    #set up schedule name 
    flat_dims = opts.img_ch*(opts.img_size**2)
    assert flat_dims >= opts.dims_to_keep, 'Dims to keep needs to be less or equal to total dims!'
    schedule = 'PRP' if opts.dims_to_keep==flat_dims else 'PRRto{}'.format(opts.dims_to_keep)
    
    #get ref time pt (zeroth one for convenience)
    ref_fname = '{}_{}_net_outputs_residuals_time0.npz'.format(opts.data_name, schedule)
    ref_vals = np.load(os.path.join(res_root, ref_fname))
    
    #calc num_batches
    if (opts.n_samples/opts.bs) % 1 ==0: 
        num_batches = opts.n_samples // opts.bs
    else: 
        num_batches = (opts.n_samples // opts.bs) + 1
    
    #compute mu_1, sigma_1
    mu_ref_netouts = np.mean(ref_vals['net_outputs'], axis=0)
    sigma_ref_netouts = np.std(ref_vals['net_outputs'], axis=0)
    
    mu_ref_netres = np.mean(ref_vals['net_residuals'], axis=0)
    sigma_ref_netres = np.std(ref_vals['net_residuals'], axis=0) 
    
    
    for t in range(opts.n_time_pts):
        
        print('*'*40)
        print('Computing Correlations for times 0 and {}'.format(t))
        print('*'*40)
        
        #load current X2 
        x2_fname = '{}_{}_net_outputs_residuals_time{}.npz'.format(opts.data_name, schedule, t)
        x2_vals = np.load(os.path.join(res_root, x2_fname))
        
        #get m2, sigma2 
        mu2_netouts = np.mean(x2_vals['net_outputs'], axis=0)
        sigma2_netouts = np.std(x2_vals['net_outputs'], axis=0)
        
        mu2_netres = np.mean(x2_vals['net_residuals'], axis=0)
        sigma2_netres = np.std(x2_vals['net_residuals'], axis=0)
        
        #loop through batches and calc (X1 - mu1) @ (X2 - mu2)
        all_acs_netouts = np.zeros((flat_dims, flat_dims))
        all_acs_netres = np.zeros((flat_dims, flat_dims))
        

        for b in tqdm(range(num_batches), total=num_batches, desc='Running through batches'):           
            #setup indices 
            if b != (num_batches-1):
                start_idx = b*opts.bs
                end_idx = (b+1)*opts.bs
            else: 
                start_idx = (num_batches-1)*opts.bs 
                end_idx = (opts.n_samples - start_idx) + start_idx
                
                
            #get curr batch netout acs 
            curr_xref_netout = ref_vals['net_outputs'][start_idx:end_idx, :] 
            curr_x2_netout = x2_vals['net_outputs'][start_idx:end_idx, :] 
            curr_netout_acs = calc_autocorr(curr_xref_netout, mu_ref_netouts, curr_x2_netout, mu2_netouts, device) 
            all_acs_netouts += curr_netout_acs
            
            #get curr batch netres acs 
            curr_xref_netres = ref_vals['net_residuals'][start_idx:end_idx, :] 
            curr_x2_netres = x2_vals['net_residuals'][start_idx:end_idx, :]  
            curr_netres_acs = calc_autocorr(curr_xref_netres, mu_ref_netres, curr_x2_netres, mu2_netres, device) 
            all_acs_netres += curr_netres_acs
                        
        
        #take avg over all samples to get E[(X1- mu1)(X2-mu2)^\top]
        #and scale by corresponding stds (\sigma_1, \sigma_2)
        all_acs_netouts /= opts.n_samples
        all_acs_netouts /= (sigma_ref_netouts * sigma2_netouts)
        
        all_acs_netres /= opts.n_samples
        all_acs_netres /= (sigma_ref_netres * sigma2_netres)
            
                        
        #save curr results 
        curr_fname = '{}_{}_autocorrelations_times0_{}.npz'.format(opts.data_name, schedule, t)
        
        np.savez(os.path.join(save_dir, curr_fname), net_output_acs = all_acs_netouts, \
                 net_residual_acs = all_acs_netres) 
    
    return

#---------------------------------------------------------------------#
@main.command()
@click.option('--save_dir',                help='Where to save the output results', metavar='DIR',                                                            type=str, required=True)
@click.option('--res_root',                help='(Full) Path to residuals extracted', metavar='DIR',                                                          type=str, required=True)
@click.option('--data_name',               help='Name of toy dataset we wish to use', metavar='STR',                                                          type=str, default='CIFAR10', show_default=True)

@click.option('--n_time_pts',              help='N for time discretization schedule', metavar='INT',                                                          type=click.IntRange(min=1), default=256, show_default=True)

@click.option('--data_dim',                help='Dimensionality of toy dataset.', metavar='INT',                                                              type=int, default=2, show_default=True)
@click.option('--dims_to_keep',            help='Number of original data dims to keep.', metavar='INT',                                                       type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--n_samples',               help='Total number of samples to compute AC over', metavar='INT',                                                  type=int, default=10000, show_default=True)

@click.option('--device_name',             help='Name of device we wish to run simulation on.', metavar='STR',                                                type=str, default='cuda', show_default=True)


def calctoys(save_dir, res_root, **kwargs):
    """
    Similar to calc method, only for toy data.
    """
    
    #get all other args and pass it to general opts dict
    opts = dnnlib.util.EasyDict(kwargs)
    
    #set up device
    device = torch.device(opts.device_name)   


    #set up saving dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)        
    
    #set up schedule name 
    schedule = 'PRP' if opts.data_dim==opts.dims_to_keep else 'PRRto{}'.format(opts.dims_to_keep)    
    
    #load .npz file with extracted results
    net_res_vals = np.load(res_root)
    
    #get mean, std for ref time pt 
    
    mu_ref_netouts = np.mean(net_res_vals['net_outputs'][0, :, :], axis=0)
    std_ref_netouts = np.std(net_res_vals['net_outputs'][0, :, :], axis=0)
    
    mu_ref_netres = np.mean(net_res_vals['net_residuals'][0, :, :], axis=0)
    std_ref_netres = np.std(net_res_vals['net_residuals'][0, :, :], axis=0)  
    
    
    all_netout_acs = []
    all_netres_acs = []
    
    #loop through time_pts collected and calc ACs
    for t in range(opts.n_time_pts):
        
        print('*'*40)
        print('Computing Correlations for times 0 and {}'.format(t))
        print('*'*40)
        
        
        #get x2 mean, std
        mu2_netouts = np.mean(net_res_vals['net_outputs'][t, :, :], axis=0)
        sigma2_netouts = np.std(net_res_vals['net_outputs'][t, :, :], axis=0)
        
        mu2_netres = np.mean(net_res_vals['net_residuals'][t, :, :], axis=0)
        sigma2_netres = np.std(net_res_vals['net_residuals'][t, :, :], axis=0)
        
        #now compute acs for this time lag
        curr_t_acs_netout = calc_autocorr(net_res_vals['net_outputs'][0, :, :], mu_ref_netouts, \
                                          net_res_vals['net_outputs'][t, :, :], mu2_netouts, device)
        curr_t_acs_netres = calc_autocorr(net_res_vals['net_residuals'][0, :, :], mu_ref_netres, \
                                          net_res_vals['net_residuals'][t, :, :], mu2_netres, device)
        
        #take avg over all samples to get E[(X1- mu1)(X2-mu2)^\top]
        #and scale by corresponding stds (\sigma_1, \sigma_2)
        curr_t_acs_netout /= opts.n_samples
        curr_t_acs_netout /= (std_ref_netouts * sigma2_netouts)
        all_netout_acs.append(curr_t_acs_netout)
        
        curr_t_acs_netres /= opts.n_samples
        curr_t_acs_netres /= (std_ref_netres * sigma2_netres)
        all_netres_acs.append(curr_t_acs_netres)
    
    #save these results 
    out_fname = '{}_{}_toy_autocorrelations.npz'.format(opts.data_name, schedule)
        
    np.savez(os.path.join(save_dir, out_fname), net_output_acs = np.array(all_netout_acs), \
                 net_residual_acs = np.array(all_netres_acs))        
    
    return 
    


#---------------------------------------------------------------------#

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------#    
      
  
  
            
            
            
    
    