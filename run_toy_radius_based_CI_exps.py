#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run Radius-Based CI experiments 
using toy dsets and nets. 
"""

#get imports
import numpy as np
import click 
import torch
import os
import pickle
from torch.utils.data import DataLoader 
import dnnlib
from torch_utils import distributed as dist
from ifs_ode_sim.toy_ODE import sim_batch_net_ODE


#-------------------------------------------------------------------------------------
# Now def our args 

@click.command()

#general sim args 
@click.option('--network', 'network_pkl',          help='Network pickle filename', metavar='PATH|URL',                                                                type=str, required=True)
@click.option('--outdir',                          help='Where to save the outputs', metavar='DIR',                                                                   type=str, required=True)
@click.option('--data_name',                       help='Which dset we will be sampling from (if running inflation|melt direction)', metavar='STR',                   type=str, required=True)
@click.option('--int_mode',                        help='Mode in which we wish to integrate ODE (i.e., fwd or bckwd in time). Melt==Inflation', metavar='melt|gen',   type=click.Choice(['melt', 'gen']), default='melt', show_default=True)
@click.option('--data_dim',                        help='Number of dimensions in original data.', metavar='INT',                                                      type=click.IntRange(min=2), default=2, show_default=True)
@click.option('--dims_to_keep',                    help='Number of original data dims to keep.', metavar='INT',                                                       type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--max_samples',                     help='Max num of test pts to sample for each random center.', metavar='INT',                                       type=click.IntRange(min=1), default=100000, show_default=True)
@click.option('--num_rcs',                         help='Number of random centers to run analysis over.', metavar='INT',                                              type=click.IntRange(min=2), default=100, show_default=True)
@click.option('--num_boundaries',                  help='Number of boundaries to construct around each random center.', metavar='INT',                                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--boundary_pts', 'bpts_per_radius', help='Number of boundary pts to sample per each boundary', metavar='INT',                                          type=click.IntRange(min=10), default=200, show_default=True)
@click.option('--max_radius',                      help='Max boundary radius val to use.', metavar='FLOAT',                                                           type=float, default=1.0, show_default=True)
@click.option('--min_radius',                      help='Min boundary radius val to use.', metavar='FLOAT',                                                           type=float, default=1e-6, show_default=True)
@click.option('--eps',                             help='Latent Space variance for compressed dimensions to be used (if running PRR schedule)', metavar='FLOAT',      type=float, default=1.0, show_default=True)
@click.option('--device_name',                     help='Device to run sim on', metavar='STR',                                                                        type=str, default='cuda')
@click.option('--verbose',                         help='If passed, prints out coverage tables during simulation.',                                                   is_flag=True)


#ODE int args 
@click.option('--steps', 'num_steps',             help='Number of ODE integration steps', metavar='INT',                                                             type=click.IntRange(min=1), default=701, show_default=True)
@click.option('--h',                              help='Step size for ODE integration', metavar='FLOAT',                                                             type=click.FloatRange(max=1e-1, max_open=True), default=1e-2, show_default=True)
@click.option('--end_vars',                       help='Ending vars per dim for scaling (A0)', metavar='FLOAT',                                                      type=float, default=1., show_default=True)
@click.option('--save_freq',                      help='How often to save ODE sim steps', metavar='INT',                                                             type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--bs',                             help='Batch size for ODE simulation.', metavar='INT',                                                              type=click.IntRange(min=1000), default=10000, show_default=True)

#----------------------------------------------------------------------------------------

def main(network_pkl, outdir, verbose, **kwargs):
    """Calls on other methods to run geometry-agnostic toy CI exps"""
    
    #init distributed option
    dist.init()    
    
    #get all other args and pass it to general opts dict
    opts = dnnlib.util.EasyDict(kwargs)
    
    #set up device
    device = torch.device(opts.device_name)
    
    #create output root directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    #set up schedule
    schedule = 'prp' if opts.data_dim == opts.dims_to_keep else 'prr'
    
    #set up init/end gamma_inv_scale
    if schedule == 'prp':
        init_gamma_scale = end_gamma_scale = np.ones(opts.data_dim)
        
    else: 
        compressed_gamma_scale = np.ones(opts.dims_to_keep)
        compressed_gamma_scale = np.concatenate([compressed_gamma_scale, np.ones(opts.data_dim - opts.dims_to_keep)*opts.eps])
        compressed_gamma_scale = np.sqrt(compressed_gamma_scale)
        uncompressed_gamma_scale = np.ones(opts.data_dim)
        
        if opts.int_mode=='melt':
            init_gamma_scale = uncompressed_gamma_scale
            end_gamma_scale = compressed_gamma_scale
        else: 
            #we are running gen mode, PRR
            init_gamma_scale = compressed_gamma_scale
            end_gamma_scale = uncompressed_gamma_scale
    
    
    #load network
    if dist.get_rank() != 0:
        torch.distributed.barrier() 
    
    dist.print0('*'*40)
    dist.print0(f'Loading network from "{network_pkl}"...')
    dist.print0('*'*40)
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f: 
        net = pickle.load(f)['ema'].to(device)
    
    # Other ranks follow.
    if dist.get_rank() == 0: 
        torch.distributed.barrier() 
            
    
    #sample random centers we will run analysis for 
    if opts.int_mode =='melt':
        sampling_fn_kwargs = {'int_mode':'melt', 
                              'dset_name':opts.data_name, 
                              'n':opts.num_rcs}
    else:
        sampling_fn_kwargs = {'int_mode':'gen', 
                              'data_dims':opts.data_dim, 
                              'dims_to_keep':opts.dims_to_keep, 
                              'device':device,
                              'n': opts.num_rcs, 
                               'eps': opts.eps}
        
    random_centers = dnnlib.util.sampling_fn_wrapper(**sampling_fn_kwargs)[0] #get just random centers! 
    
    #set up bpts_radii
    bpts_radii = np.linspace(opts.min_radius, opts.max_radius, opts.num_boundaries)
    
    #set up lists to accummulate roc results 
    all_roc_counts = []
    all_roc_rates = []
    all_aucs = []
    
    #now loop through these and run a complete radius-based analysis for each 
    for i in range(random_centers.shape[0]):
        dist.print0('*'*40)
        dist.print0('Running analyses for random center {}'.format(i))
        dist.print0('*'*40)
        
        #get curr rc
        curr_rc = random_centers[i, :]
        #make sure this is a np array! 
        if torch.is_tensor(curr_rc):
            curr_rc = curr_rc.cpu().numpy() 
        
        dist.print0('Sampling boundary and test pts ... ')
        #sample bpts 
        curr_rc_bpts = dnnlib.util.get_all_boundary_pts(bpts_radii, init_gamma_scale, \
                                                        opts.data_dim, opts.bpts_per_radius, \
                                                           center=curr_rc)
        #sample test_pts 
        sampling_fn_kwargs['n'] = opts.max_samples #change num of samples to take 
        #get testpts (num here varies per random center!)
        curr_rc_testpts = dnnlib.util.sample_rc_testpts(curr_rc, opts.max_radius, opts.data_dim, \
                                                            dnnlib.util.sampling_fn_wrapper, sampling_fn_kwargs)
        
        #compute initial coverages and gt decisions
        dist.print0('Calculating initial coverages ... ')
        
        preODE_coverages, gt_decisions = dnnlib.util.compute_preODE_coverages(curr_rc_testpts[0], \
                                                                              curr_rc, \
                                                                                  bpts_radii, init_gamma_scale, \
                                                                                      verbose=verbose)
        
        #save init coverages for curr rc 
        fname = 'init_coverages_rc{}_{}_direction_{}_schedule.txt'.format(i, opts.int_mode, schedule)
        dnnlib.util.dump_json(outdir, fname, preODE_coverages)

        
        dist.print0('Simulating ODE in {} direction for current random center and its pts ...'.format(opts.int_mode))
        
        #simulate ODE for this random center over batches
        curr_center_data = np.concatenate([np.expand_dims(curr_rc, axis=0), \
                                           curr_rc_testpts[0], curr_rc_bpts], axis=0)
        curr_center_dset = dnnlib.util.ToyDset(curr_center_data, np.zeros(curr_center_data.shape[0]))
        curr_center_loader = DataLoader(curr_center_dset, batch_size=opts.bs, shuffle=False, num_workers=1)
        
        #init rc_xs to hold results across batches
        curr_rc_xs = []
        
        for j, sample in enumerate(curr_center_loader):
            
            dist.print0('*'*40)
            dist.print0('Batch {}'.format(j))
            dist.print0('*'*40)
            
            curr_batch = sample[0].type(torch.float32).to(device) 
            curr_batch_results_dict = sim_batch_net_ODE(curr_batch, net, int_mode= opts.int_mode, \
                                                        n_iters= opts.num_steps, h=opts.h, \
                                                      A0=opts.end_vars, save_freq=opts.save_freq) 
            curr_rc_xs.append(np.array(curr_batch_results_dict['xs'])[-1, :, :])
        
        curr_rc_xs = np.concatenate(curr_rc_xs, axis=0)
            
        #compoute postODE coverages 
        dist.print0('Computing final coverages ... ')
        rc_nsamples = curr_rc_testpts[1]
        end_rc = curr_rc_xs[0, :]
        end_testpts = curr_rc_xs[1:rc_nsamples+1, :]
        end_bpts = curr_rc_xs[rc_nsamples+1:, :]
        postODE_coverages = dnnlib.util.compute_postODE_coverages(end_testpts, end_rc, \
                                                                  end_bpts, bpts_radii, end_gamma_scale, \
                                                                      bpts_per_radius=opts.bpts_per_radius, \
                                                                          verbose=verbose)
        
        #save end coverages for curr rc 
        fname = 'end_coverages_rc{}_{}_direction_{}_schedule.txt'.format(i, opts.int_mode, schedule)
        dnnlib.util.dump_json(outdir, fname, postODE_coverages)
     
                
        #compute postODE ROC counts, rates
        #these are done for each boundary!
        dist.print0('Computing ROC counts and rates ...')
        curr_rc_roc_counts = []
        curr_rc_roc_rates = []
        for j in range(bpts_radii.shape[0]):
            curr_gt_decision = gt_decisions[j, :] 
            curr_roc_counts, curr_roc_rates = dnnlib.util.compute_ROC_counts_rates(curr_gt_decision, \
                                                                                   end_testpts, end_rc, end_bpts, \
                                                                                       bpts_radii, end_gamma_scale, \
                                                                                           bpts_per_radius=opts.bpts_per_radius)
            curr_rc_roc_counts.append(curr_roc_counts)
            curr_rc_roc_rates.append(curr_roc_rates)
        
        all_roc_counts.append(np.array(curr_rc_roc_counts))
        all_roc_rates.append(np.array(curr_rc_roc_rates))
        
        dist.print0('Computing AUCs...')
        curr_rc_aucs = dnnlib.util.compute_AUCs(np.array(curr_rc_roc_rates), bpts_radii)
        all_aucs.append(curr_rc_aucs)
    
    #save ROC rates and counts for all centers 
    np.savez(os.path.join(outdir, 'roc_results.npz'), roc_counts=np.array(all_roc_counts), \
             roc_rates=np.array(all_roc_rates), aucs=np.array(all_aucs))
        
    
    return 

#----------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------