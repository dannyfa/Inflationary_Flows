#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script to run coverage experiments for 2D and 3D toys
using either 2D alpha-shapes or 3D alpha-meshes. 

"""
import os
import click
import pickle
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from ifs_ode_sim.toy_ODE import sim_batch_net_ODE 

#-------------------------------------------------------------------------------------
@click.command()
#general args
@click.option('--network', 'network_pkl',   help='Network pickle filename', metavar='PATH|URL',                                                                type=str, required=True)
@click.option('--save_dir',                 help='Where to save the output coverages', metavar='DIR',                                                          type=str, required=True)
@click.option('--data_name',                help='Which dset we will be simulating ODE for.', metavar='STR',                                                   type=str, required=True)
@click.option('--data_dim',                 help='Number of dimensions in original data.', metavar='INT',                                                      type=click.IntRange(min=2), default=2, show_default=True)
@click.option('--dims_to_keep',             help='Number of original data dims to keep.', metavar='INT',                                                       type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--device_name',              help='Str for device name we wish to use', metavar='STR',                                                          type=str, default='cuda', show_default=True)


#args for pfODE simulation
@click.option('--steps', 'num_steps',       help='Number of steps for ODE integration', metavar='INT',                                                         type=click.IntRange(min=1), default=701, show_default=True)
@click.option('--h',                        help='Step size for ODE integration', metavar='FLOAT',                                                             type=click.FloatRange(max=1e-1, max_open=True), default=1e-2, show_default=True)
@click.option('--end_vars',                 help='Ending variance per dim (for scaling, A0)', metavar='FLOAT',                                                 type=float, default=1., show_default=True)
@click.option('--eps',                      help='Latent space variance (at end of inflation) for compressed dimensions.', metavar='FLOAT',                    type=click.FloatRange(min=1e-40, min_open=True), default=1e-2, show_default=True)
@click.option('--save_freq',                help='How often to save ODE integration results', metavar='INT',                                                   type=click.IntRange(min=1), default=1, show_default=True)

#args for actual experiment set up 
@click.option('--boundary_pts', 'num_bpts', help='Number of boundary pts to sample per boundary', metavar='INT',                                               type=click.IntRange(min=1), default=200, show_default=True)
@click.option('--test_pts', 'num_testpts',  help='Number of test pts to sample', metavar='INT',                                                                type=click.IntRange(min=1), default=20000, show_default=True)
@click.option('--r_min',                    help='Smallest radius for boundaries', metavar='FLOAT',                                                            type=float, default=0.5, show_default=True)
@click.option('--r_max',                    help='Largest radius for boundaries', metavar='FLOAT',                                                             type=float, default=3.5, show_default=True)
@click.option('--r_num',                    help='Number of linearly spaced boundary radii to use', metavar='FLOAT',                                           type=int, default=7, show_default=True)
#----------------------------------------------------------------------------------------

def main(network_pkl, save_dir, data_name, **kwargs):
    """
    Calls scripts need to sample test pts and boundary points, 
    to compute coverages using 2D alpha-shapes or 3D alpha-meshes, 
    and to simulate corresponding pfODEs using a given network and sampled points. 
    
    Saves all coverages (initial and final) for each direction 
    of roundtrip ("inflation", "generation") to .txt files under the specified save_dir
    
    Also saves pfODE simulation results for both directions of roundtrip.
    """
    
    #init dist mode 
    dist.init()
    
    #get all other args and pass these to general opts dict
    opts = dnnlib.util.EasyDict(kwargs)  
    
    #setup device 
    device = torch.device(opts.device_name)
    
    #setup save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    #set up schedule name 
    schedule = 'PRP' if opts.data_dim==opts.dims_to_keep else 'PRRto{}D'.format(opts.data_dim - opts.dims_to_keep)
        
    #decide which coverage calc method to use
    calc_coverages = dnnlib.util.compute_coverages_2Dalphashape if opts.data_dim==2 else dnnlib.util.compute_coverages_3Dmesh
    
    #setup kwargs for (3D) mesh cases
    mesh_kwargs = {'save_dir':save_dir, 'schedule':schedule} if opts.data_dim==3 else {}
        
    #construct boundary radii
    bpts_radii = np.linspace(opts.r_min, opts.r_max, opts.r_num)
    
    #load network 
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    dist.print0(f'Loading network from "{network_pkl}"...') 
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)
    # Other ranks follow.
    if dist.get_rank() == 0: 
        torch.distributed.barrier()     
        
    #sample LS test pts
    eps = 1.0 if schedule=='PRP' else opts.eps
    ls_samples, ls_std_tot_dims = dnnlib.util.get_gen_samples(opts.data_dim, opts.dims_to_keep, device, \
                                                              shape=(opts.num_testpts, opts.data_dim), eps=eps)
    
    #sample LS boundary pts
    ls_bpts = dnnlib.util.get_all_boundary_pts(bpts_radii, ls_std_tot_dims, opts.data_dim, opts.num_bpts)
    
    #-------------------------------------------#
    #run gen direction
    #-------------------------------------------#
    
    #setup space variable for open3d mesh outputs fnames
    if opts.data_dim==3: 
        mesh_kwargs['space'] = 'ls' 
    
    #compute and save initial coverages in latent space
    dist.print0('*'*40)
    dist.print0('Calculating initial LS coverages ... ')
    dist.print0('*'*40)
    
    ls_init_coverages = calc_coverages(ls_samples.cpu().numpy(), ls_bpts, bpts_radii, opts.num_bpts, **mesh_kwargs)
    ls_init_coverages_fname = '{}_{}_{}lssamples_init_ls_coverages.txt'.format(data_name, schedule, opts.num_testpts)
    dnnlib.util.dump_json(save_dir, ls_init_coverages_fname, ls_init_coverages)
    
    #now run pfODE backwargs in time (gen)
    dist.print0('*'*40)
    dist.print0('Integrating pfODE in gen direction...')
    dist.print0('*'*40)
    
    net_data = np.concatenate([ls_samples.cpu().numpy(), ls_bpts], axis=0)
    net_data = torch.from_numpy(net_data).type(torch.float32).to(device)
    
    net_gen_results = sim_batch_net_ODE(net_data, net, int_mode='gen', n_iters=opts.num_steps, \
                                        h=opts.h, A0=opts.end_vars, save_freq=opts.save_freq)
    
    gen_res_fname = '{}_samples_bpts_{}_gen_{}niters_{}ss_{}gamma0_{}rho_{}A0.npz'.format(data_name, schedule, \
                                                                                          opts.num_steps, opts.h, net.gamma0, net.rho, \
                                                                                              opts.end_vars)
    np.savez(os.path.join(save_dir, gen_res_fname), tilde_xs=net_gen_results['tilde_xs'], xs=net_gen_results['xs'], \
             tilde_xs_es=net_gen_results['tilde_xs_es'], xs_es=net_gen_results['xs_es'], dxs=net_gen_results['dxs'], \
                 dxs_es=net_gen_results['dxs_es'], net_outs=net_gen_results['net_outs'], \
                     unscaled_scores=net_gen_results['unscaled_scores'])
        
    #compute data space end coverages 
    #these are initial coverages for "inflation" direction 
    
    dist.print0('*'*40)
    dist.print0('Calculating end of generation coverages')
    dist.print0('*'*40)
    
    ds_samples = np.array(net_gen_results['xs'])[-1, 0:opts.num_testpts, :]
    ds_bpts = np.array(net_gen_results['xs'])[-1, opts.num_testpts:, :]
    
    #reset space variable for open3d mesh outputs fnames
    if opts.data_dim==3:
        mesh_kwargs['space'] = 'ds' 
    ds_end_coverages = calc_coverages(ds_samples, ds_bpts, bpts_radii, opts.num_bpts, **mesh_kwargs)
    ds_end_coverages_fname = '{}_{}_{}lssamples_ds_coverages.txt'.format(data_name, schedule, opts.num_testpts)
    dnnlib.util.dump_json(save_dir, ds_end_coverages_fname, ds_end_coverages)
    
    #----------------------------------------------#
    #run inflation/melt direction 
    #----------------------------------------------#
    
    dist.print0('*'*40)
    dist.print0('Integrating pfODE in inflation direction...')
    dist.print0('*'*40)
    
    #simulate and save pfODE in melt/inflation direction
    end_gen_samples_bpts = torch.from_numpy(np.array(net_gen_results['tilde_xs'])[-1, :, :]).type(torch.float32).to(device)
    net_melt_results = sim_batch_net_ODE(end_gen_samples_bpts, net, int_mode='melt', \
                                              n_iters=opts.num_steps, h=opts.h, \
                                                  A0=opts.end_vars, save_freq=opts.save_freq)
    melt_res_fname = '{}_samples_bpts_{}_melt_{}niters_{}ss_{}gamma0_{}rho_{}A0.npz'.format(data_name, schedule, \
                                                                                          opts.num_steps, opts.h, net.gamma0, net.rho, \
                                                                                              opts.end_vars)    
    np.savez(os.path.join(save_dir, melt_res_fname), tilde_xs=net_melt_results['tilde_xs'], xs=net_melt_results['xs'], \
             tilde_xs_es=net_melt_results['tilde_xs_es'], xs_es=net_melt_results['xs_es'], dxs=net_melt_results['dxs'], \
                 dxs_es=net_melt_results['dxs_es'], net_outs=net_melt_results['net_outs'], \
                     unscaled_scores=net_melt_results['unscaled_scores'])
        
    #compute and save coverages for end of rountrip
    dist.print0('*'*40)
    dist.print0('Calculating end of inflation coverages')
    dist.print0('*'*40)    
    
    ls_rdtrp_samples = np.array(net_melt_results['xs'])[-1, 0:opts.num_testpts, :]
    ls_rdtrp_bpts = np.array(net_melt_results['xs'])[-1, opts.num_testpts:, :]
    
    #reset space variable for open3d mesh outputs fnames
    if opts.data_dim==3:
        mesh_kwargs['space'] = 'ls_end_rdtrp' 
    ls_end_rdtrp_coverages = calc_coverages(ls_rdtrp_samples, ls_rdtrp_bpts, bpts_radii, opts.num_bpts, **mesh_kwargs)
    ls_end_rdtrp_coverages_fname = '{}_{}_{}lssamples_ls_end_rdtrp_coverages.txt'.format(data_name, schedule, opts.num_testpts)
    dnnlib.util.dump_json(save_dir, ls_end_rdtrp_coverages_fname, ls_end_rdtrp_coverages)
    
    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')
      
    return 

#----------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------    
    
    
    
    