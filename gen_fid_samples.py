#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run generation only for large number of samples. 

This will be used when calculating FID scores for the different models.

This follows mostly same layout as Karras' original gemerate.py BUT 
using IFs ODE sim methods. 

"""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import gc 
from sklearn.covariance import EmpiricalCovariance
from torch_utils import distributed as dist
from pfODE_sim.ODE import sim_batch_net_ODE

#----------------------------------------------------------#
#From EDM repo 

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------#

#get args with click 

#----------------------------------------------------------------------------# 
@click.command()
@click.option('--save_dir',                help='Where to save the output results', metavar='DIR',                                                            type=str, required=True)
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                                                                type=str, required=True)
@click.option('--disc',                    help='Discretization to use when simulating ODE', metavar='ifs|vp_ode',                                            type=click.Choice(['ifs', 'vp_ode']), default='vp_ode', show_default=True)
@click.option('--vpode_disc_eps',          help='Epsilon_s param for vp_ode discretization (if using this option).', metavar='FLOAT',                         type=float, default=1e-2, show_default=True)
@click.option('--solver',                  help='Solver to use when simulating ODE', metavar='euler|heun',                                                    type=click.Choice(['euler', 'heun']), default='heun', show_default=True)
@click.option('--bs',                      help='Maximum batch size', metavar='INT',                                                                          type=click.IntRange(min=1), default=800, show_default=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                                                               type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                                                                   is_flag=True)

@click.option('--n_iters',                 help='Number of ODE integration steps', metavar='INT',                                                             type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--end_time',                help='End melt/inflation time for ODE integration', metavar='FLOAT',                                               type=float, default=15.01, show_default=True)
@click.option('--end_vars',                help='End of melt vars per dim (A0)', metavar='FLOAT',                                                             type=float, default=1., show_default=True)
@click.option('--save_freq',               help='How often to save ODE sim steps.', metavar='INT',                                                            type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--dims_to_keep',            help='Number of original data dims to keep.', metavar='INT',                                                       type=click.IntRange(min=1), default=3072, show_default=True)
@click.option('--device_name',             help='Name of device we wish to run simulation on.', metavar='STR',                                                type=str, default='cuda', show_default=True)
@click.option('--img_size',                help='Size of imgs being processed.', metavar='INT',                                                               type=int, default=32, show_default=True)
@click.option('--img_ch',                  help='Number of channels in imgs being processed.', metavar='INT',                                                 type=int, default=3, show_default=True)
@click.option('--gen_source',              help='Method used to construct gen samples', metavar='diag|empirical',                                             type=click.Choice(['diag', 'empirical']), default='diag', show_default=True)
@click.option('--eps',                     help='Variance for compressed dims during PRR gen. (if gen from diag)', metavar='FLOAT',                           type=float, default=1., show_default=True)
@click.option('--prev_melt',               help='Path to previous melt results to be used for gen (if from empirical covariance)', metavar='STR',             type=str, default='', show_default=True)

#------------------------------------------------------------------------------#
def main(save_dir, network_pkl, subdirs, seeds, bs, **kwargs):
    
    """
    Runs generation for given set of seeds using specified 
    network and disc options and saves final images. 
    
    These can later be used to compute FID scores for given 
    network/model.
    """
    
    #get all other args and pass it to general opts dict
    opts = dnnlib.util.EasyDict(kwargs)    
    
    #init dist mode
    dist.init()
    
    #set up device
    device = torch.device(opts.device_name)    
    
    #get total dims, check that dim args make sense ... 
    flat_dims = opts.img_ch*(opts.img_size**2)
    assert flat_dims >= opts.dims_to_keep, 'Dims to keep needs to be less or equal to total dims!'
    
    #get std vals we will use to scale std MVN samples appropriately 
    if opts.gen_source=='empirical':
        assert opts.prev_melt!='', 'To run gen with empirical covariance samples, a previous melt is needed!'
        
        melt_res = np.load(opts.prev_melt)
        end_melt = melt_res['xs_es'][-1, :, :]
        del melt_res
        gc.collect()
        
        cov=EmpiricalCovariance().fit(end_melt)
        emp_cov_diag = np.diagonal(cov.covariance_)
        sample_scales = torch.from_numpy(np.sqrt(emp_cov_diag)).type(torch.float32).to(device)
        
    elif opts.gen_source == 'diag':
        if flat_dims == opts.dims_to_keep:
            #if PRP, no re-scaling is needed. Model ends on std MVN latent.
            sample_scales = torch.ones(flat_dims).type(torch.float32).to(device) 
        else: 
            cov=np.ones(opts.dims_to_keep)
            cov = np.append(cov, (np.ones(flat_dims - opts.dims_to_keep)*opts.eps))
            sample_scales = torch.from_numpy(np.sqrt(cov)).type(torch.float32).to(device)
    
    else: 
        raise NotADirectoryError('Gen source not implemented!')
    

    #load network 
    if dist.get_rank() != 0:
        torch.distributed.barrier()    

    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)
    
    if dist.get_rank() == 0:
        torch.distributed.barrier()    
    
      
    #set up num_batches and seeds per batch 
    num_batches = ((len(seeds) - 1) // (bs * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    
    #now loop though batches//seeds and get samples 
    dist.print0(f'Generating {len(seeds)} images to "{save_dir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents 
        rnd = StackedRandomGenerator(device, batch_seeds)
        samples = rnd.randn([batch_size, opts.img_ch, opts.img_size, opts.img_size], device=device) #std MVN samples 
        #scale these appropriately! 
        samples = samples.reshape(-1, flat_dims) * sample_scales[None, :]
        

        # Generate images.
        images = sim_batch_net_ODE(samples, net, device, shape=[batch_size, opts.img_ch, opts.img_size, opts.img_size], \
                                     int_mode='gen', n_iters=opts.n_iters, end_time=opts.end_time, \
                                         A0=opts.end_vars, save_freq=opts.save_freq, disc=opts.disc, solver=opts.solver, \
                                             eps=opts.vpode_disc_eps, endsim_imgs_only=True) 

        # Save images.
        images = np.reshape(images, (batch_size, opts.img_ch, opts.img_size, opts.img_size)) * 127.5 + 128
        images = np.clip(images, 0, 255).astype(np.uint8)
        images = np.transpose(images, axes=[0, 2, 3, 1])
        
        for seed, image in zip(batch_seeds, images):
            image_dir = os.path.join(save_dir, f'{seed-seed%1000:06d}') if subdirs else save_dir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image.shape[2] == 1:
                PIL.Image.fromarray(image[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image, 'RGB').save(image_path)


    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------#

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------#