#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Train.py version to be used for training net on toy datasets. 

Script allows either ToyConvUNet and ToySongUNet as arch options -> 
all paper toy exps used ToyConvUNet. 

Script also allows 3 different pre-conditioning schemes: 
    1) inflationary flows (ifs) --> used in all toy experiments in paper
    2) edm --> toy implementation in image space (IS) of original EDM precond. 
    NOT used for any paper exps --> left for sanity check (if desired)
    3) edm_es --> toy implementation in eigen space (ES) of original EDM precond. 
    NOT used for any paper exps --> left for sanity check (if desired)
    
"""

import os
import re 
import json #do we need this truly? 
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import toy_training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

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

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                               type=str, required=True)
@click.option('--data_name',     help='Name of toy dset to use', metavar='STR',                                 type=str, required=True)
@click.option('--data_dim',      help='Dimensionality of toy data.', metavar='INT',                             type=int, required=True)
@click.option('--augment_to',    help='Dimension we should augment our original toy dset to.', metavar='INT',   type=int, default=0, show_default=True)
@click.option('--img_res',       help='Dimensionality UpSampled data fed to ConvNets.', metavar='INT',          type=int, default=8, show_default=True)
@click.option('--dims_to_keep',  help='Number of dimensions to keep', metavar='INT',                            type=int, required=True)
@click.option('--gamma0',        help='Initial melting kernel width', metavar='FLOAT',                          type=click.FloatRange(min=5e-4, min_open=False), default=5e-4, show_default=True)
@click.option('--rho',           help='Exponent constant factor.', metavar='FLOAT',                             type=click.FloatRange(min=1.0, min_open=False), default=1.0, show_default=True)
@click.option('--tmin',          help='Smallest melt time/sigma to consider', metavar='FLOAT',                  type=click.FloatRange(min=1e-7, min_open=False), default=1e-7, show_default=True)
@click.option('--tmax',          help='Largest melt time/sigma to consider', metavar='FLOAT',                   type=click.FloatRange(min=1.0, min_open=False), default=15.01, show_default=True)
@click.option('--arch',          help='Network architecture to use.', metavar='ToyConvUNet|ToySongUNet',        type=click.Choice(['ToyConvUNet', 'ToySongUNet']), default='ToyConvUNet', show_default=True)
@click.option('--space',         help='Space Net should be trained on', metavar='ES|IS',                        type=click.Choice(['ES', 'IS']), default='ES', show_default=True)
@click.option('--precond',       help='PreConditioning to use for Net Training', metavar='ifs|edm|edm_es',      type=click.Choice(['ifs', 'edm', 'edm_es']), default='ifs', show_default=True)
@click.option('--g_rescaling',   help='Global re-scaling factor for g tensor.', metavar='FLOAT',                type=float, default=1., show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                                      type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                                type=click.IntRange(min=1))
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST',             type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                                         type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                                          type=click.FloatRange(min=0), default=0.5, show_default=True)

# Performance-related.
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                                         type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                             type=bool, default=True, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',                   type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                              is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                           type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',                          type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                              type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',                         type=int)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',                     type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                                       is_flag=True)

def main(**kwargs):
    """
    Similar to train.py, only simpler 
    and for toy datasets... 
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    
    #setup device, g, eigs, pca_trf
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    
    #setup data_dim to match augmented dims (if using this option)
    working_data_dim = opts.data_dim if opts.augment_to==0 else opts.augment_to
    
    #if doing ifs precond, get g, eigs, and pca_trf
    if opts.precond in ['ifs', 'edm_es']:
        dset_large_smpl = dnnlib.util.get_toy_dset(opts.data_name, 70000, augment_to=opts.augment_to)
        data_eigs, pca_trf, W = dnnlib.util.get_eigenvals_basis(dset_large_smpl[0], n_comp=working_data_dim) 
        del dset_large_smpl #save mem 
        if opts.precond == 'ifs':
            #get g 
            g = dnnlib.util.get_g(working_data_dim, opts.dims_to_keep, device)
            g *= opts.g_rescaling #apply re-scaling factor (default is ==1, i.e., no rescaling)
    
    
    # Initialize config dict.
    # Set up simplified dset, network, loss, optimizer kwargs dicts
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(dset_name = opts.data_name, orig_data_dim=opts.data_dim, working_data_dim=working_data_dim, augment_to=opts.augment_to)
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)
    
    assert opts.arch in ['ToyConvUNet', 'ToySongUNet']
    
    if opts.precond=='ifs':
        c.loss_kwargs = dnnlib.EasyDict(t_min=opts.tmin, t_max=opts.tmax, rho=opts.rho, gamma0=opts.gamma0, space=opts.space, class_name='training.loss.IFsToyLoss')
        if opts.arch == "ToyConvUNet": 
            c.network_kwargs = dnnlib.EasyDict(model_type=opts.arch, channels=[32, 64, 128, 256], fc_embed_dim=2, conv_embed_dim=256, \
                                               data_dim=working_data_dim, out_ch=1, gamma0=opts.gamma0, rho=opts.rho, space=opts.space, class_name='training.networks.IFsToyPreCond') 
        else:
            c.network_kwargs = dnnlib.EasyDict(model_type=opts.arch, data_dim=working_data_dim, img_res = opts.img_res, out_ch=1, gamma0=opts.gamma0, \
                                               rho=opts.rho, space=opts.space, class_name='training.networks.IFsToyPreCond')
    elif opts.precond=='edm':
        c.loss_kwargs = dnnlib.EasyDict(P_mean=-1.2, P_std=1.2, sigma_data=1.0, class_name='training.loss.EDMToyLoss')
        if opts.arch=='ToyConvUNet':
            c.network_kwargs = dnnlib.EasyDict(model_type=opts.arch, channels=[32, 64, 128, 256], fc_embed_dim=2, conv_embed_dim=256, \
                                               data_dim=working_data_dim, out_ch=1, class_name='training.networks.EDMToyPreCond')
        else:
            c.network_kwargs = dnnlib.EasyDict(model_type=opts.arch, data_dim=working_data_dim, img_res=opts.img_res, out_ch=1, class_name='training.networks.EDMToyPreCond')
    else:
        #i.e., we are training toys on edm_es precond
        c.loss_kwargs = dnnlib.EasyDict(P_mean=-1.2, P_std=1.2, class_name='training.loss.EDMToyESLoss')
        if opts.arch=='ToyConvUNet':
            c.network_kwargs = dnnlib.EasyDict(model_type=opts.arch, channels=[32, 64, 128, 256], fc_embed_dim=2, conv_embed_dim=256, \
                                               data_dim=working_data_dim, out_ch=1, class_name='training.networks.EDMToyESPreCond')
        else:
            c.network_kwargs = dnnlib.EasyDict(model_type=opts.arch, data_dim=working_data_dim, img_res=opts.img_res, out_ch=1, class_name='training.networks.EDMToyESPreCond')        

    
    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)
    

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=device)
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)


    # Resume learning
    if opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    if opts.precond=='ifs': 
        schedule_type_str = 'prp' if working_data_dim == opts.dims_to_keep else 'prr' 
        desc = f'{opts.data_name}-{schedule_type_str}-uncond-{opts.arch}-{opts.precond}PreCond-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-fp32'
    else: 
        desc = f'{opts.data_name}-uncond-{opts.arch}-{opts.precond}PreCond-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-fp32'
        
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset name:            {c.dataset_kwargs.dset_name}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0()
        
    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)
    
    #add pca_trf and ndarray keys here to avoid issues with json dumping...
    if opts.precond in ['ifs', 'edm_es']:
        c.dataset_kwargs.update(pca_trf=pca_trf)
        if opts.precond == 'ifs':
            c.network_kwargs.update(data_eigs=data_eigs, g=g, W=W) 
            c.loss_kwargs.update(g=g, W=W)
        else:
            c.network_kwargs.update(data_eigs=data_eigs)

    # Train.
    toy_training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
