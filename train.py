# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Train diffusion-based generative model using inflationary flows 
pre-conditioning and loss. 
"""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop

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
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                                                                                                            type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                                                                                                              type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',                                                                                                       type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',                                                                                                   type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm|ifs',                                                                                            type=click.Choice(['vp', 've', 'edm', 'ifs']), default='ifs', show_default=True)
@click.option('--data_dim',      help='Dimensionality (C*H*W) of data.', metavar='INT',                                                                                                      type=int, required=True)
@click.option('--dims_to_keep',  help='Number of dimensions to keep', metavar='INT',                                                                                                         type=int, required=True)
@click.option('--g_type',        help='Type of g construction to use.', metavar='orig|constant_inflation_gap',                                                                               type=click.Choice(['orig', 'constant_inflation_gap']), default='orig', show_default=True)
@click.option('--g_rescaling',   help='Global rescaling for g tensor. Defaults to NO re-scaling(1)', metavar='FLOAT',                                                                        type=float, default=1., show_default=True) 
@click.option('--inflation_gap', help='Inflation gap to use when constructing PRR schedule g, if using constant inflation gap option. Defaults to 0.02', metavar='FLOAT',                    type=float, default=0.02, show_default=True) 
@click.option('--gamma0',        help='Initial melting kernel width', metavar='FLOAT',                                                                                                       type=click.FloatRange(min=5e-4, min_open=False), default=5e-4, show_default=True)
@click.option('--rho',           help='Constant for exponential growth/inflation.', metavar='FLOAT',                                                                                         type=click.FloatRange(min=1.0, min_open=False), default=1.0, show_default=True)
@click.option('--tmin',          help='Smallest melt time/sigma to sample', metavar='FLOAT',                                                                                                 type=click.FloatRange(min=1e-7, min_open=False), default=1e-7, show_default=True)
@click.option('--tmax',          help='Largest melt time/sigma to sample', metavar='FLOAT',                                                                                                  type=click.FloatRange(min=1.0, min_open=False), default=15.01, show_default=True)
@click.option('--space',         help='Space Net should be trained on. Defaults to image space (IS)', metavar='ES|IS',                                                                       type=click.Choice(['ES', 'IS']), default='IS', show_default=True)
@click.option('--t_sampling',    help='How to sample ts to be used during training. Defaults to uniform.', metavar='uniform|normal',                                                         type=click.Choice(['uniform', 'normal']), default='uniform', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                                                                                                                   type=click.FloatRange(min=0, min_open=True), default=275, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                                                                                                                     type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                                                                                                             type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--batch_pca',     help='Batch size used to compute data pca. Must be >= data_dim', metavar='INT',                                                                             type=int, default=50000, show_default=True)
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',                                                                                                type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST',                                                                                          type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                                                                                                                      type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                                                                                                                       type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                                                                                                                type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                                                                                                                type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                                                                                                              type=bool, default=False, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',                                                                                                     type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                                                                                                                       type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                                                                                                           type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                                                                                                         type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                                                                                                          type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',                                                                                                 type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                                                                                                            is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                                                                                                         type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',                                                                                                        type=click.IntRange(min=1), default=125, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                                                                                                            type=click.IntRange(min=1), default=125, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',                                                                                                       type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',                                                                                            type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',                                                                                                   type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                                                                                                                     is_flag=True)

def main(**kwargs):
    """
    Train diffusion-based generative model using inflationary flows 
    pre-conditioning and loss".

    Examples:

    \b
    # torchrun --rdzv_endpoint=0.0.0.0:29501 --outdir=out --data=datasets/cifar10-32x32.zip  \
    --data_dim=3072 --dims_to_keep=3072 --rho=2 --batch=512 
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    # Preconditioning & loss function.
    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'training.networks.VPPrecond'
        c.loss_kwargs.class_name = 'training.loss.VPLoss'
    elif opts.precond == 've':
        c.network_kwargs.class_name = 'training.networks.VEPrecond'
        c.loss_kwargs.class_name = 'training.loss.VELoss'
    elif opts.precond == 'edm':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'
    else: 
        assert opts.precond=='ifs'
        #add all args to our net and loss
        #g, W, data_eigs are added later on (inside training_loop) to avoud issues when priting json dump.
        c.network_kwargs.update(class_name='training.networks.IFsPreCond', space=opts.space, gamma0=opts.gamma0, rho=opts.rho)
        c.loss_kwargs.update(class_name='training.loss.IFsLoss', space=opts.space, t_min=opts.tmin, t_max=opts.tmax, gamma0=opts.gamma0, rho=opts.rho, t_sampling=opts.t_sampling)
        
    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)
    #Add options to construct g and get eigs, W 
    c.update(data_dim=opts.data_dim)
    c.update(dims_to_keep=opts.dims_to_keep)
    c.update(g_type=opts.g_type)
    c.update(inflation_gap=opts.inflation_gap)
    c.update(g_rescaling=opts.g_rescaling)
    c.update(batch_pca=opts.batch_pca)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
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
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
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

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
