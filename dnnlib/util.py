# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Miscellaneous utility classes and functions."""

import ctypes
import fnmatch
import importlib
import inspect
import numpy as np
import os
import shutil
import sys
import types
import io
import json
import open3d as o3d
import alphashape
from shapely.geometry import Point
import pickle
import re
import requests
import html
import hashlib
import glob
import tempfile
import urllib
import urllib.request
import uuid
import torch 
from numpy import trapz
from torch.utils.data.dataset import Dataset
from sklearn.decomposition import PCA
from sklearn import datasets as sklrn_dsets
from scipy.stats import multivariate_normal

from distutils.util import strtobool
from typing import Any, List, Tuple, Union, Optional


#------------------------------------------------------------------------------------------#
# Utils for IFs 
#------------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------------
#Toy data utils 

def _orthogonalize_cols(mat):
    """
    Applies the Gram-Schimdt process to orthogonalize
    cols of given matrix 
    """
    #assign first col to u1
    u1 = mat[:, 0] 
    #set up list with our new orthog vectors
    us = [u1]
    for c in range(1, mat.shape[1]):
        curr_u = mat[:, c]
        for i in range(len(us)):
            curr_u -= (np.dot(curr_u, us[i])/np.dot(us[i], us[i]))*us[i]
        us.append(curr_u)
    return np.array(us).T

def get_orthonormal_rp_matrix(n,d, seed=42):
    """
    Constructs an orthonormal random projection 
    matrix.
    Args:
        n:int: Number of rows.
        d:int. Number of cols.
        seed: random seed to use when constructing rp 
        matrix.
    """
    np.random.seed(seed=seed)
    rp_mat = np.random.randn(n,d)
    rp_mat_o = _orthogonalize_cols(rp_mat)
    rp_mat_on = rp_mat_o / np.linalg.norm(rp_mat_o, ord=2, axis=0)[None, :]
    return rp_mat_on

def _make_sine_data(dset_option, n):
    """
    
    Construct 2D sine dataset for 
    toy experiments. 
    
    Can construct either 
    1) sine 1 data (0-1, unscaled), or
    2) sine2 data (-1 - 1), scaled. We use
    sine2 in paper results.
    
    Args:
    -----
    dset_option: str indicating which
    sine data (1 or 2) to use.
    n: number of dset samples we wish 
    to create. 
    
    Returns list containing array of data pts [n, 2]
    and array of labels [n, np zeros]. 
    
    """
    if dset_option =='1':
        xs = np.random.uniform(low=0., high=1., size=n)
        ys = np.sin(4*np.pi*xs) 
        noise = np.random.normal(loc=0, scale=0.8, size=n) 
        ys_noise = ys + noise
        dset = np.array([xs, ys_noise]).T 
    else: 
        xs = np.random.uniform(low=-1., high=1., size=n)
        ys = np.sin(4*np.pi*xs)/4
        noise = np.random.normal(loc=0, scale=0.15, size=n) 
        ys_noise = ys + noise
        dset = np.array([xs, ys_noise]).T   
    return dset 

def get_toy_dset(dset_name, n=20000, augment_to=0):
    """
    
    Fetches dset and labels for a couple 
    of 2D and 3D toy cases, namely: 
        1) sine; 
        2) circles; 
        3) moons; 
        4) swirl;
        5) alt_swirl;
        5) s_curve;
        6) alt_s_curve.
    
    Alt_swirl and alt_scurve refer to datasets with 
    y dimension scaled to 0.5, instead of 1.
    
    Args:
    -----
    dset_name: str. Name of dset we wish to 
    generate. Can be any of the options listed above.
    
    n: int. Number of samples to generate for each dset. 
    
    augment_to: int. If not None, corresponds to dimension
    we will augmented original toy dset to using an 
    orthonormal random projection matrix.
    
    """
    if dset_name[:-1] == 'sine':
        dset_pts = _make_sine_data(dset_name[-1], n)
        dset_pts = (dset_pts - np.mean(dset_pts, axis=0))/np.std(dset_pts, axis=0)
        dset_labels = np.zeros(dset_pts.shape[0])
        dset = [dset_pts, dset_labels]
    elif dset_name == 'circles':
        dset = sklrn_dsets.make_circles(n_samples=n, factor=0.5, noise=0.08)
        circles_zscld = (dset[0] - np.mean(dset[0], axis=0))/np.std(dset[0], axis=0)
        dset = [circles_zscld, dset[1]]
    elif dset_name =='moons':
        dset = sklrn_dsets.make_moons(n_samples=n, noise=0.08)
        moons_zscld = (dset[0] - np.mean(dset[0], axis=0))/np.std(dset[0], axis=0)
        dset=[moons_zscld, dset[1]]
    elif dset_name == 'swirl':
        dset = sklrn_dsets.make_swiss_roll(n_samples=n, noise=1.25)
        swiss_roll_pts_zscld = (dset[0] - np.mean(dset[0], axis=0))/np.std(dset[0], axis=0)
        dset = [swiss_roll_pts_zscld, dset[1]]
    elif dset_name == 'alt_swirl':
        dset = sklrn_dsets.make_swiss_roll(n_samples=n, noise=1.25)
        swiss_roll_pts_zscld = (dset[0] - np.mean(dset[0], axis=0))/np.std(dset[0], axis=0)
        ys = np.random.randn(dset[0].shape[0]) * 0.5
        alt_swirl = np.concatenate([np.expand_dims(dset[0][:, 0], axis=-1), np.expand_dims(ys, axis=-1),\
                           np.expand_dims(dset[0][:, 2], axis=-1)], axis=1)
        dset = [alt_swirl, dset[1]]
    elif dset_name == 's_curve':
        dset = sklrn_dsets.make_s_curve(n_samples=n, noise=0.2)
        scurve_pts_zscld = (dset[0] - np.mean(dset[0], axis=0))/np.std(dset[0], axis=0)
        dset = [scurve_pts_zscld, dset[1]]
    elif dset_name == 'alt_s_curve':
        dset = sklrn_dsets.make_s_curve(n_samples=n, noise=0.2)
        scurve_pts_zscld = (dset[0] - np.mean(dset[0], axis=0))/np.std(dset[0], axis=0)
        ys = np.random.randn(dset[0].shape[0]) * 0.5
        alt_scurve = np.concatenate([np.expand_dims(dset[0][:, 0], axis=-1), np.expand_dims(ys, axis=-1),\
                           np.expand_dims(dset[0][:, 2], axis=-1)], axis=1) 
        dset = [alt_scurve, dset[1]]        
    else:
        raise NotImplementedError('Dset name chosen is not implemented!')
    if augment_to != 0: 
        #using random proj matrix to augment our original data 
        rp_matrix = get_orthonormal_rp_matrix(augment_to, dset[0].shape[1])
        aug_dset = np.squeeze(np.einsum('ij, bjk -> bik', rp_matrix, \
                                        np.expand_dims(dset[0], axis=-1)))
        dset = [aug_dset, dset[1]]
    return dset 

class ToyDset(Dataset):
    """
    Basic dset structure for toy data.
    """
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform=transform
        
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        pt_coords = self.data[idx, :]
        pt_label = self.labels[idx]
        if self.transform:
            pt_coords = self.transform(pt_coords)
        return pt_coords, pt_label

#------------------------------------------------------------------
#IFs ODE integration/schedule utils 


def get_disc_times(disc='ifs', end_time=15.01, n_iters=1501, int_mode='melt', eps=1e-2):
    """
    Gets times for 2 different disc options implemented namely original "ifs" disc
    (linearly spaced pts) vs. VP ODE disc (sees EDM paper table 1). 
    Main paper uses vp_ode discretization for all experiments.
    """
    if disc=='ifs':
        times = np.linspace(0, end_time, n_iters)
        if int_mode == 'gen':
            times = np.flip(times)
    elif disc=='vp_ode':
        times = [end_time + (i/(n_iters-1))*(eps-end_time) for i in range(0, n_iters)]
        if int_mode=='melt':
            times = np.flip(times)
    else: 
        raise NotImplementedError("Discretization options not implemented!")
    return times


def get_g(data_dim, dims_to_keep, device):
    """
    
    Computes g tensor to be used 
    when constructing gamma(t) and
    inflation kernel covariances.
    
    Args:
    ----
    data_dims: int. Original dimensionality
    of data.
    dims_to_keep:int. Number of dimensions 
    we wish to preserve as flow progresses.
    device: instance of torch.device class. 
    
    """
    if data_dim == dims_to_keep:
        #this is PRP case
        g = torch.zeros(data_dim).to(device)
    else: 
        #this is PRR case 
        g_pos = np.ones(dims_to_keep)
        g_neg = -(dims_to_keep/(data_dim - dims_to_keep))*np.ones((data_dim-dims_to_keep))
        g = np.append(g_pos, g_neg)
        g = torch.from_numpy(g).type(torch.float32).to(device)
    return g


def get_eigenvals_basis(X, n_comp=784):
    """
    Uses sklearn PCA method to obtain U_t matrix 
    containing eigenvectors of current covariance mat
    And its correspoding eigenvals.
    
    This can also be done using SVD
    (or even streaming SVD) as well.
    """
    pca = PCA(n_components=n_comp) 
    pca.fit(X)
    eigenvals = pca.explained_variance_
    U = pca.components_ #eigenvectors are rows of U 
    return eigenvals, pca, U.T #eigenvectors returned as cols of U

def get_gamma(g, ts, gamma0=5e-4, rho=1):
    """
    Method to compute gamma(t) during ODE simulation.
    Note that am treating ts as a scalar here since
    for melting/PG all samples take same t at each step.
    
    Args:
    -----
    g: torch.Tensor [dim]. Constant g tensor used for IFs 
    schedule.
    ts: float. Time pt we should compute gammas for.
    gamma0: float. Initial melting kernel width (assuming t0=0)
    rho: float. Constant for exponential growth of melting kernel.
    
    Returns: 
    ------
    gamma: torch.Tensor [dim]. Tensor containing values for 
    noise covariance diagonal at given ts time pt.
    """
    exp_term = (torch.ones(g.shape[0]).to(g.device) + g) * ts #D
    gamma = gamma0 * torch.exp(rho*exp_term) #D
    return gamma

def get_gamma_dot(g, ts, gamma0=5e-4, rho=1):
    """
    Method to compute derivative of gamma(t) 
    during ODE simulation.
    
    Note that am treating ts as a scalar here since
    for melting/PG all samples take same t at each step.
    
    Args:
    -----
    g: torch.Tensor [dim]. Constant g tensor used for IFs 
    schedule.
    ts: float. Time pt we should compute gammas for.
    gamma0: float. Initial melting kernel width (assuming t0=0)
    rho: float. Constant for exponential growth of melting kernel.
    
    Returns: 
    ------
    gamma_dot: torch.Tensor [dim]. Tensor containing values for 
    noise covariance diagonal at given ts time pt.
    """

    mult_term = rho*(torch.ones(g.shape[0]).to(g.device) + g) #D
    gamma_dot = mult_term * get_gamma(g, ts, gamma0=gamma0, rho=rho)#D
    return gamma_dot

def get_s(xi_star, g, t, A0=1, gamma0=5e-4, rho=1):
    """
    Method to compute s(t) (i.e., \alpha(t)) scaling 
    during ODE simulation.
    
    Note that am treating ts as a scalar here since
    for melting/PG all samples take same t at each step.
    
    Args:
    -----
    xi_star: float. Value for highest eigevalue of original 
    data.
    g: torch.Tensor [dim]. Constant g tensor used for IFs 
    schedules.
    t: float. Time pt we should compute s(t) for.
    A0: float. Variance we should achieve (per dimension)
    at end of melt.
    gamma0: float. Initial melting kernel width (assuming t0=0)
    rho: float. Constant for exponential growth of melting kernel.
    
    Returns: 
    ------
    s(t): float. Scaling factor to be used in ODE simulation.
    """
    g_star = torch.amax(g).cpu().numpy()
    gamma_gstar = gamma0 * np.exp(rho*(1+g_star)*t)
    den = np.sqrt(xi_star + gamma_gstar)
    s = A0 / den 
    return s 

def get_s_dot(xi_star, g, t, A0=1, gamma0=5e-4, rho=1):
    """
    Method to compute s_dot scaling derivative
    during ODE simulation.
    
    Note that am treating ts as a scalar here since
    for melting/PG all samples take same t at each step.
    
    Args:
    -----
    xi_star: float. Value for highest eigevalue of original 
    data.
    g: torch.Tensor [dim]. Constant g tensor used for IFs 
    schedules.
    t: float. Time pt we should compute s(t) for.
    A0: float. Variance we should achieve (per dimension)
    at end of melt.
    gamma0: float. Initial melting kernel width (assuming t0=0)
    rho: float. Constant for exponential growth of melting kernel.
    
    Returns: 
    ------
    s_dot: float. Scaling factor to be used in ODE simulation.
    """
    g_star = torch.amax(g).cpu().numpy()
    gamma_gstar = gamma0 * np.exp(rho*(1+g_star)*t)
    s_dot = -0.5*A0*rho*(1+g_star)*gamma_gstar*((xi_star + gamma_gstar)**(-1.5))
    return s_dot 

def get_dx_dt_params(time, **kwargs):
    """
    Computes args needed to calculate flow dx/dt for a given time pt.
    """
    s = get_s(kwargs.get('xi_star'), kwargs.get('g'), time, A0=kwargs.get('A0'), \
                               gamma0=kwargs.get('gamma0'), rho=kwargs.get('rho'))
        
    s_dot=get_s_dot(kwargs.get('xi_star'), kwargs.get('g'), time, A0=kwargs.get('A0'), \
                               gamma0=kwargs.get('gamma0'), rho=kwargs.get('rho'))
        
    gamma = get_gamma(kwargs.get('g'), time, \
                                       gamma0=kwargs.get('gamma0'), rho=kwargs.get('rho')) #dim
    gamma_inv = torch.reciprocal(gamma) #dim 
    gamma_dot = get_gamma_dot(kwargs.get('g'), time, \
                                       gamma0=kwargs.get('gamma0'), rho=kwargs.get('rho'))    
    return s, s_dot, gamma_inv, gamma_dot 

def get_gen_samples(data_dims, dims_to_keep, device, shape, eps=1e-10):
    """
    
    Samples from appropriate MVN corresponding to end of melt 
    in either PRP or PRR schedules. 
    
    Args: 
    ----
    data_dims: float. Total number of dimensions in original data.
    dims_to_keep: float. Total number of dimensions we would like to 
    preserve when inflating.
    device: torch.device instance.
    shape: tuple(bs, dim). Tuple containing shape for desired samples 
    to be generated. 
    eps: float. Small constant variance for dimensions we will be compressing.
    
    Returns: 
    -----
    samples: torch.Tensor [bs, dim]. Tensor containing appropriate MVN samples
    to be fed to code simulating ODE in backward (i.e., generative) direction.
    
    """
    #take samples from std MVN 
    #this is prp regime 
    samples = torch.randn(shape, dtype=torch.float32, device=device) #bs, C, H, W 
    std_tot_dims = torch.ones(data_dims).to(device)
    
    if data_dims > dims_to_keep: 
        #prr regime 
        var_dims_to_keep = np.ones(dims_to_keep)
        var_dims_to_compress = np.ones(data_dims - dims_to_keep)*eps 
        var_tot_dims = np.append(var_dims_to_keep, var_dims_to_compress)
        std_tot_dims = torch.from_numpy(np.sqrt(var_tot_dims)).type(torch.float32).to(device)
        samples = std_tot_dims[None, :] * samples.reshape(samples.shape[0], -1) 
    
    return samples.reshape(shape), std_tot_dims 


def get_mvn_samples(location, cov, size, random_state=42):
    """
    Samples from a mvn w/ given location
    and covariance.
    ---
    Args:
    location: np.array containing mean/loc 
    of our mvn.
    cov: np.array containing covariance
    matrix for our mvn
    size: int. Number of mvn samples we wish to 
    produce
    random_state: random seed for scipy.
    """
    samples = multivariate_normal.rvs(mean=location, cov=cov, \
                                      size=size, random_state=random_state)
    return samples

#---------------------------------------------------------------------------#
#Methods for toy CI Exps 
#---------------------------------------------------------------------------#

#----------------------------------------------------------------------------
#helpers for 2D alpha-shape and 3D mesh experiments

def dump_json(save_dir, fname, coverages):
    """
    Writes .txt file with nested dicts containing
    results of computed coverages.
    """
    with io.open(os.path.join(save_dir, fname), 'w', encoding='utf-8') as f: 
        f.write(json.dumps(coverages, ensure_ascii=False))
    return 

def print_coverage_table(coverage_dict):
    """
    Prints vals stored in coverage nested dicts.
    """
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format('Radius', 'Total_in',\
                                                      'Total_Out', 'Percent_In', 'Percent_Out'))
    for k,v in coverage_dict.items():
        vals_to_print = [] 
        for k2, v2 in v.items(): 
            vals_to_print.append(v2) 
        vals_to_print = np.round(np.array(vals_to_print), decimals=2)
        k = np.round(k, decimals=2)
        print("{:<10} {:<10} {:<10} {:<10} {:<10}".format(k, vals_to_print[0], \
                                              vals_to_print[1], vals_to_print[2], \
                                                  vals_to_print[3]))
    return 


def get_all_boundary_pts(bpts_radii, std_tot_dims, d, n, center=None):
    """
    Runs through a list of desired boundary/shell radii 
    and samples n boundary pts for each one of these.
    Can either take center=None which defaults to zero center
    or some other custom center.
    
    Args
    -------
    bpts_radii: list containing radii for boundaries we wish 
    to sample from. 
    std_tot_dims: np.array. Contains standard devs (per dimesion)
    for diagonal Gaussian latent space we wish to sample from. 
    d: int. Dimension of Gaussian latent space.
    n: int. Number of boundary pts to sample per boundary sphere/radius.
    center: np.array. Defaults to None -> np.zeros(d). 
    
    Returns
    -------
    np.array [len(bpts_radii), n, d]
    containing all desired boundary pt samples per each 
    radius/boundary.
    """
    if center is None: 
        center = np.zeros(d) #default to zero center 
    
    #pass any potential tensors to np
    if torch.is_tensor(std_tot_dims):
        std_tot_dims = std_tot_dims.cpu().numpy()
               
    all_bpts = []
    for r in bpts_radii:
        curr_bpts = sample_boundary_pts(center=center, radius=r*std_tot_dims, \
                                        d=d, n=n)
        all_bpts.append(curr_bpts)
    all_bpts = np.concatenate(all_bpts, axis=0)
    
    return all_bpts


def sample_boundary_pts(center, radius=np.array([1., 1.]), d=2, n=200):
    """
    Samples n, d-dimensional boundary pts uniformly from 
    a sphere of given radius and center coordinates. 
    """
    boundary_pts = np.random.randn(n, d)
    boundary_pts /= np.linalg.norm(boundary_pts, ord=2, axis=-1)[:, None]
    boundary_pts *= radius[None, :]
    boundary_pts += center[None, :]
    return boundary_pts


def compute_coverages_2Dalphashape(samples, bpts, bpts_radii, bpts_per_radii, **kwargs):
    """
    For each set of bpts defining a bounding sphere, 
    fit an alpha shape to these, then query whether samples
    lie inside or outside of this alpha-shape and report counts and percentiles.
    
    Args
    ------
    samples: np.array [n, d]. Array with test pts we wish to eval coverages for.
    bpts: np.array[len(bpts_radii), n', d]. Array with all sampled bpts per each 
          boundary.
    bpts_radii: list containing radii for boundaries we wish 
    to sample from. 
    bpts_per_radii: int. Number of boundary pts sampled per each boundary.
    
    
    Returns: 
    --------
    nested dictionary containing total counts and percent coverages for each
        boudnary/sphere. 
    Also explicitly prints these results as a table. 
    """
    bpts_idxs = np.arange(0, len(bpts_radii)*bpts_per_radii, bpts_per_radii)
    all_coverages = {}
    for i in range(bpts_idxs.shape[0]):
        curr_bpts = bpts[bpts_idxs[i]:(bpts_idxs[i]+bpts_per_radii), :]
        curr_alphashape = alphashape.alphashape(curr_bpts, 0.005)
        curr_pts_in = 0
        curr_pts_out = 0
        for pt in range(samples.shape[0]):
            curr_pt = Point(samples[pt, 0], samples[pt, 1])
            curr_decision = curr_alphashape.contains(curr_pt)
            if curr_decision==True: 
                curr_pts_in+=1
            else:
                curr_pts_out+=1
        assert (curr_pts_in+curr_pts_out)==samples.shape[0], 'Total in and out MUST add to total samples!'
        curr_pin = (curr_pts_in/samples.shape[0])*100
        curr_pout = (curr_pts_out/samples.shape[0])*100
        curr_coverages = {'total_in':curr_pts_in, 'total_out':curr_pts_out, 
                         'percent_in':curr_pin, 'percent_out':curr_pout}
        all_coverages[float(bpts_radii[int(i)])] = curr_coverages
    
    print_coverage_table(all_coverages)
    return all_coverages

def compute_coverages_3Dmesh(samples, bpts, bpts_radii, bpts_per_radius, **kwargs):
    """
    Takes each set of bpts (for a given radius), fits a point-cloud to it, then
    computes triangular mesh for this point-cloud and uses this mesh to 
    query whether or not each one of samples given lies 
    inside or outside of mesh.     
    
    Args
    ------
    samples: np.array [n, d]. Array with test pts we wish to eval coverages for.
    bpts: np.array[len(bpts_radii), n', d]. Array with all sampled bpts per each 
          boundary.
    bpts_radii: list containing radii for boundaries we wish 
    to sample from. 
    bpts_per_radii: int. Number of boundary pts sampled per each boundary.
    
    
    Returns: 
    --------
    nested dictionary containing total counts and percent coverages for each
        boudnary/sphere. 
    Also explicitly prints these results as a table.     
    Saves fitted 3D alpha meshes for each radius given to a separate directory. 
    """
    #get schedule, space, schedule info 
    #to construct subdirs for fitted meshes 
    schedule = kwargs.get('schedule')
    space=kwargs.get('space')
    save_dir = kwargs.get('save_dir')
    
    #construct our idxs
    idxs = np.arange(0, len(bpts_radii)*bpts_per_radius, bpts_per_radius)
    #pass ls_samples to o3d tensor 
    all_query_pts = o3d.core.Tensor([samples], dtype=o3d.core.Dtype.Float32)
    #init data coverages dict
    data_coverages = {}
    for i in range(idxs.shape[0]):
        #pass bpts to point cloud 
        curr_bpts = bpts[idxs[i]:idxs[i]+bpts_per_radius, :]
        curr_pcd = o3d.geometry.PointCloud()
        curr_pcd.points = o3d.utility.Vector3dVector(curr_bpts)
        #construct triangular mesh from point cloud
        curr_tetra_mesh, curr_pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(curr_pcd)
        curr_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd=curr_pcd, \
                                                                     alpha=10, \
                                                                     tetra_mesh=curr_tetra_mesh, \
                                                                     pt_map=curr_pt_map)
        curr_mesh.compute_vertex_normals()
        #check that mesh is watertight 
        #this is needed to test occupancy of points!
        assert curr_mesh.is_watertight()==True, 'Mesh needs to be watertight!'
        
        #save mesh info to a .ply file
        curr_fname = "{}_{}_mesh_{}radius_bpts.ply".format(schedule, space, bpts_radii[int(i)])
        curr_out_path = os.path.join(save_dir, '{}_{}_meshes'.format(schedule, space), curr_fname)
        if not os.path.exists(os.path.join(save_dir, '{}_{}_meshes'.format(schedule, space))):
            os.makedirs(os.path.join(save_dir, '{}_{}_meshes'.format(schedule, space)))
        o3d.io.write_triangle_mesh(curr_out_path, curr_mesh)
        
        #construct a RaycastingScene, add our mesh to it
        curr_scene = o3d.t.geometry.RaycastingScene()
        curr_t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(curr_mesh)
        _ = curr_scene.add_triangles(curr_t_mesh)
        #compute ocupancy
        curr_occupancy = curr_scene.compute_occupancy(all_query_pts)
        #now compute total in, out and percentiles 
        curr_total_in = np.sum(curr_occupancy.numpy())
        curr_total_out = samples.shape[0] - curr_total_in
        curr_percent_in, curr_percent_out = (curr_total_in/samples.shape[0])*100, \
        (curr_total_out/samples.shape[0])*100
        curr_res_dict = {'total_in':int(curr_total_in), 'total_out':int(curr_total_out),
                    'percent_in':float(curr_percent_in), 'percent_out':float(curr_percent_out)} 
        data_coverages[float(bpts_radii[int(i)])] = curr_res_dict
    
    print_coverage_table(data_coverages)
    return data_coverages



#------------------------------------------------------------------------------#

# Util classes (EDM repo)

#-------------------------------------------------------------------------------#


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: Optional[str] = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: Union[str, bytes]) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None


# Cache directories
# ------------------------------------------------------------------------------------------#

_dnnlib_cache_dir = None

def set_cache_dir(path: str) -> None:
    global _dnnlib_cache_dir
    _dnnlib_cache_dir = path

def make_cache_dir_path(*paths: str) -> str:
    if _dnnlib_cache_dir is not None:
        return os.path.join(_dnnlib_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)

# Small util functions
# ------------------------------------------------------------------------------------------#


def format_time(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)


def format_time_brief(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m".format(s // (60 * 60), (s // 60) % 60)
    else:
        return "{0}d {1:02}h".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24)


def ask_yes_no(question: str) -> bool:
    """Ask the user the question until the user inputs a valid answer."""
    while True:
        try:
            print("{0} [y/n]".format(question))
            return strtobool(input().lower())
        except ValueError:
            pass


def tuple_product(t: Tuple) -> Any:
    """Calculate the product of the tuple elements."""
    result = 1

    for v in t:
        result *= v

    return result


_str_to_ctype = {
    "uint8": ctypes.c_ubyte,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "int8": ctypes.c_byte,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double
}


def get_dtype_and_ctype(type_obj: Any) -> Tuple[np.dtype, Any]:
    """Given a type name string (or an object having a __name__ attribute), return matching Numpy and ctypes types that have the same size in bytes."""
    type_str = None

    if isinstance(type_obj, str):
        type_str = type_obj
    elif hasattr(type_obj, "__name__"):
        type_str = type_obj.__name__
    elif hasattr(type_obj, "name"):
        type_str = type_obj.name
    else:
        raise RuntimeError("Cannot infer type name from input")

    assert type_str in _str_to_ctype.keys()

    my_dtype = np.dtype(type_str)
    my_ctype = _str_to_ctype[type_str]

    assert my_dtype.itemsize == ctypes.sizeof(my_ctype)

    return my_dtype, my_ctype


def is_pickleable(obj: Any) -> bool:
    try:
        with io.BytesIO() as stream:
            pickle.dump(obj, stream)
        return True
    except:
        return False


# Functionality to import modules/objects by name, and call functions by name
# ------------------------------------------------------------------------------------------#

def get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    """Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed)."""

    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [(".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name) # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith("No module named '" + module_name + "'"):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverses the object name and returns the last (rightmost) python object."""
    if obj_name == '':
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_obj_by_name(name: str) -> Any:
    """Finds the python object with the given name."""
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    """Finds the python object with the given name and calls it as a function."""
    assert func_name is not None
    func_obj = get_obj_by_name(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args, class_name: str = None, **kwargs) -> Any:
    """Finds the python class with the given name and constructs it with the given arguments."""
    return call_func_by_name(*args, func_name=class_name, **kwargs)


def get_module_dir_by_obj_name(obj_name: str) -> str:
    """Get the directory path of the module containing the given object name."""
    module, _ = get_module_from_obj_name(obj_name)
    return os.path.dirname(inspect.getfile(module))


def is_top_level_function(obj: Any) -> bool:
    """Determine whether the given object is a top-level function, i.e., defined at module scope using 'def'."""
    return callable(obj) and obj.__name__ in sys.modules[obj.__module__].__dict__


def get_top_level_function_name(obj: Any) -> str:
    """Return the fully-qualified name of a top-level function."""
    assert is_top_level_function(obj)
    module = obj.__module__
    if module == '__main__':
        module = os.path.splitext(os.path.basename(sys.modules[module].__file__))[0]
    return module + "." + obj.__name__


# File system helpers
# ------------------------------------------------------------------------------------------#

def list_dir_recursively_with_ignore(dir_path: str, ignores: List[str] = None, add_base_to_relative: bool = False) -> List[Tuple[str, str]]:
    """List all files recursively in a given directory while ignoring given file and directory names.
    Returns list of tuples containing both absolute and relative paths."""
    assert os.path.isdir(dir_path)
    base_name = os.path.basename(os.path.normpath(dir_path))

    if ignores is None:
        ignores = []

    result = []

    for root, dirs, files in os.walk(dir_path, topdown=True):
        for ignore_ in ignores:
            dirs_to_remove = [d for d in dirs if fnmatch.fnmatch(d, ignore_)]

            # dirs need to be edited in-place
            for d in dirs_to_remove:
                dirs.remove(d)

            files = [f for f in files if not fnmatch.fnmatch(f, ignore_)]

        absolute_paths = [os.path.join(root, f) for f in files]
        relative_paths = [os.path.relpath(p, dir_path) for p in absolute_paths]

        if add_base_to_relative:
            relative_paths = [os.path.join(base_name, p) for p in relative_paths]

        assert len(absolute_paths) == len(relative_paths)
        result += zip(absolute_paths, relative_paths)

    return result


def copy_files_and_create_dirs(files: List[Tuple[str, str]]) -> None:
    """Takes in a list of tuples of (src, dst) paths and copies files.
    Will create all necessary directories."""
    for file in files:
        target_dir_name = os.path.dirname(file[1])

        # will create all intermediate-level directories
        if not os.path.exists(target_dir_name):
            os.makedirs(target_dir_name)

        shutil.copyfile(file[0], file[1])


# URL helpers
# ------------------------------------------------------------------------------------------#

def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True


def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    assert is_url(url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        safe_name = safe_name[:min(len(safe_name), 128)]
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file) # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)
