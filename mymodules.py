import skdim
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.hermite import hermval
from itertools import combinations
from itertools import product
from numpy.polynomial.legendre import legval
from group_lasso import GroupLasso
import statsmodels.api as sm
import time
import os
from scipy.linalg import svd
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import ElasticNetCV
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import *
from scipy.special import gamma
import ot 
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
import math
from kneed import KneeLocator
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import random
from matplotlib.patches import Patch
import requests
import json
from sklearn.decomposition import SparsePCA
from pydiffmap import diffusion_map as dm
from sklearn.preprocessing import KernelCenterer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import pwlf
import ruptures as rpt
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed, parallel_backend
import matplotlib as mpl

# ESS
def ESS(d, p, n, K, sample):
    data = sample
    dimensions = np.zeros(n)
    
    knn_model = NearestNeighbors(n_neighbors=K + 1, algorithm='auto', n_jobs=4)
    knn_model.fit(sample)
    # Get the K-nearest neighbors
    distances, indices = knn_model.kneighbors(sample)
    
    # Step 1: Precompute thresholds using Gamma function
    thresholds = np.zeros(p)
    for q in range(1, p + 1):
        thresholds[q - 1] = gamma(q / 2)**2 / (gamma((q + 1)/2) * gamma((q - 1)/2))

    #print(thresholds)
    
    for k in range(n):
        neighbors = data[indices[k, 1:K+1]]
        center = data[k]
        diff = neighbors - center  # (K, D)
        norms = np.linalg.norm(diff, axis=1)  # (K,)
    
        # Cosine of angle between all pairs
        dot_products = diff @ diff.T  # (K, K)
        norm_matrix = np.outer(norms, norms)  # (K, K)
    
        # Avoid divide by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_theta = np.clip(dot_products / norm_matrix, -1.0, 1.0)
            sin_theta = np.sqrt(1 - cos_theta**2)
    
        # 2 times Triangle area matrix: ||a|| * ||b|| * sin(theta)
        area_matrix = norm_matrix * sin_theta
    
        total_area = np.sum(area_matrix)
        total_norm_product = np.sum(norm_matrix)
    
        result = total_area / total_norm_product if total_norm_product > 0 else 0
        #print(result)
        # Invert from thresholds to estimated dimension
        dimensions[k] = p
        for q in range(0, p - 1):
            if result >= thresholds[q] and result < thresholds[q + 1]:
                # Linear interpolation for fractional dimension
                dimensions[k] = q + 1 + (thresholds[q + 1] - result) / (thresholds[q + 1] - thresholds[q])
                break
    
    return np.mean(dimensions)
    
# ABID
def ABID(d, p, n, K, sample):
    data = sample
    dimensions = np.zeros(n)
    
    knn_model = NearestNeighbors(n_neighbors=K + 1, algorithm='auto', n_jobs=4)
    knn_model.fit(sample)
    # Get the K-nearest neighbors
    distances, indices = knn_model.kneighbors(sample)
    
    for k in range(n):
        # Get K neighbor difference vectors (exclude self by skipping indices[k, 0])
        neighbors = data[indices[k, 1:K+1]]  # shape (K, D)
        center = data[k]  # shape (D,)
        diff = neighbors - center  # shape (K, D)
    
        # Compute cosine similarity matrix for the diff vectors
        # First normalize the diff vectors
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        normalized_diff = diff / norms  # shape (K, D)
    
        # Cosine similarity matrix (K x K) via dot products
        cosine_matrix = normalized_diff @ normalized_diff.T  # shape (K, K)
    
        # Square the cosine values and sum over all pairs
        squared_cosines = cosine_matrix**2
        _sum = np.sum(squared_cosines)
    
        # Estimate dimension using the same formula
        dimensions[k] = K * K / _sum
    
    # Quantiles
    return np.mean(dimensions)
    #return dimensions

def wasserstein_single_run(data, n, alpha, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n1 = int(n / (2 + 2 * alpha))
    n2 = int(alpha * n1)
    
    indices_all = np.random.permutation(n)
    subsample1 = data[indices_all[:n1]]
    subsample2 = data[indices_all[n1:2*n1]]
    subsample3 = data[indices_all[2*n1:2*n1 + n2]]
    subsample4 = data[indices_all[2*n1 + n2:2*n1 + 2*n2]]
    
    a = np.ones(n1) / n1
    b = np.ones(n1) / n1
    
    M1 = ot.dist(subsample1, subsample2, metric='euclidean')
    transport_plan1 = ot.emd(a, b, M1)
    W1 = np.sum(transport_plan1 * M1)
    
    a = np.ones(n2) / n2
    b = np.ones(n2) / n2
    
    M2 = ot.dist(subsample3, subsample4, metric='euclidean')
    transport_plan2 = ot.emd(a, b, M2)
    W2 = np.sum(transport_plan2 * M2)
    
    W1 = max(W1, 1e-12)
    W2 = max(W2, 1e-12)
    
    # Force estimate calculation; add a tiny epsilon to denominator to avoid div by zero
    denominator = math.log(W1) - math.log(W2)
    if abs(denominator) < 1e-12:
        denominator = 1e-12 if denominator >= 0 else -1e-12
    
    est = math.log(alpha) / denominator
    return est


def Wasserstein_new(d, p, n, K, data, alpha):
    results = Parallel(n_jobs=-1)(
        delayed(wasserstein_single_run)(data, n, alpha, seed=321+i) for i in range(100)
    )
    return np.median(results)
    
# ISOMAP
def ISOMAP(d, p, n, K, data):
    dim = 1
    gain = 0
    previous = 1
    ratios = [0]
    
    if p > n:
        print("Warning: ambient dimension is truncated, n might be too small.")
        p = int(0.25 * n)
    
    for q in range(1, p + 1):
         
        embedding = Isomap(n_components=q)
        data_transformed = embedding.fit_transform(data)
    
        # Original geodesic distances (used by Isomap)
        G = embedding.dist_matrix_  # This is the geodesic distance matrix Isomap computes internally
    
        # Euclidean distances in the low-dimensional embedding
        D = pairwise_distances(data_transformed)
    
        # Residual variance = 1 - squared correlation between the two distance matrices
        # just treat elements as repeated observations, the smaller, the better
        corr = np.corrcoef(D.ravel(), G.ravel())[0, 1]
        residual_variance = 1 - corr**2
    
        if q > 1:
            temp = (previous - residual_variance) / previous
            ratios.append(temp)
            #print(temp)
            #print(gain)
            if temp > gain and previous > 1e-12:
                gain = temp
                dim = q
        # #print(dim)
        previous = residual_variance
        
            
    #knee = KneeLocator(range(1, p + 1), ratios, curve='convex', direction='increasing').knee#dim - 1
    #print(knee, dim)
    if True:
        return dim


# weighted local PCA
def lPCA_weighted(p, n, K, sample, alpha=0.05):
    if p > K:
        p = K
        print("Warning: ambient dimension is truncated, K might be too small!")

    data = sample
    dimensions = np.zeros(n)

    knn_model = NearestNeighbors(n_neighbors=K + 1, algorithm='auto', n_jobs=-1)
    knn_model.fit(data)
    _, indices = knn_model.kneighbors(data)

    for k in range(n):
        neighbors = data[indices[k, 1:K]]  # exclude self
        temp_data = np.vstack((data[k], neighbors))
        center = np.mean(temp_data, axis=0)
        diff = temp_data - center
        norms = np.linalg.norm(diff, axis=1) + 1e-12  # to avoid division by 0

        # PCA instead of full covariance + eigh
        pca = PCA(n_components=p)
        local_coords = pca.fit_transform(diff)
        local_coords /= norms[:, np.newaxis]

        squared_coords = local_coords ** 2
        squared_sums = np.sum(squared_coords, axis=0)
        cumulative_sums = np.cumsum(squared_sums)

        for j in range(p):
            if cumulative_sums[j] / cumulative_sums[-1] >= 1 - alpha:
                dimensions[k] = j + 1
                break

    return np.mean(dimensions)

# linear embedding
def compute_l_dimension_raw(k, data, indices, K, p):
    all_neighbors = data[indices[k, :K]]
    center = np.mean(all_neighbors, axis=0)
    diff = all_neighbors - center

    # Use only the first `p` neighbors to determine the PCA basis and index
    pca_subset = diff[:K]
    pca = PCA(n_components=p)
    pca.fit(pca_subset)
    diff_pca = pca.transform(diff)  # Project all K neighbors into the PCA space

    # Determine intrinsic index from just the p neighbors
    subset_pca = pca.transform(pca_subset)
    stds = np.std(subset_pca, axis=0)
    index = next((p - 1 - j for j in range(p) if stds[p - 1 - j] > 1e-12), -1)

    if index < 0:
        return {'f_pvalues': [], 'adj_r2': []}
    if index == 0:
        return {'f_pvalues': [1.0], 'adj_r2': [1.0]}

    f_pvalues = []
    adj_r2 = []

    for d in range(1, index + 1):
        if d >= diff_pca.shape[1]:
            continue

        X_d = diff_pca[:, :d]    # Use all K neighbors here
        y = diff_pca[:, d]

        X_design = np.hstack([np.ones((X_d.shape[0], 1)), X_d])
        model = sm.OLS(y, X_design)
        results = model.fit()

        f_pvalues.append(results.f_pvalue)
        adj_r2.append(results.rsquared_adj)

    # print(index, f_pvalues)

    return {'f_pvalues': f_pvalues, 'adj_r2': adj_r2}

def process_l_batch_raw(batch_indices, data, indices, K, p):
    return [compute_l_dimension_raw(k, data, indices, K, p) for k in batch_indices]

def l_estimator_parallel_v13(p, n, K, sample, n_jobs=-1, batch_size=None, num_neighborhoods=100):
    data = sample
    knn_model = NearestNeighbors(n_neighbors=K + 1, algorithm='auto', n_jobs=n_jobs)
    knn_model.fit(data)
    _, indices = knn_model.kneighbors(data)

    if p >= K: # we need K - (p - 1) - 1 > 0, i.e., K > p
        p = K - 1
        print("Warning: ambient dimension is truncated, K might be too small.")

    # Sample specified number of neighborhoods or all if less than requested
    nbhs_to_use = np.random.choice(n, size=min(num_neighborhoods, n), replace=False)

    if batch_size is None:
        batch_size = max(10, len(nbhs_to_use) // (4 * os.cpu_count()))

    batches = [nbhs_to_use[i: i + batch_size] for i in range(0, len(nbhs_to_use), batch_size)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_l_batch_raw)(batch, data, indices, K, p) for batch in batches
    )

    flat_results = [r for batch_result in results for r in batch_result]

    max_len = max(len(res['f_pvalues']) for res in flat_results if res['f_pvalues'])

    all_f_pvalues = np.full((len(flat_results), max_len), np.nan)
    all_adj_r2 = np.full((len(flat_results), max_len), np.nan)

    for i, res in enumerate(flat_results):
        length = len(res['f_pvalues'])
        all_f_pvalues[i, :length] = res['f_pvalues']
        all_adj_r2[i, :length] = res['adj_r2']

    dimensions = []
    weights = []
    
    for row_pvalues, row_adj_r2 in zip(all_f_pvalues, all_adj_r2):
        if np.all(np.isnan(row_pvalues)) or np.all(np.isnan(row_adj_r2)):
            continue
    
        dim = len(row_pvalues) + 1  # default if no condition met
        weight = 0  # default weight

        found = False
        
        for i in range(len(row_pvalues)):
            if not np.isnan(row_pvalues[i]) and not np.isnan(row_adj_r2[i]):
                if row_pvalues[i] < 0.01 and row_adj_r2[i] > 0:
                    remaining_pvals = row_pvalues[i+1:]
                    if np.all(np.isnan(remaining_pvals)) or np.all(remaining_pvals < 0.01):
                        dim = i + 1
                        weight = row_adj_r2[i]
                        found = True
                        break
            
        dimensions.append(dim)
        weights.append(weight)
    
    # Convert to numpy arrays for weighted average calculation
    dimensions = np.array(dimensions)
    weights = np.array(weights)
    
    # Avoid division by zero
    if np.sum(weights) == 0:
        final_dimension = np.mean(dimensions)  # fallback to unweighted mean
    else:
        final_dimension = np.average(dimensions, weights=weights)
    
    # Optional: round to nearest int
    # final_dimension = int(round(final_dimension))
    
    return final_dimension

# quadratic embedding
def compute_q_dimension_raw(k, data, indices, K, p):
    
    neighbors_pca = data[indices[k, 0:K]]
    center = np.mean(neighbors_pca, axis=0)
    diff_pca = neighbors_pca - center

    pca = PCA(n_components=p)
    diff_pca_transformed = pca.fit_transform(diff_pca)

    stds = np.std(diff_pca_transformed, axis=0)
    index = next((p - 1 - j for j in range(p) if stds[p - 1 - j] > 1e-12), -1)

    if index < 0:
        return {'f_pvalues': [], 'adj_r2': []}
    if index == 0:
        return {'f_pvalues': [1.0], 'adj_r2': [1.0]}

    # For regression: use all K neighbors, but projected into PCA space computed above
    neighbors_reg = data[indices[k, 0:K]]
    diff_reg = neighbors_reg - center  # center using p neighbors mean from PCA step

    # Project all K neighbors into the PCA basis computed from p neighbors
    diff_reg_transformed = pca.transform(diff_reg)

    f_pvalues = []
    adj_r2 = []

    for d in range(1, index + 1):
        if d >= diff_reg_transformed.shape[1]:
            continue

        X_d = diff_reg_transformed[:, :d]
        X_design = [X_d, X_d ** 2]

        cross_terms = [X_d[:, i] * X_d[:, j] for i in range(d) for j in range(i + 1, d)]
        if cross_terms:
            X_design.append(np.stack(cross_terms, axis=1))

        X_design = np.hstack(X_design)
        X_design = sm.add_constant(X_design)
        y = diff_reg_transformed[:, d]

        model = sm.OLS(y, X_design)
        results = model.fit()

        f_pvalues.append(results.f_pvalue)
        adj_r2.append(results.rsquared_adj)

    return {'f_pvalues': f_pvalues, 'adj_r2': adj_r2}

def process_q_batch_raw(batch_indices, data, indices, K, p):
    return [compute_q_dimension_raw(k, data, indices, K, p) for k in batch_indices]

def q_estimator_parallel_v13(p, n, K, sample, n_jobs=-1, batch_size=None, num_neighborhoods=100):
    data = sample
    knn_model = NearestNeighbors(n_neighbors=K + 1, algorithm='auto', n_jobs=n_jobs)
    knn_model.fit(data)
    _, indices = knn_model.kneighbors(data)

    if K <= p * (p - 1) / 2 + 1: # we need K - (p - 1)p/2 - 1 > 0
        val = (-1 + math.sqrt(1 + 8 * K)) / 2
        if val.is_integer():
            p = int(val) - 1
        else:
            p = math.floor(val)
        print(p, "Warning: ambient dimension is truncated, K might be too small.")

    # Sample specified number of neighborhoods or all if less than requested
    nbhs_to_use = np.random.choice(n, size=min(num_neighborhoods, n), replace=False)

    if batch_size is None:
        batch_size = max(10, len(nbhs_to_use) // (4 * os.cpu_count()))

    batches = [nbhs_to_use[i: i + batch_size] for i in range(0, len(nbhs_to_use), batch_size)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_q_batch_raw)(batch, data, indices, K, p) for batch in batches
    )

    flat_results = [r for batch_result in results for r in batch_result]

    max_len = max(len(res['f_pvalues']) for res in flat_results if res['f_pvalues'])

    all_f_pvalues = np.full((len(flat_results), max_len), np.nan)
    all_adj_r2 = np.full((len(flat_results), max_len), np.nan)

    for i, res in enumerate(flat_results):
        length = len(res['f_pvalues'])
        all_f_pvalues[i, :length] = res['f_pvalues']
        all_adj_r2[i, :length] = res['adj_r2']

    dimensions = []
    weights = []
    
    for row_pvalues, row_adj_r2 in zip(all_f_pvalues, all_adj_r2):
        if np.all(np.isnan(row_pvalues)) or np.all(np.isnan(row_adj_r2)):
            continue
    
        dim = len(row_pvalues) + 1  # default if no condition met
        weight = 0  # default weight

        found = False
        
        for i in range(len(row_pvalues)):
            if not np.isnan(row_pvalues[i]) and not np.isnan(row_adj_r2[i]):
                if row_pvalues[i] < 0.01 and row_adj_r2[i] > 0:
                    remaining_pvals = row_pvalues[i+1:]
                    if np.all(np.isnan(remaining_pvals)) or np.all(remaining_pvals < 0.01):
                        dim = i + 1
                        weight = row_adj_r2[i]
                        found = True
                        break
            
        dimensions.append(dim)
        weights.append(weight)
    
    # Convert to numpy arrays for weighted average calculation
    dimensions = np.array(dimensions)
    weights = np.array(weights)
    
    # Avoid division by zero
    if np.sum(weights) == 0:
        final_dimension = np.mean(dimensions)  # fallback to unweighted mean
    else:
        final_dimension = np.average(dimensions, weights=weights)
    
    # Optional: round to nearest int
    # final_dimension = int(round(final_dimension))
    
    return final_dimension
    
# total least squares
def compute_dimension_tls(k, data, indices, K, p):
    neighbors = data[indices[k, 0:K]]
    center = np.mean(neighbors, axis=0)
    diff = neighbors - center

    # PCA fit only on first p neighbors
    pca = PCA(n_components=p)
    pca.fit(diff[:K]) 
    diff_pca = pca.transform(diff)  # Transform all K neighbors

    # Determine active dimension index using only first p projected neighbors
    stds = np.std(diff_pca[:p], axis=0)
    index = next((p - 1 - j for j in range(p) if stds[p - 1 - j] > 1e-12), -1)

    if index < 0:
        return {'rss': []}
    if index == 0:
        return {'rss': [0.0]}

    rss = []
    n = diff_pca.shape[0]

    for d in range(1, index + 1):
        if d >= diff_pca.shape[1]:
            continue

        X = diff_pca[:, :d]
        y = diff_pca[:, d]

        # Build polynomial design matrix: linear, quadratic, cross terms
        X_design = [X, X ** 2]
        cross_terms = [X[:, i] * X[:, j] for i in range(d) for j in range(i + 1, d)]
        if cross_terms:
            X_design.append(np.stack(cross_terms, axis=1))
        X_poly = np.hstack(X_design)

        # Center predictors and response
        X_poly_std = X_poly - np.mean(X_poly, axis=0)
        y_std = y - np.mean(y)

        # TLS: joint matrix
        joint_matrix = np.hstack([X_poly_std, y_std[:, np.newaxis]])
        U, S, Vt = np.linalg.svd(joint_matrix, full_matrices=False)
        normal_vector = Vt[-1, :]  # Last row of V^T is the normal

        residuals = joint_matrix @ normal_vector
        rss_tls = np.sum(residuals ** 2)

        complexity = 2 * d + (d * (d - 1)) // 2# heuristic weight
        rss.append(complexity * rss_tls)
    
    return {'rss': rss}

def process_tls(batch_indices, data, indices, K, p):
    return [compute_dimension_tls(k, data, indices, K, p) for k in batch_indices]

def tls_estimator_parallel_v13(p, n, K, sample, n_jobs=-1, batch_size=None, num_neighborhoods=100):
    data = sample
    knn_model = NearestNeighbors(n_neighbors=K + 1, algorithm='auto', n_jobs=n_jobs)
    knn_model.fit(data)
    _, indices = knn_model.kneighbors(data)

    if K <= p * (p - 1) / 2 + 1: # we need K - (p - 1)p/2 - 1 > 0
        val = (-1 + math.sqrt(1 + 8 * K)) / 2
        if val.is_integer():
            p = int(val) - 1
        else:
            p = math.floor(val)
        print("Warning: ambient dimension is truncated, K might be too small.")

    # Sample specified number of neighborhoods or all if less than requested
    nbhs_to_use = np.random.choice(n, size=min(num_neighborhoods, n), replace=False)

    if batch_size is None:
        batch_size = max(10, len(nbhs_to_use) // (4 * os.cpu_count()))

    batches = [nbhs_to_use[i: i + batch_size] for i in range(0, len(nbhs_to_use), batch_size)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_tls)(batch, data, indices, K, p) for batch in batches
    )

    flat_results = [r for batch in results for r in batch if r['rss']]

    max_len = max(len(r['rss']) for r in flat_results)
    all_rss = np.full((len(flat_results), max_len), np.nan)

    for i, res in enumerate(flat_results):
        all_rss[i, :len(res['rss'])] = res['rss']

    estimated_dims = []
    
    for rss in all_rss:
        # Remove NaNs from the current RSS vector
        valid_rss = rss[~np.isnan(rss)]
        est_dim = len(valid_rss) + 1  # default if no drop found
    
        for i in range(1, len(valid_rss)):
            if valid_rss[i] < valid_rss[i - 1]:
                if all(valid_rss[j] <= valid_rss[j - 1] for j in range(i + 1, len(valid_rss))):
                    est_dim = i + 1  # 1-based indexing
                    break

        estimated_dims.append(est_dim)
    
    # Take the average of all estimated dimensions
    final_estimated_dim = np.mean(estimated_dims)
    
    return final_estimated_dim

def CAPCA(p, n, K, sample):

    if p > K:
        p = K # with K points, we can find at most K different directions

    data = sample
    dimensions = np.zeros(n)

    knn_model = NearestNeighbors(n_neighbors=K + 2, algorithm='auto', n_jobs=-1)
    knn_model.fit(data)
    _, indices = knn_model.kneighbors(data)

    for k in range(n):
        neighbors = data[indices[k, 0:K]]  # exclude self
        
        r_1 = np.linalg.norm(data[indices[k, K]] - data[indices[k, 0]])
        r_2 = np.linalg.norm(data[indices[k, K + 1]] - data[indices[k, 0]])
        r = (r_1 + r_2) / 2
        
        temp_data = neighbors
        center = np.mean(temp_data, axis=0)
        diff = temp_data - center
        
        # PCA instead of full covariance + eigh
        pca = PCA(n_components=p)
        local_coords = pca.fit_transform(diff)
        eigenvalues = pca.explained_variance_

        adjusted_norms = []
        for d in range(1, p + 1):
            expected = np.zeros(p)
            expected[:d] = 1 / (d + 2)
            realised = 1 / r ** 2 * eigenvalues + (3 * d + 4) / (d * (d + 4)) * np.sum(eigenvalues[d:])
            realised[d:] = 0
            adjusted_norms.append(np.linalg.norm(realised - expected) + 2 * np.sum(eigenvalues[d:]))

        dimensions[k] = np.argmin(adjusted_norms) + 1
        # print(dimensions[k], norms)
        
    return np.mean(dimensions)

class SphereSampler:
    def __init__(self, n=1500, d=2, p=3, R=1, sigma=0.0, seed=None):
        self.n = n
        self.d = d
        self.p = p
        self.R = R
        self.seed = seed
        self.sigma = sigma

        # Seed
        if seed is not None:
            np.random.seed(seed)

        # Generate random orthogonal rotation matrix Q
        matrix = np.random.randn(p, p)
        Q, T = np.linalg.qr(matrix)
        D = np.diag(np.sign(np.diag(T)))
        Q = Q @ D
        T = D @ T

        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
            T[0, :] *= -1

        self.rotation = Q
        self.translation = 100 * skdim.datasets.hyperBall(n=1, d=p, radius=1, random_state=0)

    def f_uniform_density(self, x):
        # Uniform density on d-sphere
        return 1 / (2 * math.pi ** ((self.d + 1) / 2) * self.R ** self.d / gamma((self.d + 1) / 2))

    def f_nonuniform_density(self, x):
        return np.linalg.norm(x, axis=1)

    def phi_inverse(self, theta):
        n, d = theta.shape[0], self.d
        x = np.zeros((n, d + 1))
        sin_values = np.sin(theta)
        cos_values = np.cos(theta)
        for j in range(d + 1):
            if j != d:
                x[:, j] = self.R * cos_values[:, j]
            else:
                x[:, j] = self.R * sin_values[:, d - 1]
            for i in range(min(j, d - 1)):
                x[:, j] *= sin_values[:, i]
        return x

    def J_phi_inverse(self, theta):
        n = theta.shape[0]
        Js = np.ones(n) * self.R ** self.d
        sin_values = np.sin(theta)
        for j in range(self.d - 1):
            Js *= sin_values[:, j] ** (self.d - 1 - j)
        return Js

    def sample(self, M=None, uniform=True):
        if M is None:
            if uniform:
                M = self.f_uniform_density(None) / math.pi ** self.d / 2 * 1.1  # heuristic
            else:
                M = 1
        count = 0
        sample = np.zeros((self.n, self.p))    

        while count < self.n:
            theta = np.random.uniform(0, math.pi, (self.n, self.d - 1))
            theta = np.hstack((theta, np.random.uniform(0, 2 * math.pi, (self.n, 1))))
            xs = self.phi_inverse(theta)
            if uniform:
                densities = self.f_uniform_density(None) * np.ones(self.n)
            else:
                #densities = self.f_nonuniform_density(xs) * np.ones(self.n)
                # Example Beta parameters for the first (d-1) dimensions
                alpha_1 = 2.0
                beta_1 = 5.0
                # Sample from Beta distribution and scale to [0, pi]
                theta = np.random.beta(alpha_1, beta_1, (self.n, self.d - 1)) * math.pi
                # For the last dimension, sample from Beta but scale to [0, 2*pi]
                alpha_2 = 2.0
                beta_2 = 2.0
                theta_last = np.random.beta(alpha_2, beta_2, (self.n, 1)) * 2 * math.pi
                # Concatenate the results
                theta = np.hstack((theta, theta_last))
                xs = self.phi_inverse(theta)
                sample[:, 0:(self.d + 1)] = xs
                return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(np.zeros(self.p), self.sigma ** 2 * np.eye(self.p),size=self.n)
            Js = self.J_phi_inverse(theta)
            us = np.random.uniform(0, 1, (self.n, 1)).flatten()
            ps = densities * Js / M
            for k in range(self.n):
                if us[k] <= ps[k]:
                    sample[count, 0:(self.d + 1)] = xs[k, :]
                    count += 1
                    if count == self.n:
                        break

        return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(np.zeros(self.p), self.sigma ** 2 * np.eye(self.p),size=self.n)
        

    def sample_fast(self):
        sample = np.zeros((self.n, self.p))
        sample[:,:self.d + 1] = self.R * skdim.datasets.hyperSphere(n = self.n, d = self.d, random_state=self.seed)
        return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(np.zeros(self.p), self.sigma ** 2 * np.eye(self.p),size=self.n)


class CosSinSampler:
    def __init__(self, n=1000, d=2, sigma=0.0, seed=None, uniform=True, beta_params=(2.0, 5.0),
                 R=1.0, r=0.5, k=2.0):
        self.n = n
        self.d = d
        self.sigma = sigma
        self.uniform = uniform
        self.beta_params = beta_params
        self.R = R
        self.r = r
        self.k = k
        
        if seed is not None:
            np.random.seed(seed)
        
        # Random 2d×2d rotation matrix
        matrix = np.random.randn(2*d, 2*d)
        Q, R_ = np.linalg.qr(matrix)
        D = np.diag(np.sign(np.diag(R_)))
        Q = Q @ D
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        self.rotation = Q
        self.translation = np.zeros(2*d)

        # Estimate upper bound of Jacobian norm for rejection sampling
        self.Jmax = 4 * np.pi * (abs(R) + abs(r) * (1 + abs(k))) ** d

    def embedding(self, X):
        X = np.asarray(X)
        theta = 2 * np.pi * X
        phi = 2 * self.k * np.pi * X

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)

        x_part = self.R * cos_theta + self.r * cos_phi * cos_theta
        y_part = self.R * sin_theta + self.r * cos_phi * sin_theta

        return np.hstack([x_part, y_part])

    def jacobian_det(self, X):
        X = np.asarray(X)
        theta = 2 * np.pi * X
        phi = 2 * self.k * np.pi * X

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        # Compute f'(x_j) and g'(x_j) per dimension
        df_dx = -2 * np.pi * (
            self.R * sin_theta +
            self.r * (self.k * sin_phi * cos_theta + cos_phi * sin_theta)
        )
        dg_dx = 2 * np.pi * (
            self.R * cos_theta +
            self.r * (-self.k * sin_phi * sin_theta + cos_phi * cos_theta)
        )

        norms = np.sqrt(df_dx**2 + dg_dx**2)
        return np.prod(norms, axis=1)

    def draw_base_samples(self, m):
        if self.uniform:
            return np.random.uniform(0, 1, size=(m, self.d))
        else:
            alpha, beta = self.beta_params
            return np.random.beta(alpha, beta, size=(m, self.d))

    def sample(self):
        if self.uniform:
            samples = []
            while len(samples) < self.n:
                m = self.n - len(samples)
                x = self.draw_base_samples(m)
                Jvals = self.jacobian_det(x)
                u = np.random.uniform(0, self.Jmax, size=m)
                accepted_mask = u <= Jvals
                accepted = x[accepted_mask]
                if accepted.size > 0:
                    samples.append(accepted)
            samples = np.vstack(samples)[:self.n]
        else:
            samples = self.draw_base_samples(self.n)

        embedded = self.embedding(samples)
        transformed = embedded @ self.rotation.T + self.translation

        if self.sigma > 0:
            noise = np.random.normal(0, self.sigma, transformed.shape)
            transformed += noise

        return transformed
        
        

class BallSampler:
    def __init__(self, n=1500, d=2, p=3, R=1, sigma=0.0, seed=None):
        self.n = n
        self.d = d
        self.p = p
        self.R = R
        self.sigma = sigma
        self.seed = seed

        # Seed
        if seed is not None:
            np.random.seed(seed)

        # Generate random orthogonal rotation matrix Q
        matrix = np.random.randn(p, p)
        Q, T = np.linalg.qr(matrix)
        D = np.diag(np.sign(np.diag(T)))
        Q = Q @ D
        T = D @ T

        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
            T[0, :] *= -1

        self.rotation = Q
        self.translation = 100 * skdim.datasets.hyperBall(n=1, d=p, radius=1, random_state=0)

    def f_uniform_density(self, x):
        # Uniform density on d-sphere
        return gamma(self.d / 2 + 1) / (self.R ** self.d * math.pi ** (self.d / 2))

    def f_nonuniform_density(self, x):
        return np.linalg.norm(x, axis=1)

    def phi_inverse(self, theta):
        n = theta.shape[0]
        r = theta[:, 0]
        angles = theta[:, 1:]
    
        x = np.zeros((n, self.d))
        #print(angles)
        for i in range(self.d):
            x[:, i] = r
            for j in range(i):
                x[:, i] *= np.sin(angles[:, j])
            if i < self.d - 1:
                x[:, i] *= np.cos(angles[:, i])
            else:
                x[:, i] *= 1
        return x

    def J_phi_inverse(self, theta):
        r = theta[:, 0]
        angles = theta[:, 1:]
        Js = r ** (self.d - 1)
        for j in range(self.d - 2):  # θ_{d-1} is excluded
            Js *= np.sin(angles[:, j]) ** (self.d - j - 2)
        return Js

    def sample(self, M=None, uniform=True):
        if M is None:
            if uniform:
                M = self.f_uniform_density(None) / math.pi ** (self.d - 1) / 2 / self.R * 1.1  # heuristic
            else:
                M = 1
        count = 0
        sample = np.zeros((self.n, self.p))

        while count < self.n:
            theta = np.random.uniform(0, math.pi, (self.n, self.d - 2))
            theta = np.hstack((theta, np.random.uniform(0, 2 * math.pi, (self.n, 1))))
            theta = np.hstack((np.random.uniform(0, self.R, (self.n, 1)), theta))
            xs = self.phi_inverse(theta)
            if uniform:
                densities = self.f_uniform_density(None) * np.ones(self.n)
            else:
                # Beta params (adjust these to get the desired shape)
                alpha_angle = 2.0
                beta_angle = 5.0

                alpha_last_angle = 2.0
                beta_last_angle = 2.0

                alpha_radius = 2.0
                beta_radius = 3.0

                # Sample first (d-2) angles from Beta, scaled to [0, pi]
                theta_angles = np.random.beta(alpha_angle, beta_angle, (self.n, self.d - 2)) * math.pi

                # Sample last angle from Beta, scaled to [0, 2*pi]
                theta_last_angle = np.random.beta(alpha_last_angle, beta_last_angle, (self.n, 1)) * 2 * math.pi

                # Sample radius value from Beta, scaled to [0, R]
                theta_radius = np.random.beta(alpha_radius, beta_radius, (self.n, 1)) * self.R

                # Concatenate radius and angles
                theta = np.hstack((theta_radius, theta_angles, theta_last_angle))

                xs = self.phi_inverse(theta)
                sample[:, 0:(self.d)] = xs
                return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(np.zeros(self.p), self.sigma ** 2 * np.eye(self.p),size=self.n)
            Js = self.J_phi_inverse(theta)
            us = np.random.uniform(0, 1, (self.n, 1)).flatten()
            ps = densities * Js / M
            for k in range(self.n):
                if us[k] <= ps[k]:
                    sample[count, 0:self.d] = xs[k, :]
                    count += 1
                    if count == self.n:
                        break

        return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(np.zeros(self.p), self.sigma ** 2 * np.eye(self.p),size=self.n)
    
    def sample_fast(self):
        """
        Sample points uniformly inside a d-dimensional ball of radius R using skdim's hyperball.
        Applies rotation, translation, and Gaussian noise.
        """
        # Step 1: Sample from the d-dimensional unit ball
        samples = skdim.datasets.hyperBall(n=self.n, d=self.d, random_state=self.seed)

        # Step 2: Scale to desired radius
        samples *= self.R

        # Step 3: Embed in p-dimensional space if needed
        if self.p > self.d:
            samples_full = np.zeros((self.n, self.p))
            samples_full[:, :self.d] = samples
        else:
            samples_full = samples[:, :self.p]

        # Step 4: Apply rotation
        rotated = samples_full @ self.rotation.T

        # Step 5: Apply translation
        translated = rotated + self.translation

        # Step 6: Add Gaussian noise
        noisy = translated + np.random.multivariate_normal(
            mean=np.zeros(self.p),
            cov=self.sigma ** 2 * np.eye(self.p),
            size=self.n
        )

        return noisy
        
# nonuniform use sample_fast()
class NormalSurfaceSampler:
    def __init__(self, n=1000, d=2, p=3, sd=0.5, sigma=0.0, seed=None):
        self.n = n
        self.d = d 
        self.p = p
        self.sd = sd
        self.sigma = sigma

        mu = np.zeros(d)               # Mean vector
        Sigma = sd ** 2 * np.eye(d)    # Covariance matrix
        self.rv = multivariate_normal(mean=mu, cov=Sigma)

        if seed is not None:
            np.random.seed(seed)

        # Generate random rotation matrix
        matrix = np.random.randn(p, p)
        Q, T = np.linalg.qr(matrix)
        D = np.diag(np.sign(np.diag(T)))
        Q = Q @ D
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        self.rotation = Q
        
        # Reduced translation magnitude for stability
        self.translation = 10 * skdim.datasets.hyperBall(n=1, d=p, radius=1, random_state=0)

        # Cache constant for J_phi_inverse
        self.const_J = 1 / ((2 * math.pi * sd ** 2) ** (d / 2) * sd ** 2)

    def estimate_max_J_density(self, num_trials=100000):
        theta = np.random.uniform(-3 * self.sd, 3 * self.sd, (num_trials, self.d))
        Js = self.J_phi_inverse(theta)
        # Uniform density on parameter space assumed 1 here, adjust if needed
        fs = np.ones(num_trials)
        return np.max(Js * fs) * 1.2

    def f_uniform_density(self, x):
        # Uniform density on parameter space (simplified)
        return 1

    def phi_inverse(self, theta):
        # theta shape: (m, d)
        theta = np.asarray(theta)
        x = np.zeros((theta.shape[0], self.d + 1))
        x[:, 0:self.d] = theta
        x[:, self.d] = self.rv.pdf(theta)
        return x

    def J_phi_inverse(self, theta):
        # theta: shape (n, d)
        pdf_vals = self.rv.pdf(theta)  # shape (n,)
        norm_theta_sq = np.sum(theta**2, axis=1)  # shape (n,)
        
        # Gradient norm squared of pdf wrt theta_j
        grad_pdf_norm_sq = (pdf_vals ** 2) * (norm_theta_sq) / (self.sd ** 4)
        
        # Jacobian volume scaling factor
        Js = np.sqrt(1 + grad_pdf_norm_sq)
    
        return Js

    def sample(self, M=None, uniform=True):
        if M is None:
            if uniform:
                M = self.estimate_max_J_density()
            else:
                M = 1

        count = 0
        sample = np.zeros((self.n, self.p))

        while count < self.n:
            m = self.n - count
            # Sample theta uniformly in parameter domain
            theta = np.random.uniform(-3 * self.sd, 3 * self.sd, (m, self.d))
            xs = self.phi_inverse(theta)

            if uniform:
                densities = self.f_uniform_density(None) * np.ones(m)
            else:
                # Nonuniform sampling via Beta distribution on [0,1] scaled to [-3sd, 3sd]
                alpha, beta = 2.0, 5.0
                beta_samples = np.random.beta(alpha, beta, size=(m, self.d))
                theta = beta_samples * (6 * self.sd) - 3 * self.sd
                xs = self.phi_inverse(theta)
                # Direct return, no rejection sampling here
                sample_part = np.zeros((m, self.p))
                sample_part[:, :self.d + 1] = xs
                transformed = sample_part @ self.rotation.T + self.translation
                if self.sigma > 0:
                    noise = np.random.normal(0, self.sigma, size=transformed.shape)
                    transformed += noise
                # Fill remaining slots (if any) and return early
                sample[count:count+m] = transformed
                return sample

            Js = self.J_phi_inverse(theta)
            us = np.random.uniform(0, 1, m)
            ps = densities * Js / M

            for k in range(m):
                if us[k] <= ps[k]:
                    sample[count, 0:(self.d + 1)] = xs[k, :]
                    count += 1
                    if count == self.n:
                        break

        transformed = sample @ self.rotation.T + self.translation
        if self.sigma > 0:
            noise = np.random.normal(0, self.sigma, size=transformed.shape)
            transformed += noise
        return transformed

    def sample_fast(self):
        # Sample from Beta distribution (parameters can be tuned for shape)
        alpha, beta = 2.0, 5.0  # example params, you can adjust
        
        # Sample Beta in [0,1]^d
        beta_samples = np.random.beta(alpha, beta, (self.n, self.d))
        
        # Scale Beta samples from [0,1] to approx Gaussian support, e.g. [-3*sd, 3*sd]
        X = beta_samples * (6 * self.sd) - 3 * self.sd  # shape (n, d)
        
        # Calculate pdf at those points
        pdf_vals = self.rv.pdf(X)[:, None]  # shape (n, 1)
        
        # Create manifold embedding
        X_surface = np.hstack((X, pdf_vals))  # (n, d+1)
        
        # Pad to ambient dimension p
        padded = np.zeros((self.n, self.p))
        padded[:, :self.d + 1] = X_surface
        
        # Apply rotation and translation
        rotated = padded @ self.rotation.T
        noisy = rotated + self.translation
        
        # Add noise if sigma > 0
        if self.sigma > 0:
            noisy += np.random.multivariate_normal(np.zeros(self.p), self.sigma ** 2 * np.eye(self.p), size=self.n)
        
        return noisy
        
class CylinderSampler:
    def __init__(self, n=1000, d=2, p=3, R=1, h=1, sigma=0.0, seed=None):
        self.n = n
        self.d = d  # dimension of the cylinder base (circle -> 2D)
        self.p = p  # embedding dimension
        self.R = R  # radius
        self.h = h  # height
        self.sigma = sigma

        if seed is not None:
            np.random.seed(seed)

        # Generate random rotation matrix
        matrix = np.random.randn(p, p)
        Q, T = np.linalg.qr(matrix)
        D = np.diag(np.sign(np.diag(T)))
        Q = Q @ D
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        self.rotation = Q
        self.translation = 100 * skdim.datasets.hyperBall(n=1, d=p, radius=1, random_state=0)

    def f_uniform_density(self, x):
        # Uniform density on the surface of a cylinder
        return 1 / (2 * math.pi * self.R * self.h)

    def f_nonuniform_density(self, x):
        return np.linalg.norm(x, axis=1)

    def phi_inverse(self, theta):
        """
        theta: array of shape (n, 2)
        column 0: angle theta in [0, 2pi]
        column 1: height z in [0, h]
        """
        n = theta.shape[0]
        x = np.zeros((n, self.d + 1))
        angle = theta[:, 0]
        height = theta[:, 1]
        x[:, 0] = self.R * np.cos(angle)
        x[:, 1] = self.R * np.sin(angle)
        x[:, 2] = height
        return x

    def J_phi_inverse(self, theta):
        # For cylindrical coordinates, Jacobian is constant R
        return np.ones(theta.shape[0]) * self.R

    def sample(self, M=None, uniform=True):
        if M is None:
            if uniform:
                M = self.f_uniform_density(None) / math.pi / 2 / self.h * 1.1  # heuristic
            else:
                M = 1

        count = 0
        sample = np.zeros((self.n, self.p))

        while count < self.n:
            theta = np.random.uniform(0, 2 * math.pi, (self.n, 1))
            height = np.random.uniform(0, self.h, (self.n, 1))
            params = np.hstack((theta, height))

            xs = self.phi_inverse(params)
            if uniform:
                densities = self.f_uniform_density(None) * np.ones(self.n)
            else:
            # Beta parameters for theta and height (adjust to control skewness)
                alpha_theta = 2.0
                beta_theta = 2.0  # Symmetric Beta, bell-shaped over [0, 2pi]

                alpha_height = 2.0
                beta_height = 5.0  # Skewed Beta, more samples near 0

                # Sample theta from Beta and scale to [0, 2*pi]
                theta = np.random.beta(alpha_theta, beta_theta, (self.n, 1)) * 2 * math.pi

                # Sample height from Beta and scale to [0, h]
                height = np.random.beta(alpha_height, beta_height, (self.n, 1)) * self.h

                params = np.hstack((theta, height))

                xs = self.phi_inverse(params)
                sample[:, 0:(self.d + 1)] = xs
                return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(np.zeros(self.p), self.sigma ** 2 * np.eye(self.p),size=self.n)
            Js = self.J_phi_inverse(params)
            us = np.random.uniform(0, 1, self.n)
            ps = densities * Js / M

            for k in range(self.n):
                if us[k] <= ps[k]:
                    sample[count, 0:self.d + 1] = xs[k]
                    count += 1
                    if count == self.n:
                        break

        # Apply rotation, translation, and Gaussian noise
        return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(
            np.zeros(self.p), self.sigma ** 2 * np.eye(self.p), size=self.n
        )
        


class HelixSampler:
    def __init__(self, n=1000, d=1, p=3, R=1, h=1, num=6, sigma=0.0, seed=None):
        self.n = n
        self.d = d  # Dimension of the helix base (1D base, spiraling in p-dimensional space)
        self.p = p  # Embedding dimension (typically 3 for 3D space)
        self.R = R  # Radius of the helix
        self.h = h  # Height of the helix
        self.num = num  # Number of rotations per unit height
        self.sigma = sigma

        if seed is not None:
            np.random.seed(seed)

        # Generate random rotation matrix
        matrix = np.random.randn(p, p)
        Q, T = np.linalg.qr(matrix)
        D = np.diag(np.sign(np.diag(T)))
        Q = Q @ D
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        self.rotation = Q
        self.translation = 100 * skdim.datasets.hyperBall(n=1, d=p, radius=1, random_state=0)

    def f_uniform_density(self, x):
        # Uniform density along the helix
        return 1 / (self.num * math.sqrt((2 * math.pi * self.R) ** 2 + self.h ** 2))

    def f_nonuniform_density(self, x):
        # Uniform density on the surface of a cylinder
        return np.linalg.norm(x, axis=1)

    def phi_inverse(self, theta):
        """
        theta: array of shape (n, 1) containing angles (theta) for the helix
        Returns: array of shape (n, d + 2), where d=1 for this example, 
        the first two columns are the helix's (x, y) coordinates, and the third is the height.
        """
        n = theta.shape[0]
        x = np.zeros((n, self.d + 2))
        sin_values = np.sin(theta[:, 0])
        cos_values = np.cos(theta[:, 0])

        # Parametrize the 2D helix
        x[:, 0] = self.R * cos_values
        x[:, 1] = self.R * sin_values
        x[:, 2] = theta[:, 0] * self.h / (2 * math.pi)  # Helix height based on angle

        return x

    def J_phi_inverse(self, theta):
        # Jacobian is constant and related to the helix's radius and height
        Js = np.zeros(theta.shape[0])
        Js = Js + self.R ** 2 + (self.h / (2 * math.pi)) ** 2
        return Js

    def sample(self, M=None, uniform=True):
        if M is None:
            if uniform:
                M = self.f_uniform_density(None) / math.pi / 2 / self.num * 1.1  # heuristic
            else:
                M = 1

        count = 0
        sample = np.zeros((self.n, self.p))

        while count < self.n:
            theta = np.random.uniform(0, 2 * math.pi * self.num, (self.n, 1))
            xs = self.phi_inverse(theta)
            if uniform:
                densities = self.f_uniform_density(None) * np.ones(self.n)
            else:
                # Beta distribution parameters (tweak these to adjust skewness)
                alpha = 2.0
                beta = 5.0

                # Sample from Beta(α, β) and scale to [0, 2*pi*self.num]
                theta = np.random.beta(alpha, beta, (self.n, 1)) * 2 * math.pi * self.num

                xs = self.phi_inverse(theta)
                sample[:, 0:(self.d + 2)] = xs
                return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(np.zeros(self.p), self.sigma ** 2 * np.eye(self.p),size=self.n)
            Js = self.J_phi_inverse(theta)
            us = np.random.uniform(0, 1, self.n)
            ps = densities * Js / M

            for k in range(self.n):
                if us[k] <= ps[k]:
                    sample[count, 0:(self.d + 2)] = xs[k, :]
                    count += 1
                    if count == self.n:
                        break

        # Apply rotation, translation, and Gaussian noise (if required)
        return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(
            np.zeros(self.p), self.sigma ** 2 * np.eye(self.p), size=self.n
        )
        

class SwissRollSampler:
    def __init__(self, n=3000, d=2, p=3, a=0, b=4*math.pi, v1=0, v2=20, sigma=0.0, seed=None):
        self.n = n
        self.d = d  # Dimension of the Swiss roll
        self.p = p  # Embedding dimension (typically 3 for 3D space)
        self.a = a  # Lower bound for theta
        self.b = b  # Upper bound for theta
        self.v1 = v1  # Lower bound for the second parameter
        self.v2 = v2  # Upper bound for the second parameter
        self.sigma = sigma

        if seed is not None:
            np.random.seed(seed)

        # Generate random rotation matrix
        matrix = np.random.randn(p, p)
        Q, T = np.linalg.qr(matrix)
        D = np.diag(np.sign(np.diag(T)))
        Q = Q @ D
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        self.rotation = Q
        self.translation = 100 * skdim.datasets.hyperBall(n=1, d=p, radius=1, random_state=0)

    def f_uniform_density(self, x):
        # Uniform density for the Swiss roll
        return 1 / ((self.v2 - self.v1) * 1 / 2 * ((self.b * math.sqrt(1 + self.b ** 2) + np.arcsinh(self.b)) -
                                                     (self.a * math.sqrt(1 + self.a ** 2) + np.arcsinh(self.a))))

    def f_nonuniform_density(self, x):
        # Uniform density on the surface of a cylinder
        return np.linalg.norm(x, axis=1)
    
    def phi_inverse(self, theta):
        """
        theta: array of shape (n, 2) containing the two parameters (theta, z)
        Returns: array of shape (n, d + 1), where the first two columns are (x, y)
        and the third is the height (z).
        """
        n = theta.shape[0]
        x = np.zeros((n, self.d + 1))

        # Parametrize the Swiss roll
        sin_values = np.sin(theta[:, 0])
        cos_values = np.cos(theta[:, 0])

        x[:, 0] = theta[:, 0] * cos_values
        x[:, 1] = theta[:, 0] * sin_values
        x[:, 2] = theta[:, 1]  # Height parameter (z)

        return x

    def J_phi_inverse(self, theta):
        """
        Jacobian of the inverse map for the Swiss roll parametrization.
        """
        Js = theta[:, 0] ** 2
        return Js

    def sample(self, M=None, uniform=True):
        if M is None:
            if uniform:
                M = self.f_uniform_density(None) / (self.b - self.a) / (self.v2 - self.v1) * 1.1  # heuristic
            else:
                M = 1

        count = 0
        sample = np.zeros((self.n, self.p))

        while count < self.n:
            # Sample theta uniformly over the parameter space
            theta = np.random.uniform(self.a, self.b, (self.n, 1))
            theta = np.hstack((theta, np.random.uniform(self.v1, self.v2, (self.n, 1))))

            xs = self.phi_inverse(theta)
            if uniform:
                densities = self.f_uniform_density(None) * np.ones(self.n)
            else:
                # Beta params for first component (adjust to desired skew/shape)
                alpha1, beta1 = 2.0, 5.0

                # Beta params for second component
                alpha2, beta2 = 3.0, 3.0

                # Sample from Beta and scale to [a, b]
                theta1 = np.random.beta(alpha1, beta1, (self.n, 1)) * (self.b - self.a) + self.a

                # Sample from Beta and scale to [v1, v2]
                theta2 = np.random.beta(alpha2, beta2, (self.n, 1)) * (self.v2 - self.v1) + self.v1

                # Combine
                theta = np.hstack((theta1, theta2))

                xs = self.phi_inverse(theta)
                sample[:, 0:(self.d + 1)] = xs
                return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(np.zeros(self.p), self.sigma ** 2 * np.eye(self.p),size=self.n)
            Js = self.J_phi_inverse(theta)
            us = np.random.uniform(0, 1, self.n)
            ps = densities * Js / M

            for k in range(self.n):
                if us[k] <= ps[k]:
                    sample[count, 0:(self.d + 1)] = xs[k, :]
                    count += 1
                    if count == self.n:
                        break

        # Apply rotation and translation to the samples
        return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(
            np.zeros(self.p), self.sigma ** 2 * np.eye(self.p), size=self.n
        )
        

class MobiusBandSampler:
    def __init__(self, n=3000, d=2, p=3, R=1.5, w=1, sigma=0.0, seed=None):
        self.n = n
        self.d = d  # Dimension of the manifold (Mobius Band is 2D)
        self.p = p  # Embedding dimension (typically 3 for 3D space)
        self.R = R  # Radius of the Mobius Band
        self.w = w  # Width of the Mobius Band
        self.sigma = sigma

        if seed is not None:
            np.random.seed(seed)

        # Generate random rotation matrix
        matrix = np.random.randn(p, p)
        Q, T = np.linalg.qr(matrix)
        D = np.diag(np.sign(np.diag(T)))
        Q = Q @ D
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        self.rotation = Q
        self.translation = 100 * skdim.datasets.hyperBall(n=1, d=p, radius=1, random_state=0)

    def f_uniform_density(self, x):
        # Uniform density for the Mobius Band
        return 1

    def f_nonuniform_density(self, x):
        # Uniform density on the surface of a cylinder
        return np.linalg.norm(x, axis=1)
    
    def phi_inverse(self, theta):
        """
        Parametrization of the Mobius band in 3D space.
        theta: Array of shape (n, 2), where first column is the angle along the band and second column is the width along the band.
        """
        n = theta.shape[0]
        x = np.zeros((n, self.d + 1))

        # Parametrize the Mobius Band
        sin_values = np.sin(theta[:, 0])
        cos_values = np.cos(theta[:, 0])

        sin_values_half = np.sin(0.5 * theta[:, 0])
        cos_values_half = np.cos(0.5 * theta[:, 0])

        x[:, 0] = (self.R + theta[:, 1] / 2 * cos_values_half) * cos_values
        x[:, 1] = (self.R + theta[:, 1] / 2 * cos_values_half) * sin_values
        x[:, 2] = theta[:, 1] / 2 * sin_values_half

        return x

    def J_phi_inverse(self, theta):
        """
        Jacobian of the inverse map for the Mobius Band parametrization.
        """
        sin_values_half = np.sin(0.5 * theta[:, 0])
        cos_values_half = np.cos(0.5 * theta[:, 0])
        Js = (self.R + theta[:, 1] * cos_values_half)**2 + theta[:, 1]**2 / 4
        return Js

    def sample(self, M=None, uniform=True):
        if M is None:
            if uniform:
                M = self.f_uniform_density(None) / 2 / math.pi / 2 / self.w * 1.1  # heuristic
            else:
                M = 1

        count = 0
        sample = np.zeros((self.n, self.p))
        #print(sample)
        while count < self.n:
            # Sample theta uniformly over the parameter space
            theta = np.random.uniform(0, 2 * math.pi, (self.n, 1))
            theta = np.hstack((theta, np.random.uniform(-self.w, self.w, (self.n, 1))))

            xs = self.phi_inverse(theta)
            if uniform:
                densities = self.f_uniform_density(None) * np.ones(self.n)
            else:
            # Beta parameters for the first component (angle in [0, 2π])
                alpha_theta = 2.0
                beta_theta = 2.0  # symmetric, bell-shaped around π

                # Beta parameters for the second component (in [-w, w])
                alpha_w = 2.0
                beta_w = 5.0  # skewed toward lower bound (near -w)

                # Sample from Beta(α, β) on [0,1]
                theta_raw = np.random.beta(alpha_theta, beta_theta, (self.n, 1))
                # Scale to [0, 2π]
                theta = theta_raw * 2 * math.pi

                # Sample from Beta(α, β) on [0,1]
                w_raw = np.random.beta(alpha_w, beta_w, (self.n, 1))
                # Scale to [-w, w]
                w_scaled = w_raw * 2 * self.w - self.w

                # Stack the two parameters
                theta = np.hstack((theta, w_scaled))

                xs = self.phi_inverse(theta)
                sample[:, 0:(self.d + 1)] = xs
                return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(np.zeros(self.p), self.sigma ** 2 * np.eye(self.p),size=self.n)
            Js = self.J_phi_inverse(theta)
            us = np.random.uniform(0, 1, self.n)
            ps = densities * Js / M

            for k in range(self.n):
                if us[k] <= ps[k]:
                    sample[count, 0:(self.d + 1)] = xs[k, :]
                    count += 1
                    if count == self.n:
                        break
        #print(sample)
        # Apply rotation and translation to the samples
        return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(
            np.zeros(self.p), self.sigma ** 2 * np.eye(self.p), size=self.n
        )
        

class TorusSampler:
    def __init__(self, n=3000, d=2, p=3, R=3, r=1, sigma=0.0, seed=None):
        self.n = n
        self.d = d  # Dimension of the manifold (Torus is 2D)
        self.p = p  # Embedding dimension (typically 3 for 3D space)
        self.R = R  # Major radius of the Torus
        self.r = r  # Minor radius of the Torus
        self.sigma = sigma

        if seed is not None:
            np.random.seed(seed)

        # Generate random rotation matrix
        matrix = np.random.randn(p, p)
        Q, T = np.linalg.qr(matrix)
        D = np.diag(np.sign(np.diag(T)))
        Q = Q @ D
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        self.rotation = Q
        self.translation = 100 * skdim.datasets.hyperBall(n=1, d=p, radius=1, random_state=0)

    def f_uniform_density(self, x):
        # Uniform density for the Torus
        return 1 / (4 * math.pi**2 * self.R * self.r)

    def f_nonuniform_density(self, x):
        # Uniform density on the surface of a cylinder
        return np.linalg.norm(x, axis=1)

    def phi_inverse(self, theta):
        """
        Parametrization of the Torus in 3D space.
        theta: Array of shape (n, 2), where the first column is the angle around the major circle and the second column is the angle around the minor circle.
        """
        n = theta.shape[0]
        x = np.zeros((n, self.d + 1))

        # Parametrize the Torus
        sin_values_u = np.sin(theta[:, 0])
        cos_values_u = np.cos(theta[:, 0])

        sin_values_v = np.sin(theta[:, 1])
        cos_values_v = np.cos(theta[:, 1])

        x[:, 0] = (self.R + self.r * cos_values_v) * cos_values_u
        x[:, 1] = (self.R + self.r * cos_values_v) * sin_values_u
        x[:, 2] = self.r * sin_values_v

        return x

    def J_phi_inverse(self, theta):
        """
        Jacobian of the inverse map for the Torus parametrization.
        """
        cos_values_v = np.cos(theta[:, 1])
        Js = self.r**2 * (self.R + self.r * cos_values_v)**2
        return Js

    def sample(self, M=None, uniform=True):
        if M is None:
            if uniform:
                M = self.f_uniform_density(None) / 2 / math.pi * 1.1  # heuristic
            else:
                M = 1

        count = 0
        sample = np.zeros((self.n, self.p))

        while count < self.n:
            # Sample theta uniformly over the parameter space
            theta = np.random.uniform(0, 2 * math.pi, (self.n, 2))

            xs = self.phi_inverse(theta)
            if uniform:
                densities = self.f_uniform_density(None) * np.ones(self.n)
            else:
            # Beta parameters for both dimensions (adjust as needed)
                alpha_1, beta_1 = 2.0, 5.0  # skewed toward 0 for first dim
                alpha_2, beta_2 = 3.0, 3.0  # symmetric for second dim

                # Sample from Beta and scale to [0, 2*pi]
                theta_1 = np.random.beta(alpha_1, beta_1, (self.n, 1)) * 2 * math.pi
                theta_2 = np.random.beta(alpha_2, beta_2, (self.n, 1)) * 2 * math.pi

                # Combine
                theta = np.hstack((theta_1, theta_2))

                xs = self.phi_inverse(theta)
                sample[:, 0:(self.d + 1)] = xs
                return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(np.zeros(self.p), self.sigma ** 2 * np.eye(self.p),size=self.n)
            Js = self.J_phi_inverse(theta)
            us = np.random.uniform(0, 1, self.n)
            ps = densities * Js / M

            for k in range(self.n):
                if us[k] <= ps[k]:
                    sample[count, 0:(self.d + 1)] = xs[k, :]
                    count += 1
                    if count == self.n:
                        break

        # Apply rotation and translation to the samples
        return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(
            np.zeros(self.p), self.sigma ** 2 * np.eye(self.p), size=self.n
        )
        
class HyperbolicSurfaceSampler:
    def __init__(self, n=1000, d=2, p=3, a=2, b=2, c=1, h=1, sigma=0.0, seed=None):
        self.n = n
        self.d = d  # Dimension of the manifold (Hyperbolic surface is 2D)
        self.p = p  # Embedding dimension (typically 3 for 3D space)
        self.a = a  # Parameter a (scale along x-axis)
        self.b = b  # Parameter b (scale along y-axis)
        self.c = c  # Parameter c (scale along z-axis)
        self.h = h  # Hyperbolic surface height scale
        self.sigma = sigma  # Optional sigma (though not used in the code above)

        if seed is not None:
            np.random.seed(seed)

        # Generate random rotation matrix
        matrix = np.random.randn(p, p)
        Q, T = np.linalg.qr(matrix)
        D = np.diag(np.sign(np.diag(T)))
        Q = Q @ D
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        self.rotation = Q
        self.translation = 100 * skdim.datasets.hyperBall(n=1, d=p, radius=1, random_state=0)

    def f_uniform_density(self, x):
        # Uniform density for the hyperbolic surface
        return 1  # This is the simplification, you can adjust it based on your specific needs

    def f_nonuniform_density(self, x):
        # Uniform density on the surface of a cylinder
        return np.linalg.norm(x, axis=1)
    
    def phi_inverse(self, theta):
        """
        Parametrization of the Hyperbolic surface in 3D space.
        theta: Array of shape (n, 2), where the first column is the angle around the first axis and the second column is the hyperbolic angle.
        """
        n = theta.shape[0]
        x = np.zeros((n, self.d + 1))

        # Parametrize the Hyperbolic surface
        sin_values = np.sin(theta[:, 0])
        cos_values = np.cos(theta[:, 0])

        sinh_values = np.sinh(theta[:, 1])
        cosh_values = np.cosh(theta[:, 1])

        x[:, 0] = self.a * cosh_values * cos_values
        x[:, 1] = self.b * cosh_values * sin_values
        x[:, 2] = self.c * sinh_values

        return x

    def J_phi_inverse(self, theta):
        """
        Jacobian of the inverse map for the Hyperbolic surface parametrization.
        """
        sin_values = np.sin(theta[:, 0])
        cos_values = np.cos(theta[:, 0])

        sinh_values = np.sinh(theta[:, 1])
        cosh_values = np.cosh(theta[:, 1])
        
        Js = cosh_values ** 2 * (self.a ** 2 * sin_values ** 2 + self.b ** 2 * cos_values ** 2) * \
             (sinh_values ** 2 * (self.a ** 2 * cos_values ** 2 + self.b ** 2 * sin_values ** 2) + self.c ** 2 * cosh_values ** 2)
    
        return Js

    def sample(self, M=None, uniform=True):
        if M is None:
            if uniform:
                M = self.f_uniform_density(None) / 2 / math.pi / 2 / self.h * 1.1  # heuristic
            else:
                M = 1

        count = 0
        sample = np.zeros((self.n, self.p))

        while count < self.n:
            # Sample theta uniformly over the parameter space
            theta = np.random.uniform(0, 2 * math.pi, (self.n, 1))
            theta = np.hstack((theta, np.random.uniform(-self.h, self.h, (self.n, 1))))

            xs = self.phi_inverse(theta)
            if uniform:
                densities = self.f_uniform_density(None) * np.ones(self.n)
            else:
            # Beta parameters — tweak to control shape/skewness
                alpha_theta, beta_theta = 2.0, 2.0  # symmetric Beta for angle
                alpha_h, beta_h = 3.0, 5.0           # skewed Beta for second param

                # Sample from Beta on [0,1] and scale to [0, 2π]
                theta_1 = np.random.beta(alpha_theta, beta_theta, (self.n, 1)) * 2 * math.pi

                # Sample from Beta on [0,1] and scale to [-h, h]
                # Note: Scale [0,1] -> [-h,h] by: val * (2h) - h
                theta_2 = np.random.beta(alpha_h, beta_h, (self.n, 1)) * 2 * self.h - self.h

                # Stack parameters
                theta = np.hstack((theta_1, theta_2))

                xs = self.phi_inverse(theta)
                sample[:, 0:(self.d + 1)] = xs
                return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(np.zeros(self.p), self.sigma ** 2 * np.eye(self.p),size=self.n)
            Js = self.J_phi_inverse(theta)
            us = np.random.uniform(0, 1, self.n)
            ps = densities * Js / M

            for k in range(self.n):
                if us[k] <= ps[k]:
                    sample[count, 0:(self.d + 1)] = xs[k, :]
                    count += 1
                    if count == self.n:
                        break

        # Apply rotation and translation to the samples
        return sample @ self.rotation.T + self.translation + np.random.multivariate_normal(
            np.zeros(self.p), self.sigma ** 2 * np.eye(self.p), size=self.n
        )