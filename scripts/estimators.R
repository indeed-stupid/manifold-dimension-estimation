# =============================================
# Intrinsic Dimension Estimation Experiments
# =============================================

rm(list = ls())
setwd("C:/Users/z5281286/Desktop/project 01-02")

# --- Load libraries ---
library(R.matlab)   # for reading .mat files
library(Rdimtools)  # for various dimension estimators
library(FNN)        # for fast nearest neighbors
library(stats)      # for PCA (prcomp)
library(transport)  # for optimal transport / Wasserstein distance
library(parallel)
library(future.apply)
library(RSpectra)
library(Rcpp)

source("DanCo.R")

# ------------------------------------------------
# Function: Algorithm 2
# ------------------------------------------------

find_stable_K_range <- function(d_hat) {
  k_max <- length(d_hat)
  s_min <- Inf
  s_max <- 0
  k_star <- 0
  k1 <- 1
  k2 <- 3
  
  for (k in 1:(k_max - 2)) {
    s <- sd(d_hat[k:(k + 2)])
    if (s < s_min) {
      s_min <- s
      k_star <- k
    }
    if (s > s_max) {
      s_max <- s
    }
  }
  
  if (s_max > 1.25 * s_min) {
    k1 <- k_star
    k2 <- k_star + 2
  } else {
    k1 <- 1
    k2 <- k_max
  }
  
  return(c(k1, k2))
}


# ------------------------------------------------
# Function: Sample points on a d-sphere embedded in R^p
# ------------------------------------------------
sample_on_sphere <- function(n, d, p) {
  if (p < d + 1) stop("p must be at least d + 1")
  
  mat <- matrix(rnorm(n * (d + 1)), nrow = n, ncol = d + 1)
  norms <- sqrt(rowSums(mat^2))
  sphere_points <- mat / norms  # project to unit sphere
  
  if (p > d + 1) {
    padding <- matrix(0, nrow = n, ncol = p - (d + 1))
    sphere_points <- cbind(sphere_points, padding)
  }
  
  return(sphere_points)
}

Rcpp::sourceCpp("tle.cpp")

TLE_fast <- function(X, K) {
  if (!requireNamespace("FNN", quietly = TRUE)) {
    stop("Please install the FNN package: install.packages('FNN')")
  }
  
  if (!is.matrix(X)) {
    X <- as.matrix(X)
  }
  
  knn_res <- FNN::get.knn(X, k = K)
  neighbors_idx <- knn_res$nn.index
  
  result <- TLE_cpp(X, neighbors_idx, K)
  return(result)
}

# ------------------------------------------------
# Estimator: Local PCA-based intrinsic dimension
# ------------------------------------------------
est_local_pca <- function(X, k = 10) {
  n <- nrow(X)
  dims <- numeric(n)
  knn <- get.knn(X, k = k)
  
  for (i in 1:n) {
    neighbors_idx <- c(i, knn$nn.index[i, ])
    neighbors <- X[neighbors_idx, , drop = FALSE]
    
    # Center neighbors
    # neighbors_centered <- scale(neighbors, center = TRUE, scale = FALSE)
    # cov_mat <- cov(neighbors_centered)
    
    # Use SVD (numerically stable)
    # k_eff <- min(ncol(X), k + 1)
    # sv <- svds(cov_mat, k = k_eff, nu = 0, nv = 0)
    # eig_vals <- sv$d
    
    pca_res <- prcomp(neighbors, center = TRUE, scale. = FALSE)
    eig_vals <- pca_res$sdev^2
    
    # Threshold: eigenvalue > 5% of the largest
    threshold <- 0.05 * max(eig_vals)
    dims[i] <- sum(eig_vals > threshold)
    
    print(dims[i])
  }
  
  return(mean(dims))
}

# ------------------------------------------------
# Estimator: Curvature-Aware PCA (CAPCA)
# ------------------------------------------------
CAPCA <- function(p, n, K, sample) {
  if (p > K) {
    p <- K
  }
  
  dimensions <- numeric(n)
  knn_res <- get.knn(sample, k = K + 2)
  indices <- knn_res$nn.index
  
  for (k in 1:n) {
    neighbors_idx <- indices[k, 2:(K + 1)]
    neighbors <- sample[neighbors_idx, , drop = FALSE]
    
    r1 <- sqrt(sum((sample[indices[k, K + 1], ] - sample[k, ])^2))
    r2 <- sqrt(sum((sample[indices[k, K + 2], ] - sample[k, ])^2))
    R <- (r1 + r2) / 2  # average distance for local scaling
    
    center <- colMeans(neighbors)
    diff <- sweep(neighbors, 2, center)
    
    pca_res <- prcomp(diff, center = FALSE, scale. = FALSE)
    eigenvalues <- (pca_res$sdev)^2
    
    if (length(eigenvalues) < p) {
      eigenvalues <- c(eigenvalues, rep(0, p - length(eigenvalues)))
    }
    
    adjusted_errors <- numeric(p)
    
    for (q in 1:p) {
      curvature_term <- ((3 * q + 4) / (q * (q + 4))) * sum(eigenvalues[(q+1):length(eigenvalues)])
      squared_error_sum <- 0
      
      for (j in 1:q) {
        realised <- (eigenvalues[j] + curvature_term) / (R^2)
        target <- 1 / (q + 2)
        squared_error_sum <- squared_error_sum + (target - realised)^2
      }
      
      norm_error <- sqrt(squared_error_sum)
      penalty <- 2 * sum(eigenvalues[(q+1):length(eigenvalues)]) / (R^2)
      adjusted_errors[q] <- norm_error + penalty
    }
    
    dimensions[k] <- which.min(adjusted_errors)
    print(dimensions[k])
  }
  
  return(mean(dimensions))
}


# ------------------------------------------------
# Estimator: Wasserstein-based (Wasserstein_new)
# ------------------------------------------------
wasserstein_single_run <- function(data, n, alpha, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  n1 <- floor(n / (2 + 2 * alpha))
  n2 <- floor(alpha * n1)
  
  indices_all <- sample(1:n, n)
  
  subsample1 <- data[indices_all[1:n1], , drop = FALSE]
  subsample2 <- data[indices_all[(n1 + 1):(2 * n1)], , drop = FALSE]
  subsample3 <- data[indices_all[(2 * n1 + 1):(2 * n1 + n2)], , drop = FALSE]
  subsample4 <- data[indices_all[(2 * n1 + n2 + 1):(2 * n1 + 2 * n2)], , drop = FALSE]
  
  a <- rep(1 / n1, n1)
  b <- rep(1 / n1, n1)
  
  M1 <- as.matrix(dist(rbind(subsample1, subsample2)))[1:n1, (n1 + 1):(2 * n1)]
  transport_plan1 <- transport::transport(a, b, costm = M1, method = "revsimplex")
  W1 <- sum(transport_plan1$mass * M1[cbind(transport_plan1$from, transport_plan1$to)])
  
  a <- rep(1 / n2, n2)
  b <- rep(1 / n2, n2)
  
  M2 <- as.matrix(dist(rbind(subsample3, subsample4)))[1:n2, (n2 + 1):(2 * n2)]
  transport_plan2 <- transport::transport(a, b, costm = M2, method = "revsimplex")
  W2 <- sum(transport_plan2$mass * M2[cbind(transport_plan2$from, transport_plan2$to)])
  
  W1 <- max(W1, 1e-12)
  W2 <- max(W2, 1e-12)
  
  denominator <- log(W1) - log(W2)
  if (abs(denominator) < 1e-12) {
    denominator <- ifelse(denominator >= 0, 1e-12, -1e-12)
  }
  
  est <- log(alpha) / denominator
  return(est)
}

Wasserstein_new <- function(p, n, data, alpha) {
  seeds <- 321:(321 + 99)
  results <- numeric(length(seeds))
  
  for (i in seq_along(seeds)) {
    results[i] <- wasserstein_single_run(data, n, alpha, seed = seeds[i])
    print(results[i])
  }
  
  return(median(results))
}

# ------------------------------------------------
# Estimator: Tight Locality Estimator (TLE)
# ------------------------------------------------
TLE <- function(X, K) {
  n <- nrow(X)
  p <- ncol(X)
  knn_res <- get.knn(X, k = K)
  neighbors_idx <- knn_res$nn.index
  
  d_hat <- numeric(n)
  
  for (k in 1:n) {
    x_k <- X[k, ]
    neighbors <- X[neighbors_idx[k, ], , drop = FALSE]
    R <- sqrt(sum((x_k - neighbors[K, ])^2))
    
    sum_terms <- 0
    count <- 0
    
    for (i in 1:K) {
      v <- neighbors[i, ]
      xk_minus_v <- x_k - v
      norm_xk_minus_v_sq <- sum(xk_minus_v^2)
      
      for (j in 1:K) {
        if (j == i) next
        w <- neighbors[j, ]
        w_minus_v <- w - v
        dot_xkv_wv <- sum(xk_minus_v * w_minus_v)
        
        # Compute d(v, w)
        if (abs(R^2 - norm_xk_minus_v_sq) < 1e-12) {
          if (abs(dot_xkv_wv) < 1e-12) next
          rad <- (R * sum(w_minus_v^2)) / (2 * dot_xkv_wv)
        } else {
          denom <- R^2 - norm_xk_minus_v_sq
          u <- (R * w_minus_v) / denom
          u_dot_xkv <- sum(u * xk_minus_v)
          u_dot_wv <- sum(u * w_minus_v)
          inner <- u_dot_xkv^2 + R * u_dot_wv
          if (inner < 0) next
          rad <- sqrt(inner) - u_dot_xkv
        }
        
        if (!is.finite(rad) || rad <= 0) next
        
        # Compute d(2x_k - v, w)
        v_reflected <- 2 * x_k - v
        xk_minus_vr <- x_k - v_reflected
        w_minus_vr <- w - v_reflected
        norm_xk_minus_vr_sq <- sum(xk_minus_vr^2)
        dot_xkvr_wvr <- sum(xk_minus_vr * w_minus_vr)
        
        if (abs(R^2 - norm_xk_minus_vr_sq) < 1e-12) {
          if (abs(dot_xkvr_wvr) < 1e-12) next
          rad_reflected <- (R * sum(w_minus_vr^2)) / (2 * dot_xkvr_wvr)
        } else {
          denom2 <- R^2 - norm_xk_minus_vr_sq
          u2 <- (R * w_minus_vr) / denom2
          u_dot_xkvr <- sum(u2 * xk_minus_vr)
          u_dot_wvr <- sum(u2 * w_minus_vr)
          inner2 <- u_dot_xkvr^2 + R * u_dot_wvr
          if (inner2 < 0) next
          rad_reflected <- sqrt(inner2) - u_dot_xkvr
        }
        
        if (!is.finite(rad_reflected) || rad_reflected <= 0) next
        
        term <- log(rad / R) + log(rad_reflected / R)
        sum_terms <- sum_terms + term
        count <- count + 1
      }
    }
    
    if (count > 0) {
      d_hat[k] <- -1 / (sum_terms / (count * 2))
    } else {
      d_hat[k] <- NA
    }
  }
  
  return(mean(d_hat, na.rm = TRUE))
}

