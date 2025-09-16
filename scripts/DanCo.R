library(RANN)
library(doParallel)
library(foreach)

# Vectorized and faster danco_part1_fast
danco_part1_fast <- function(X, neighbors, N, D, k) {
  # Get the coordinates of neighbors: N x k x D array
  nb_coords <- array(0, c(N, k, D))
  for (dim_i in 1:D) {
    nb_coords[,,dim_i] <- X[neighbors[,1:k], dim_i]
  }
  
  # Calculate distances between each point and its neighbors
  # Vectorized dist: difference along dims, squared, summed, sqrt
  diffs <- array(0, c(N, k))
  for (dim_i in 1:D) {
    diffs <- diffs + (matrix(X[,dim_i], N, k) - nb_coords[,,dim_i])^2
  }
  dists <- sqrt(diffs)
  
  # Compute min(dists / max(dists)) for each row n
  # max and min over neighbors
  vec.rho <- apply(dists, 1, function(d) min(d / max(d)))
  
  funmax <- function(d) {
    term1 <- N * log(k * d)
    term2 <- (d - 1) * sum(log(vec.rho))
    term3 <- (k - 1) * sum(log(1 - (vec.rho^d)))
    term1 + term2 + term3
  }
  
  as.double(stats::optimize(funmax, lower = 1, upper = D, maximum = TRUE)$maximum)
}

# Pre-allocate thetas and vectorize danco_part2
danco_part2 <- function(X, neighbors, N, D, k) {
  thetas <- matrix(0, nrow = N, ncol = k*(k-1)/2)  # number of unique pairs
  
  for (n in 1:N) {
    idx <- neighbors[n, 1:k]
    nb <- X[idx, , drop = FALSE]
    nbctr <- nb - matrix(X[n, ], nrow = k, ncol = ncol(X), byrow = TRUE)
    thetas[n, ] <- danco_part2_theta(nbctr)
  }
  
  # Vectorize angle calculations
  cs <- rowSums(cos(thetas))
  ss <- rowSums(sin(thetas))
  m <- ncol(thetas)
  eta <- sqrt((cs / m)^2 + (ss / m)^2)
  nu_hat <- atan2(ss, cs)
  
  Ainv <- function(eta) {
    ifelse(eta < 0.53,
           2 * eta + eta^3 + 5 * (eta^5) / 6,
           ifelse(eta < 0.85,
                  -0.4 + 1.39 * eta + 0.43 / (1 - eta),
                  1 / ((eta^3) - 4 * (eta^2) + 3 * eta)))
  }
  
  tau_hat <- Ainv(eta)
  
  c(mean(nu_hat), mean(tau_hat))
}

# Same function but a little more efficient by avoiding redundant checks
danco_part2_theta <- function(X) {
  k <- nrow(X)
  norms <- sqrt(rowSums(X^2))
  output <- matrix(0, k, k)
  for (i in 1:(k - 1)) {
    tgti <- X[i, ]
    nrmi <- norms[i]
    for (j in (i + 1):k) {
      tgtj <- X[j, ]
      nrmj <- norms[j]
      denom <- nrmi * nrmj
      val <- if (denom == 0) 1 else sum(tgti * tgtj) / denom
      val <- max(-1, min(1, val))
      output[i, j] <- output[j, i] <- acos(val)
    }
  }
  output[upper.tri(output)]
}

danco_cost <- function(k, dML, ddML, muvvv, mutau, dvvv, dtau){
  # Distance KL (matches Python _KLd)
  Hk  = sum(1/(1:k))
  quo = ddML / dML
  i   = 0:k
  a   = ((-1)^i) * choose(k, i) * psigamma(1 + i/quo, deriv = 0)
  KLd = Hk * quo - log(quo) - (k - 1) * sum(a)
  
  # Angle KL (matches Python _KLnutau)
  I0 <- function(x) besselI(x, 0)
  I1 <- function(x) besselI(x, 1)
  v1 = muvvv; v2 = dvvv
  tau1 = mutau; tau2 = dtau
  I0_tau1 = ifelse(I0(tau1) == 0, .Machine$double.xmin, I0(tau1))
  I0_tau2 = ifelse(I0(tau2) == 0, .Machine$double.xmin, I0(tau2))
  KLangle = log(I0_tau2) - log(I0_tau1) + (I1(tau1) / I0_tau1) * (tau1 - tau2 * cos(v2 - v1))
  
  as.numeric(KLd + KLangle)
}

# In est.danco, reuse cluster and register it once, reduce number of cores if too many
est.danco <- function(X, k = 5, fractal = FALSE, spline_method = c("monoH.FC", "natural"), max_p = NULL) {
  spline_method <- match.arg(spline_method)
  
  if (!requireNamespace("RANN", quietly = TRUE)) stop("Please install the 'RANN' package.")
  if (!requireNamespace("foreach", quietly = TRUE)) stop("Please install the 'foreach' package.")
  if (!requireNamespace("doParallel", quietly = TRUE)) stop("Please install the 'doParallel' package.")
  
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.numeric(X)) stop("X must be numeric.")
  
  N <- nrow(X)
  D_full <- ncol(X)
  
  if (!is.null(max_p)) {
    if (!is.numeric(max_p) || length(max_p) != 1 || max_p < 1 || max_p > D_full) {
      stop("max_p must be a numeric value between 1 and the number of columns of X.")
    }
    D <- min(D_full, max_p)
  } else {
    D <- D_full
  }
  
  k <- round(k)
  
  neighbors <- RANN::nn2(X, k = k + 1)$nn.idx[, -1]
  
  data.dML <- danco_part1_fast(X, neighbors, N, D, k)
  # cat("data.dML =", data.dML, "\n")
  muvec <- danco_part2(X, neighbors, N, D, k)
  # cat("muvec =", muvec, "\n")
  
  if (any(!is.finite(muvec))) {
    warning("Non-finite values in muvec. Returning NA.")
    return(list(estdim_int = NA_integer_, estdim = NA_real_))
  }
  
  data.vvv <- muvec[1]
  data.tau <- muvec[2]
  
  cores <- max(1, parallel::detectCores(logical = TRUE) - 1)  # leave one core free
  cl <- parallel::makeCluster(cores)
  doParallel::registerDoParallel(cl)
  
  boot_res <- foreach(d = 1:D, .combine = rbind, 
                      .packages = c("RANN"), 
                      .export = c("danco_part1_fast", "danco_part2", "danco_part2_theta")) %dopar% {
                        Z <- matrix(rnorm(N * d), ncol = d)
                        Z <- Z / sqrt(rowSums(Z^2))
                        r <- runif(N)^(1/d)
                        Y <- Z * r
                        
                        if (d == 1) Y <- cbind(Y, rep(0, N))
                        
                        nbs <- RANN::nn2(Y, k = k + 1)$nn.idx[, -1]
                        ddML <- danco_part1_fast(Y, nbs, N, ncol(Y), k)
                        muvec <- danco_part2(Y, nbs, N, ncol(Y), k)
                        c(ddML, muvec[1], muvec[2], d)
                      }
  
  parallel::stopCluster(cl)
  
  boot.dML <- boot_res[, 1]
  boot.vvv <- boot_res[, 2]
  boot.tau <- boot_res[, 3]
  
  dims_checked <- boot_res[, 4]
  cat("Completed dimensions: ", paste(dims_checked, collapse = ", "), "\n")
  
  d.cost <- sapply(1:D, function(d) danco_cost(k,
                                               data.dML, boot.dML[d],
                                               data.vvv, data.tau,
                                               boot.vvv[d], boot.tau[d]))
  d.fin <- which.min(d.cost)
  
  est_frac <- NA_real_
  if (isTRUE(fractal)) {
    dims <- which(is.finite(d.cost))
    if (length(dims) >= 3) {
      if (D == 2) {
        f <- stats::approxfun(x = dims, y = d.cost[dims], method = "linear", rule = 2)
        opt <- optimize(f, interval = c(1, D + 1))
        est_frac <- as.numeric(opt$minimum)
      } else {
        f <- stats::splinefun(x = dims, y = d.cost[dims], method = "fmm")
        opt <- optimize(f, interval = c(1, D + 1))
        est_frac <- as.numeric(opt$minimum)
      }
    } else {
      est_frac <- as.numeric(d.fin)
    }
  }
  
  list(
    estdim_int = as.integer(d.fin),
    estdim = if (isTRUE(fractal)) est_frac else as.integer(d.fin)
  )
}
