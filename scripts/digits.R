source("estimators.R")

# ------------------------------------------------
# Load and process real data from MATLAB file - MNIST
# ------------------------------------------------
# Read the CSV file (skip the header row)
data <- read.csv("mnist_train.csv", header = TRUE)

# Extract labels (first column)
labels <- data[, 1]

# Extract images (all other columns)
images <- data[, -1]

# Select only rows where label == 1
sample <- images[labels == 1, ]

# Get number of samples
n <- nrow(sample)
p <- ncol(sample)

alphas <- c(1.01, 1.2, 1.4, 1.6, 1.8, 2, 4, 6, 8, 10)
Ks <- c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
Ks_DanCo <- c(3, 4, 6, 8, 10, 12, 14, 16, 18, 20)

results <- list()

# Helper function to save partial results
save_partial <- function(results_list, filename = "estimator_results_partial.rds") {
  saveRDS(results_list, filename)
  cat(sprintf("Partial results saved to %s\n", filename))
}


times <- c()

# 1. MADE
cat("Starting MADE estimation...\n")
start_time <- Sys.time()
results$MADE <- sapply(Ks, function(k) mean(est.made(sample, k = k)$estloc))
end_time <- Sys.time()
elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
times <- c(times, elapsed)
save_partial(results)

# 2. MLE1
cat("Starting MLE1 estimation...\n")
start_time <- Sys.time()
results$MLE1 <- sapply(Ks, function(k) est.mle1(sample, k1 = k, k2 = k + 1)$estdim)
end_time <- Sys.time()
elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
times <- c(times, elapsed)
save_partial(results)

# 3. TLE
cat("Starting TLE estimation...\n")
start_time <- Sys.time()
results$TLE <- sapply(Ks, function(k) TLE_fast(sample, K = k))
end_time <- Sys.time()
elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
times <- c(times, elapsed)
save_partial(results)

# 4. LocalPCA
cat("Starting LocalPCA estimation...\n")
start_time <- Sys.time()
results$LocalPCA <- sapply(Ks, function(k) est_local_pca(sample, k = k))
end_time <- Sys.time()
elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
times <- c(times, elapsed)
save_partial(results)

# 5. CAPCA
cat("Starting CAPCA estimation...\n")
start_time <- Sys.time()
results$CAPCA <- sapply(Ks, function(k) CAPCA(p, n, k, sample))
end_time <- Sys.time()
elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
times <- c(times, elapsed)
save_partial(results)

# 6. Wasserstein
cat("Starting Wasserstein estimation...\n")
start_time <- Sys.time()
results$Wasserstein <- sapply(alphas, function(alpha) Wasserstein_new(p, n, sample, alpha))
end_time <- Sys.time()
elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
times <- c(times, elapsed)
save_partial(results)

# 7. DanCo
# cat("Starting DanCo estimation...\n")
# start_time <- Sys.time()
# results$DanCo <- sapply(Ks_DanCo, function(k) est.danco(sample, k = k, max_p = 100)$estdim)
# end_time <- Sys.time()
# elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
# times <- c(times, elapsed)
# save_partial(results)

# 8. TwoNN
cat("Starting TwoNN estimation...\n")
start_time <- Sys.time()
results$TwoNN <- est.twonn(sample)
end_time <- Sys.time()
elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
times <- c(times, elapsed)
save_partial(results)

stable_K_ranges = list()
for (name in names(results)) {
  d_hat <- results[[name]]
  
  if (name == "DanCo" && length(d_hat) == length(Ks_DanCo)) {
    stable_K_ranges[[name]] <- find_stable_K_range(d_hat)
  } else if (name == "Wasserstein" && length(d_hat) == length(alphas)) {
    stable_K_ranges[[name]] <- find_stable_K_range(d_hat)
  } else if (length(d_hat) == length(Ks)) {
    stable_K_ranges[[name]] <- find_stable_K_range(d_hat)
  } else {
    cat(sprintf("Skipping %s (not applicable)\n", name))
  }
}

save(results, file = "digits.RData")
