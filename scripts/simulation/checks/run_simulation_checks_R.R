# Compare the R reference estimators with Python using shared simulated data.
# Runs 2Stage and 3Stage on one saved dataset per case and exports predictions.
# Estimator functions are adapted from Wang and Li's (2013) reference code.
# Run from the repository root after generating the shared sample CSVs.

for (pkg in c("quantreg", "doParallel", "foreach")) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}
library(quantreg)
library(doParallel)
library(foreach)

# ============================================================
# Estimator functions adapted from the authors' R reference implementation.
# Includes residual rounding and diagnostic prints used for comparison.
# ============================================================

PowT.1tau.func <- function(y, x, tau, lams = seq(-2, 2, 0.1), a) {
  n <- length(y)
  compare.x <- diag(n)
  for (i in 1:n) {
    for (j in 1:n) {
      compare.x[i, j] <- prod(x[i, ] < x[j, ])
    }
  }
  Vn = bhat <- NULL
  for (lam in lams) {
    if (lam == 0) { Lam.y <- log(y + a) } else { Lam.y <- ((y + a)^lam - 1) / lam }
    idx.keep <- which(!is.na(Lam.y))
    Lam.y <- Lam.y[idx.keep]
    x2 <- x[idx.keep, ]
    fit <- rq(Lam.y ~ x2, tau)
    res <- fit$res
    res <- round(res, 10)
    bhat <- rbind(bhat, fit$coef)
    score <- tau - 1 * (res <= 0)
    Rn <- compare.x[idx.keep, idx.keep] * score
    Rn <- apply(Rn, 2, sum) / n
    Vn <- c(Vn, mean(Rn^2))
    cat(sprintf("lambda=%.1f  score=%.6f\n", lam, Vn[length(Vn)]))
    if (lam == -1) {
      cat("First 10 residuals at lambda=-1:\n")
      print(round(res[1:10], 8))
    }
    
  }
  idx <- order(Vn)[1]
  lam <- lams[idx]
  coef <- bhat[idx, ]
  return(list(lam = lam, coef = bhat))
}

qpareto = function(p, gamma) { (1 - p)^(-gamma) }

rpareto = function(n, gamma) {
  u = runif(n, 0, 1)
  (1 - u)^(-gamma)
}

# Est EVI gamma on original scale
# input: est intermediate conditional q on normal scale
EVI.CFG.func = function(x, tol = 1e-4, min.prop = 0.3, taus) {
  tt = mean(diff(x) > tol, na.rm = T)
  if (!is.na(tt) & tt > min.prop) {
    idx = which(x > 0 & !is.na(x))
    f = x[idx]
    taus2 = taus[idx]
    tmp = plot(rearrange(stepfun(taus2[-length(taus2)], f), xmax = 1))
    x = tmp$y
    out = mean(log(x / x[1]))
  } else out = NA
  return(out)
}

TwoStage = function(y, x, xstar, tau.e, k, tol = 1e-4) {
  n = length(y)
  tau = 1 - k / n
  p = ncol(x)
  nx = length(xstar) / p
  max.tau = (n - as.integer(n^(0.1))) / (n + 1)
  taus = seq(tau, max.tau, length = k)
  rq1 = rq(y ~ x, taus)
  Q = cbind(1, xstar) %*% rq1$coef
  pdf(file = NULL)
  gamma.x = apply(Q, 1, EVI.CFG.func, tol = tol, taus = taus)
  dev.off()
  Q2Stage = t(outer(((1 - tau) / (1 - tau.e)), (gamma.x), "^")) * Q[, 1]
  out = list(Q2Stage = Q2Stage, gamma.x = gamma.x)
  return(out)
}

ThreeStage = function(y, x, xstar, tau.e, grid.lam = seq(-2, 2, 0.1), grid.k, tau.lam, a = 0, tol = 1e-4) {
  x = as.matrix(x)
  n = length(y)
  p = ncol(x)
  nx = length(xstar) / p
  max.tau = (n - as.integer(n^(0.1))) / (n + 1)
  
  if (length(grid.lam) > 1) {
    tmp = PowT.1tau.func(y, x, tau = tau.lam, lams = grid.lam, a)
    lam = tmp$lam
  } else if (length(grid.lam) == 1) lam = grid.lam
  
  if (lam == 0) { Lam.y = log(y + a) } else { Lam.y = ((y + a)^lam - 1) / lam }
  
  if (length(grid.k) == 1) k = grid.k
  else if (length(grid.k) > 1)
    k = select.k.func(y = y, x = x, Lam.y = Lam.y, lam = lam, a = a, max.tau = max.tau, grid.k = grid.k, n = n)
  
  
  ########## Obtain conditional quantiles of Y at the intermediate levels taus
  tau = 1 - k / n
  taus = seq(tau, max.tau, length = k)
  
  rq1 = rq(Lam.y ~ x, taus)
  Lam.Q = cbind(1, xstar) %*% rq1$coef
  
  ### calculate EVI on the original scale
  tt = est.gamma.func(taus = taus, Lam.Q, lam, a, tol)
  
  gamma.x = tt$gamma.x
  
  Q = tt$Q
  
  cgamma = mean(gamma.x, na.rm = T)
  
  ########## extrapolation to the extreme tails tau.e
  Q3Stage  = t(outer(((1 - tau) / (1 - tau.e)), (gamma.x), "^")) * Q[1:nx, 1]
  Q3StageP = outer(Q[1:nx, 1], (((1 - tau) / (1 - tau.e))^cgamma), "*")
  
  out = list(lam = lam, k = k, Q3Stage = Q3Stage, Q3StageP = Q3StageP, gamma.x = gamma.x, cgamma = cgamma)
  return(out)
}

select.k.func <- function(y, x, Lam.y, lam, a, max.tau, grid.k, n) {
  obj = NULL
  grid.k = grid.k[grid.k > as.integer((1 - max.tau) * n) + 1]
  for (k in grid.k) {
    tau = 1 - k / n
    taus = seq(tau, max.tau, length = k)
    rq1 = rq(Lam.y ~ x, taus)
    Lam.Q = cbind(1, x) %*% rq1$coef
    if (lam == 0) { Q = exp(Lam.Q) - a } else { Q = (Lam.Q * lam + 1)^(1 / lam) - a }
    gamma.x      = apply(Q,     1, function(x) { x = sort(x[x > 0]); mean(log(x / x[1])) })
    Mn1          = apply(Lam.Q, 1, function(x) { x = sort(x[x > 0]); mean(log(x / x[1])) })
    Mn2          = apply(Lam.Q, 1, function(x) { x = sort(x[x > 0]); mean((log(x / x[1]))^2) })
    gamma.minus  = 1 - 0.5 / (1 - Mn1^2 / Mn2)
    gamma.star.x = Mn1 + gamma.minus
    obj = c(obj, mean((lam * gamma.x - gamma.star.x)^2, na.rm = T))
  }
  idx = which.min(obj)
  k = grid.k[idx]
  return(k)
}

# Est EVI gamma on original scale
# input: est intermediate conditional q on transformed scale
est.gamma.func = function(taus, Lam.Q, lam, a = 0, tol) {
  if (lam == 0) { Q = exp(Lam.Q) - a }
  else if (lam < 0) {
    tt = Lam.Q * lam + 1
    tol2 = max(tol, quantile(as.vector(tt[tt > 0]), 0.002, na.rm = T))
    tt2 = 1 * (abs(tt[, 1]) < tol)
    tt[which(tt2 == 1), ] = NA
    tt3 = rowMeans(tt < tol2)
    tt[which(tt3 > 0.3)] = NA
    tt[which(tt < tol2)] = NA
    Q = tt^(1 / lam) - a
  } else if (lam > 0) {
    tt = Lam.Q * lam + 1
    tt2 = 1 * (abs(tt[, 1]) < tol)
    tt[which(tt2 == 1), ] = NA
    tt3 = rowMeans(tt < tol)
    tt[which(tt3 > 0.3)] = NA
    tt[which(tt < tol)] = NA
    Q = tt^(1 / lam) - a
  }
  pdf(file = NULL)
  gamma.x = apply(Q, 1, EVI.CFG.func, tol = tol, taus = taus)
  dev.off()
  return(list(gamma.x = gamma.x, Q = Q))
}

# ============================================================
# HELPERS
# ============================================================

box_cox_inv = function(lam, z) {
  if (lam == 0) exp(z) else (lam * z + 1)^(1 / lam)
}






project_dir = getwd()

cases = 1:6
sample_size = 2000
mc_id = 0
tau_n = 0.99

out_dir = file.path(project_dir, "outputs", "implementation_checks")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)


for (case in cases) {
  start = proc.time()[["elapsed"]]
  
  data_path = file.path(
    project_dir, "data", "raw",
    sprintf("mc_samples_case%d_n%d.csv", case, sample_size)
  )
  
  cat("\n", strrep("=", 60), "\nCASE ", case,
      ": loading ", basename(data_path), "\n", sep = "")
  flush.console()
  
  df = read.csv(data_path)
  
  if ("mc" %in% names(df)) {
    df = df[df$mc == mc_id, , drop = FALSE]
  }
  if ("obs" %in% names(df)) {
    df = df[order(df$obs), , drop = FALSE]
  }
  
  if (nrow(df) != sample_size) {
    stop(sprintf(
      "Case %d: expected %d rows, got %d.",
      case, sample_size, nrow(df)
    ))
  }
  
  covariates = intersect(c("x1", "x2"), names(df))
  if (length(covariates) == 0) {
    stop("No covariates found.")
  }
  
  y = df$y
  x = as.matrix(df[, covariates, drop = FALSE])
  n = length(y)
  obs = if ("obs" %in% names(df)) df$obs else seq_len(n) - 1
  
  if (n == 500) {
    grid_k = 20:70
    tau_lam = 0.85
  } else if (n == 2000) {
    grid_k = 30:100
    tau_lam = 0.90
  } else {
    stop("This check is configured for n=500 or n=2000.")
  }
  
  # Match the response shift used by the Python pipeline.
  a = max(0, -min(y) + 1e-6)
  
  cat("Running 3Stage...\n")
  flush.console()
  
  out3 = ThreeStage(
    y, x, xstar = x,
    tau.e = tau_n,
    grid.lam = round(seq(-2, 2, 0.1), 1),
    grid.k = grid_k,
    tau.lam = tau_lam,
    a = a
  )
  
  cat("Running 2Stage...\n")
  flush.console()
  
  k2 = as.integer(round(4.5 * n^(1 / 3)))
  out2 = TwoStage(y, x, xstar = x, tau.e = tau_n, k = k2)
  
  q3 = as.numeric(out3$Q3Stage)
  gamma3 = as.numeric(out3$gamma.x)
  q2 = as.numeric(out2$Q2Stage)
  gamma2 = as.numeric(out2$gamma.x)
  
  # Recover bases using the original R reference thresholds.
  base3 = q3 / ((out3$k / n) / (1 - tau_n))^gamma3
  base2 = q2 / ((k2 / n) / (1 - tau_n))^gamma2
  
  results = data.frame(
    case = case,
    mc = mc_id,
    obs = obs,
    lambda_3stage = out3$lam,
    k_3stage = out3$k,
    k_2stage = k2,
    gamma_3stage = gamma3,
    q_base_3stage = base3,
    q_hat_3stage = q3,
    gamma_2stage = gamma2,
    q_base_2stage = base2,
    q_hat_2stage = q2
  )
  
  out_path = file.path(
    out_dir, sprintf("r_case%d_n%d_mc%d.csv", case, n, mc_id)
  )
  write.csv(results, out_path, row.names = FALSE)
  
  cat("\n3Stage: lambda=", out3$lam, ", k=", out3$k, "\n", sep = "")
  cat("2Stage: k=", k2, "\n", sep = "")
  cat("3Stage mean gamma:",
      mean(gamma3[is.finite(gamma3)]), "\n")
  cat("2Stage mean gamma:",
      mean(gamma2[is.finite(gamma2)]), "\n")
  cat("3Stage finite predictions:", sum(is.finite(q3)), "/", n, "\n")
  cat("2Stage finite predictions:", sum(is.finite(q2)), "/", n, "\n")
  
  print(results, row.names = FALSE, digits = 10,
        max = nrow(results) * ncol(results) + 100)
  
  cat("\nSaved:", out_path, "\n")
  cat("Runtime:", proc.time()[["elapsed"]] - start, "seconds\n")
  flush.console()
}