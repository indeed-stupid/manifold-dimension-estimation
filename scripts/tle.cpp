#include <Rcpp.h>
#include <cmath>
using namespace Rcpp;

// [[Rcpp::export]]
double TLE_cpp(NumericMatrix X, IntegerMatrix neighbors_idx, int K) {
  int n = X.nrow();
  int p = X.ncol();

  NumericVector d_hat(n);

  for (int k = 0; k < n; ++k) {
    NumericVector x_k = X(k, _);
    NumericMatrix neighbors(K, p);

    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < p; ++j) {
        neighbors(i, j) = X(neighbors_idx(k, i) - 1, j);  // R is 1-indexed
      }
    }

    // Compute R
    double R = 0.0;
    for (int j = 0; j < p; ++j) {
      double diff = x_k[j] - neighbors(K - 1, j);
      R += diff * diff;
    }
    R = std::sqrt(R);

    double sum_terms = 0.0;
    int count = 0;

    for (int i = 0; i < K; ++i) {
      NumericVector v = neighbors(i, _);
      NumericVector xk_minus_v(p);
      double norm_xk_minus_v_sq = 0.0;

      for (int j = 0; j < p; ++j) {
        xk_minus_v[j] = x_k[j] - v[j];
        norm_xk_minus_v_sq += xk_minus_v[j] * xk_minus_v[j];
      }

      for (int j = 0; j < K; ++j) {
        if (j == i) continue;

        NumericVector w = neighbors(j, _);
        NumericVector w_minus_v(p);
        for (int d = 0; d < p; ++d) {
          w_minus_v[d] = w[d] - v[d];
        }

        double dot_xkv_wv = 0.0, norm_wv_sq = 0.0;
        for (int d = 0; d < p; ++d) {
          dot_xkv_wv += xk_minus_v[d] * w_minus_v[d];
          norm_wv_sq += w_minus_v[d] * w_minus_v[d];
        }

        double rad = -1.0;
        if (std::abs(R * R - norm_xk_minus_v_sq) < 1e-12) {
          if (std::abs(dot_xkv_wv) < 1e-12) continue;
          rad = (R * norm_wv_sq) / (2.0 * dot_xkv_wv);
        } else {
          double denom = R * R - norm_xk_minus_v_sq;
          NumericVector u(p);
          for (int d = 0; d < p; ++d) u[d] = R * w_minus_v[d] / denom;

          double u_dot_xkv = 0.0, u_dot_wv = 0.0;
          for (int d = 0; d < p; ++d) {
            u_dot_xkv += u[d] * xk_minus_v[d];
            u_dot_wv  += u[d] * w_minus_v[d];
          }

          double inner = u_dot_xkv * u_dot_xkv + R * u_dot_wv;
          if (inner < 0) continue;

          rad = std::sqrt(inner) - u_dot_xkv;
        }

        if (!std::isfinite(rad) || rad <= 0.0) continue;

        // Reflected point: v_reflected = 2*x_k - v
        NumericVector v_reflected(p);
        for (int d = 0; d < p; ++d) {
          v_reflected[d] = 2.0 * x_k[d] - v[d];
        }

        NumericVector xk_minus_vr(p), w_minus_vr(p);
        double norm_xk_minus_vr_sq = 0.0, dot_xkvr_wvr = 0.0;

        for (int d = 0; d < p; ++d) {
          xk_minus_vr[d] = x_k[d] - v_reflected[d];
          w_minus_vr[d] = w[d] - v_reflected[d];
          norm_xk_minus_vr_sq += xk_minus_vr[d] * xk_minus_vr[d];
          dot_xkvr_wvr += xk_minus_vr[d] * w_minus_vr[d];
        }

        double rad_reflected = -1.0;
        if (std::abs(R * R - norm_xk_minus_vr_sq) < 1e-12) {
          if (std::abs(dot_xkvr_wvr) < 1e-12) continue;
          double norm_wvr_sq = 0.0;
          for (int d = 0; d < p; ++d) {
            norm_wvr_sq += w_minus_vr[d] * w_minus_vr[d];
          }
          rad_reflected = (R * norm_wvr_sq) / (2.0 * dot_xkvr_wvr);
        } else {
          double denom2 = R * R - norm_xk_minus_vr_sq;
          NumericVector u2(p);
          for (int d = 0; d < p; ++d) u2[d] = R * w_minus_vr[d] / denom2;

          double u_dot_xkvr = 0.0, u_dot_wvr = 0.0;
          for (int d = 0; d < p; ++d) {
            u_dot_xkvr += u2[d] * xk_minus_vr[d];
            u_dot_wvr  += u2[d] * w_minus_vr[d];
          }

          double inner2 = u_dot_xkvr * u_dot_xkvr + R * u_dot_wvr;
          if (inner2 < 0) continue;

          rad_reflected = std::sqrt(inner2) - u_dot_xkvr;
        }

        if (!std::isfinite(rad_reflected) || rad_reflected <= 0.0) continue;

        double term = std::log(rad / R) + std::log(rad_reflected / R);
        sum_terms += term;
        count++;
      }
    }

    if (count > 0) {
      d_hat[k] = -1.0 / (sum_terms / (count * 2.0));
    } else {
      d_hat[k] = NA_REAL;
    }
  }

  double sum_d = 0.0;
  int valid = 0;
  for (int i = 0; i < n; ++i) {
    if (R_finite(d_hat[i])) {
      sum_d += d_hat[i];
      valid++;
    }
  }

  return valid > 0 ? sum_d / valid : NA_REAL;
}