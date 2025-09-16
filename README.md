# manifold-dimension-estimation

This repository contains the Python code used in the following research articles:

- **Manifold Dimension Estimation: An Empirical Study**
- **Beyond PCA: Two Curvature-Aware Manifold Dimension Estimators through Quadratic Embedding**

The code implements various algorithms and tools for estimating the intrinsic dimension of data lying on a manifold, including curvature-aware methods.

## 🗂️ File Descriptions

| File(s) | Description |
|---------|-------------|
| `mymodules.py` | Implementation of manifolds and estimators. |
| `run_nbh_new.py` | Benchmark Dimension Estimates versus Neighborhood Sizes. |
| `run_n_new.py` | Benchmark Dimension Estimates versus Sample Sizes. |
| `run_p_new.py` | Benchmark Dimension Estimates versus Ambient Dimensions. |
| `run_noise_new.py` | Benchmark Dimension Estimates versus Gaussian Noise. |
| - `run_curvarture_01_new.py`<br>- `run_curvarture_02_new.py`<br>- `run_curvarture_03_new.py` | Benchmark Dimension Estimates versus Different Aspects of Curvature. |
| `run_nc_new.py` | Demonstrate Noise-Curvature Dilemma Effect. |
| - `run_new_uniform_500.py`<br>- `run_new_uniform_2000.py`<br>- `run_new_nonuniform_500.py`<br>- `run_new_nonuniform_2000.py`<br>- `run_new_noise_500.py`<br>- `run_new_noise_2000.py` | Compare Dimension Estimates across Simulated Datasets. |
| - `run_isomap.py`<br>- `run_mnist.py`<br>- `run_isolet.py` | Compare Dimension Estimates across Real-world Datasets. |
| - `tle.cpp`<br>- `estimators.R`<br>- `DanCo.R`<br>- `faces.R`<br>- `digits.R`<br>- `isolet.R` | R implementation of the estimators, along with testing on real-world datasets. |

## 📚 Related Publications

1. **Manifold Dimension Estimation: An Empirical Study**  
   *Bi, Zelong and Pierre Lafaye de Micheaux*  
   [Link or DOI]

2. **Beyond PCA: Two Curvature-Aware Manifold Dimension Estimators through Quadratic Embedding**  
   *(Author list)*  
   [Link or DOI]

## 📬 Contact

For questions or further information, please contact:  
**Bi, Zelong** – zelong.bi@unsw.edu.au
