'''
Simplified spatial-temporal scenarios based on R code.
Using funcs.py style for consistency.
'''
import numpy as np
from scipy.stats import t as t_dist, norm, cauchy

class Benchmark:
    def __init__(self):
        pass

    def noiseless(self, X):
        raise NotImplementedError

    def quantile(self, X, q):
        raise NotImplementedError

    def sample(self, X):
        raise NotImplementedError

# R Scenario 1: Piecewise constant with spatial dependency
class RScenario1(Benchmark):
    def __init__(self):
        super().__init__()
        self.n_in = 2
        self.label = 'R Scenario 1'
    
    def noiseless(self, X):
        """Piecewise constant based on distance to centers"""
        center1 = np.array([0.25, 0.25])
        center2 = np.array([0.75, 0.75])
        dist1 = np.linalg.norm(X - center1, axis=1)
        dist2 = np.linalg.norm(X - center2, axis=1)
        return np.where(dist1 < dist2, 1.0, -1.0)
    
    def quantile(self, X, q):
        return self.noiseless(X) + norm.ppf(q, 0, 1)
    
    def sample(self, X):
        """Add spatial correlation via distance-based noise"""
        base = self.noiseless(X)
        
        # Spatial correlation: nearby points have correlated noise
        n = X.shape[0]
        dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        K = np.exp(-dists**2 / 0.5)  # Gaussian kernel
        
        # Cholesky decomposition for correlated noise
        try:
            L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
            spatial_noise = L @ np.random.randn(n)
        except:
            spatial_noise = np.random.randn(n)
        
        # Independent noise
        indep_noise = norm.rvs(0, 0.5, size=n)
        
        return base + 0.5 * spatial_noise + 0.5 * indep_noise

# R Scenario 2: Linear boundary with heavy-tailed noise
class RScenario2(Benchmark):
    def __init__(self):
        super().__init__()
        self.n_in = 2
        self.label = 'R Scenario 2'
    
    def noiseless(self, X):
        """Linear boundary: (5/4)*x1 + (3/4)*x2 > 1"""
        return ((5/4) * X[:, 0] + (3/4) * X[:, 1] > 1).astype(float)
    
    def quantile(self, X, q):
        return self.noiseless(X) + t_dist.ppf(q, 2)
    
    def sample(self, X):
        """Add spatial correlation with t-distributed noise"""
        base = self.noiseless(X)
        
        # Spatial correlation
        n = X.shape[0]
        dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        K = np.exp(-dists**2 / 0.5)
        
        try:
            L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
            spatial_noise = L @ np.random.randn(n)
        except:
            spatial_noise = np.random.randn(n)
        
        # Heavy-tailed noise
        heavy_noise = cauchy.rvs(0, 1, size=n)
        
        return base + 0.5 * spatial_noise + 0.5 * heavy_noise

# R Scenario 3: Piecewise constant with quantile-specific shift
class RScenario3(Benchmark):
    def __init__(self, tau=0.5):
        super().__init__()
        self.n_in = 2
        self.tau = tau
        self.label = 'R Scenario 3'
    
    def noiseless(self, X):
        """Same as RScenario1"""
        center1 = np.array([0.25, 0.25])
        center2 = np.array([0.75, 0.75])
        dist1 = np.linalg.norm(X - center1, axis=1)
        dist2 = np.linalg.norm(X - center2, axis=1)
        return np.where(dist1 < dist2, 1.0, -1.0)
    
    def quantile(self, X, q):
        """Quantile-specific: shift by tau"""
        return self.noiseless(X) + norm.ppf(self.tau, 0, 1) + norm.ppf(q, 0, 1)
    
    def sample(self, X):
        """Add spatial correlation with t(3) noise"""
        base = self.noiseless(X) + norm.ppf(self.tau, 0, 1)
        
        # Spatial correlation
        n = X.shape[0]
        dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        K = np.exp(-dists**2 / 0.5)
        
        try:
            L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
            spatial_noise = L @ np.random.randn(n)
        except:
            spatial_noise = np.random.randn(n)
        
        # t-distributed noise
        t_noise = t_dist.rvs(3, size=n)
        
        return base + 0.5 * spatial_noise + 0.5 * t_noise
