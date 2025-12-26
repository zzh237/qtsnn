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

# R Scenario 6: Piecewise constant with spatial dependency
class RScenario6(Benchmark):
    def __init__(self):
        super().__init__()
        self.n_in = 2
        self.label = 'R Scenario 6'
    
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
        """Add spatial correlation via distance-based noise (optimized)"""
        base = self.noiseless(X)
        n = X.shape[0]
        
        # For large n, use approximate spatial correlation
        if n > 1000:
            # Simple approach: add local correlation only
            spatial_noise = np.random.randn(n)
            # Smooth with moving average
            window = min(50, n // 20)
            spatial_noise = np.convolve(spatial_noise, np.ones(window)/window, mode='same')
        else:
            # Full Cholesky for small n
            dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
            K = np.exp(-dists**2 / 0.5)
            try:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
                spatial_noise = L @ np.random.randn(n)
            except:
                spatial_noise = np.random.randn(n)
        
        indep_noise = norm.rvs(0, 0.5, size=n)
        return base + 0.5 * spatial_noise + 0.5 * indep_noise

# R Scenario 7: Linear boundary with heavy-tailed noise
class RScenario7(Benchmark):
    def __init__(self):
        super().__init__()
        self.n_in = 2
        self.label = 'R Scenario 7'
    
    def noiseless(self, X):
        """Linear boundary: (5/4)*x1 + (3/4)*x2 > 1"""
        return ((5/4) * X[:, 0] + (3/4) * X[:, 1] > 1).astype(float)
    
    def quantile(self, X, q):
        return self.noiseless(X) + t_dist.ppf(q, 2)
    
    def sample(self, X):
        """Add spatial correlation with t-distributed noise (optimized)"""
        base = self.noiseless(X)
        n = X.shape[0]
        
        # Optimized spatial correlation
        if n > 1000:
            spatial_noise = np.random.randn(n)
            window = min(50, n // 20)
            spatial_noise = np.convolve(spatial_noise, np.ones(window)/window, mode='same')
        else:
            dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
            K = np.exp(-dists**2 / 0.5)
            try:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
                spatial_noise = L @ np.random.randn(n)
            except:
                spatial_noise = np.random.randn(n)
        
        heavy_noise = cauchy.rvs(0, 1, size=n)
        return base + 0.5 * spatial_noise + 0.5 * heavy_noise

# R Scenario 8: Piecewise constant with quantile-specific shift
class RScenario8(Benchmark):
    def __init__(self, tau=0.5):
        super().__init__()
        self.n_in = 2
        self.tau = tau
        self.label = 'R Scenario 8'
    
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
        """Add spatial correlation with t(3) noise (optimized)"""
        base = self.noiseless(X) + norm.ppf(self.tau, 0, 1)
        n = X.shape[0]
        
        # Optimized spatial correlation
        if n > 1000:
            spatial_noise = np.random.randn(n)
            window = min(50, n // 20)
            spatial_noise = np.convolve(spatial_noise, np.ones(window)/window, mode='same')
        else:
            dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
            K = np.exp(-dists**2 / 0.5)
            try:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
                spatial_noise = L @ np.random.randn(n)
            except:
                spatial_noise = np.random.randn(n)
        
        t_noise = t_dist.rvs(3, size=n)
        return base + 0.5 * spatial_noise + 0.5 * t_noise
