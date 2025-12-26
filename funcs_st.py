'''
Spatial-temporal wrapper for original scenarios.
Adds temporal and spatial dependencies to the original 5 scenarios.
'''
import numpy as np
from scipy.stats import t as t_dist, norm, cauchy, laplace
import sys
sys.path.append('/Users/bleachvex/Downloads/projects/quantile-regression/python')
from funcs import Scenario1, Scenario2, Scenario3, Scenario4, Scenario5

class SpatialTemporalWrapper:
    """Wraps original scenarios with spatial-temporal dependencies"""
    def __init__(self, base_scenario, n_time=50, temporal_corr=0.3, spatial_corr=0.5):
        self.base = base_scenario
        self.n_in = base_scenario.n_in
        self.n_time = n_time
        self.temporal_corr = temporal_corr  # AR(1) coefficient for errors
        self.spatial_corr = spatial_corr    # Spatial correlation strength
        
    def _add_spatial_correlation(self, X):
        """Add spatial correlation using distance-based kernel"""
        n = X.shape[0]
        # Compute pairwise distances
        dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        # Gaussian kernel
        K = np.exp(-dists**2 / (2 * self.spatial_corr**2))
        # Generate spatially correlated noise
        z = np.random.randn(n)
        L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
        return L @ z
    
    def sample(self, X):
        """Generate samples with temporal and spatial correlation"""
        n_total = X.shape[0]
        n_per_time = n_total // self.n_time
        
        # Generate base samples
        y_base = self.base.sample(X)
        
        # Add temporal correlation to errors
        errors = np.zeros(n_total)
        past_error = 0
        
        for t in range(self.n_time):
            start_idx = t * n_per_time
            end_idx = start_idx + n_per_time if t < self.n_time - 1 else n_total
            
            # Spatial correlation within time step
            X_t = X[start_idx:end_idx]
            spatial_noise = self._add_spatial_correlation(X_t)
            
            # Temporal correlation (AR(1))
            temporal_noise = self.temporal_corr * past_error + np.sqrt(1 - self.temporal_corr**2) * np.random.randn(len(X_t))
            
            # Combine
            errors[start_idx:end_idx] = 0.5 * spatial_noise + 0.5 * temporal_noise
            past_error = temporal_noise.mean()
        
        return y_base + errors
    
    def quantile(self, X, q):
        """Compute quantile (spatial-temporal effects are in the noise)"""
        return self.base.quantile(X, q)
    
    def noiseless(self, X):
        """Noiseless function"""
        return self.base.noiseless(X)

# Create spatial-temporal versions
class STScenario1(SpatialTemporalWrapper):
    def __init__(self):
        super().__init__(Scenario1(), n_time=50, temporal_corr=0.3, spatial_corr=0.5)
        self.label = 'ST Scenario 1'

class STScenario2(SpatialTemporalWrapper):
    def __init__(self):
        super().__init__(Scenario2(), n_time=50, temporal_corr=0.3, spatial_corr=0.5)
        self.label = 'ST Scenario 2'

class STScenario3(SpatialTemporalWrapper):
    def __init__(self):
        super().__init__(Scenario3(), n_time=50, temporal_corr=0.3, spatial_corr=0.5)
        self.label = 'ST Scenario 3'

class STScenario4(SpatialTemporalWrapper):
    def __init__(self):
        super().__init__(Scenario4(), n_time=50, temporal_corr=0.3, spatial_corr=0.5)
        self.label = 'ST Scenario 4'

class STScenario5(SpatialTemporalWrapper):
    def __init__(self):
        super().__init__(Scenario5(), n_time=50, temporal_corr=0.3, spatial_corr=0.5)
        self.label = 'ST Scenario 5'

# Add R-based scenarios
# NOTE: Commented out - using funcs_r2.py instead for full R implementation
# from funcs_r import RScenario6, RScenario7, RScenario8
# 
# class STScenario6(SpatialTemporalWrapper):
#     def __init__(self):
#         super().__init__(RScenario6(), n_time=50, temporal_corr=0.3, spatial_corr=0.5)
#         self.label = 'ST Scenario 6 (R1)'
# 
# class STScenario7(SpatialTemporalWrapper):
#     def __init__(self):
#         super().__init__(RScenario7(), n_time=50, temporal_corr=0.3, spatial_corr=0.5)
#         self.label = 'ST Scenario 7 (R2)'
# 
# class STScenario8(SpatialTemporalWrapper):
#     def __init__(self):
#         super().__init__(RScenario8(), n_time=50, temporal_corr=0.3, spatial_corr=0.5)
#         self.label = 'ST Scenario 8 (R3)'

# Import R scenarios from funcs_r2.py (full R implementation)
from qtsnn.funcs_r2_notrue import STScenario6, STScenario7, STScenario8

# Import univariate scenarios from funcs_r1.py
from funcs_r1 import STScenario9, STScenario10, STScenario11, STScenario12, STScenario13, STScenario14
