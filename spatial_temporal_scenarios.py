'''
Spatial-temporal data generation scenarios.
Converted from R code in QTS-main/Multivariate/scenarios.R
'''
import numpy as np
from scipy.stats import t as t_dist, norm, cauchy

class SpatialTemporalScenario:
    def __init__(self, n=200, m_mult=1, d=2, tau=0.5):
        self.n = n
        self.m_mult = m_mult
        self.d = d
        self.tau = tau
        self.n_in = d
        
    def _generate_m(self):
        m = np.zeros(self.n, dtype=int)
        m[:self.n//4] = 16 * self.m_mult
        m[self.n//4:self.n//2] = 24 * self.m_mult
        m[self.n//2:3*self.n//4] = 20 * self.m_mult
        m[3*self.n//4:] = 10 * self.m_mult
        return m
    
    def _h_function(self, x, t):
        products = np.array([(1/np.sqrt(2)) * np.pi * np.sin(t_val * x) for t_val in t])
        return np.prod(products, axis=1)

class STScenario1(SpatialTemporalScenario):
    def __init__(self, n=200, m_mult=1, d=2, tau=0.5):
        super().__init__(n, m_mult, d, tau)
        self.label = 'ST Scenario 1'
    
    def noiseless(self, X):
        center1 = np.ones(self.d) * 0.25
        center2 = np.ones(self.d) * 0.75
        dist1 = np.linalg.norm(X - center1, axis=1)
        dist2 = np.linalg.norm(X - center2, axis=1)
        return np.where(dist1 < dist2, 1, -1)
    
    def quantile(self, X, q):
        return self.noiseless(X) + norm.ppf(q, 0, 1)
    
    def sample(self, X):
        m = self._generate_m()
        M = m.max()
        base_x = np.random.uniform(0, 1, (self.n, M, self.d))
        
        for i in range(1, self.n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = where.sum()
                base_x[i, :n_keep] = base_x[i-1, :m[i-1]][where]
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        past_b = np.zeros(25)
        past_e = np.zeros(M)
        
        for i in range(self.n):
            X_i = base_x[i, :m[i]]
            beta_i = self.noiseless(X_i)
            
            b = np.random.normal(0, 1, 25)
            delta = np.zeros(m[i])
            for j in range(m[i]):
                h_vals = self._h_function(X_i[j], t)
                delta[j] = np.sum(0.5 * past_b * h_vals) + np.sum((1/t) * b * h_vals)
            
            epsilon = 0.3 * past_e + np.random.normal(0, 0.5, M)
            y_i = beta_i + delta + epsilon[:m[i]]
            
            past_b = 0.5 * past_b + (1/t) * b
            past_e = epsilon
            
            X_list.append(X_i)
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        
        if len(X) == len(X_full):
            return y_full
        
        indices = np.random.choice(len(X_full), len(X), replace=False)
        return y_full[indices]

class STScenario2(SpatialTemporalScenario):
    def __init__(self, n=200, m_mult=1, d=2, tau=0.5):
        super().__init__(n, m_mult, d, tau)
        self.label = 'ST Scenario 2'
    
    def noiseless(self, X):
        return ((5/4) * X[:, 0] + (3/4) * X[:, 1] > 1).astype(float)
    
    def quantile(self, X, q):
        return self.noiseless(X) + t_dist.ppf(q, 2)
    
    def sample(self, X):
        m = self._generate_m()
        M = m.max()
        base_x = np.random.uniform(0, 1, (self.n, M, self.d))
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        past_b = np.zeros(25)
        past_e = np.zeros(M)
        
        for i in range(self.n):
            X_i = base_x[i, :m[i]]
            beta_i = self.noiseless(X_i)
            
            b = t_dist.rvs(2, size=25)
            delta = np.zeros(m[i])
            for j in range(m[i]):
                h_vals = self._h_function(X_i[j], t)
                delta[j] = np.sum(0.5 * past_b * h_vals) + np.sum((1/t) * b * h_vals)
            
            epsilon = 0.3 * past_e + cauchy.rvs(0, 1, M)
            y_i = beta_i + delta + epsilon[:m[i]]
            
            past_b = 0.5 * past_b + (1/t) * b
            past_e = epsilon
            
            X_list.append(X_i)
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        
        if len(X) == len(X_full):
            return y_full
        
        indices = np.random.choice(len(X_full), len(X), replace=False)
        return y_full[indices]

class STScenario3(SpatialTemporalScenario):
    def __init__(self, n=200, m_mult=1, d=2, tau=0.5):
        super().__init__(n, m_mult, d, tau)
        self.label = 'ST Scenario 3'
    
    def noiseless(self, X):
        center1 = np.ones(self.d) * 0.25
        center2 = np.ones(self.d) * 0.75
        dist1 = np.linalg.norm(X - center1, axis=1)
        dist2 = np.linalg.norm(X - center2, axis=1)
        return np.where(dist1 < dist2, 1, -1)
    
    def quantile(self, X, q):
        return self.noiseless(X) + norm.ppf(q, 0, 1)
    
    def sample(self, X):
        m = self._generate_m()
        M = m.max()
        base_x = np.random.uniform(0, 1, (self.n, M, self.d))
        
        for i in range(1, self.n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = where.sum()
                base_x[i, :n_keep] = base_x[i-1, :m[i-1]][where]
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        past_b = np.zeros(25)
        past_e = np.zeros(M)
        
        for i in range(self.n):
            X_i = base_x[i, :m[i]]
            beta_i = self.noiseless(X_i) + norm.ppf(self.tau, 0, 1)
            
            scale_delta = 1 if i == 0 else np.sqrt(0.5**2 + 1**2)
            scale_epsilon = 1 if i == 0 else np.sqrt(0.3**2 + 1**2)
            
            b = (0.5 * past_b + (1/t) * np.random.normal(0, 1, 25)) / scale_delta
            delta = np.zeros(m[i])
            for j in range(m[i]):
                h_vals = self._h_function(X_i[j], t)
                delta[j] = np.sum(b * h_vals)
            
            epsilon = (0.3 * past_e + t_dist.rvs(3, size=M)) / scale_epsilon
            y_i = beta_i + delta + epsilon[:m[i]]
            
            past_b = b
            past_e = epsilon
            
            X_list.append(X_i)
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        
        if len(X) == len(X_full):
            return y_full
        
        indices = np.random.choice(len(X_full), len(X), replace=False)
        return y_full[indices]
