'''
Univariate spatial-temporal data generation scenarios.
Converted from R code in QTS-main/Univariate/simulate_d1.R
'''
import numpy as np
from scipy.stats import t as t_dist, norm, cauchy
from tqdm import tqdm

class UnivariateSTScenario:
    def __init__(self, m_mult=1, tau=0.5):
        self.m_mult = m_mult
        self.tau = tau
        self.n_in = 1
        self.d = 1
        
    def _generate_m(self, n=None):
        self.n = n
        m = np.zeros(n, dtype=int)
        m[:n//4] = 16 * self.m_mult
        m[n//4:n//2] = 24 * self.m_mult
        m[n//2:3*n//4] = 20 * self.m_mult
        m[3*n//4:] = 10 * self.m_mult
        return m
    
    def _h_function(self, x, t):
        return (1/np.sqrt(2)) * np.pi * np.sin(t * x)

class STScenario10(UnivariateSTScenario):
    def __init__(self, m_mult=1, tau=0.5):
        super().__init__(m_mult, tau)
        self.label = 'ST Scenario 10 (Doppler)'
    
    def noiseless(self, X):
        return 2 * np.sin(2*np.pi / (X + 0.1)**0.5) * X**0.25
    
    def quantile(self, X, q):
        n = X.shape[0]
        m = self._generate_m(n)
        M = m.max()
        
        for i in range(1, n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = where.sum()
                X[i, :n_keep] = X[i-1, :m[i-1]][where]
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        past_b = np.zeros(25)
        past_e = np.zeros(M)
        
        for i in tqdm(range(n), desc='Quantile computation', leave=False):
            X_i = X[i, :m[i], 0]
            beta_i = self.noiseless(X_i)
            f_bar = beta_i.mean()
            beta_i = beta_i - f_bar
            
            b = norm.ppf(q, 0, 1, size=25)
            delta = np.zeros(m[i])
            for j in range(m[i]):
                h_vals = self._h_function(X_i[j], t)
                delta[j] = np.sum(0.5 * past_b * h_vals) + np.sum((1/t) * b * h_vals)
            
            epsilon = 0.3 * past_e + norm.ppf(q, 0, 1)
            y_i = beta_i + delta + epsilon[:m[i]]
            
            past_b = 0.5 * past_b + (1/t) * b
            past_e = epsilon
            
            X_list.append(X_i.reshape(-1, 1))
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        return X_full, y_full
    
    def sample(self, X):
        n = X.shape[0]
        m = self._generate_m(n)
        M = m.max()
        
        for i in range(1, n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = where.sum()
                X[i, :n_keep] = X[i-1, :m[i-1]][where]
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        past_b = np.zeros(25)
        past_e = np.zeros(M)
        
        for i in tqdm(range(n), desc='Sample generation', leave=False):
            X_i = X[i, :m[i], 0]
            beta_i = self.noiseless(X_i)
            f_bar = beta_i.mean()
            beta_i = beta_i - f_bar
            
            b = np.random.normal(0, 1, 25)
            delta = np.zeros(m[i])
            for j in range(m[i]):
                h_vals = self._h_function(X_i[j], t)
                delta[j] = np.sum(0.5 * past_b * h_vals) + np.sum((1/t) * b * h_vals)
            
            epsilon = 0.3 * past_e + np.random.normal(0, 1, M)
            y_i = beta_i + delta + epsilon[:m[i]]
            
            past_b = 0.5 * past_b + (1/t) * b
            past_e = epsilon
            
            X_list.append(X_i.reshape(-1, 1))
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        return X_full, y_full

class STScenario11(UnivariateSTScenario):
    def __init__(self, m_mult=1, tau=0.5):
        super().__init__(m_mult, tau)
        self.label = 'ST Scenario 11 (Doppler, t-dist)'
    
    def noiseless(self, X):
        return 2 * np.sin(2*np.pi / (X + 0.1)**0.5) * X**0.25
    
    def quantile(self, X, q):
        n = X.shape[0]
        m = self._generate_m(n)
        M = m.max()
        
        for i in range(1, n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = where.sum()
                X[i, :n_keep] = X[i-1, :m[i-1]][where]
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        past_b = np.zeros(25)
        past_e = np.zeros(M)
        
        for i in tqdm(range(n), desc='Quantile computation', leave=False):
            X_i = X[i, :m[i], 0]
            beta_i = self.noiseless(X_i)
            f_bar = beta_i.mean()
            beta_i = beta_i - f_bar
            
            b = t_dist.ppf(q, 2, size=25)
            delta = np.zeros(m[i])
            for j in range(m[i]):
                h_vals = self._h_function(X_i[j], t)
                delta[j] = np.sum(0.5 * past_b * h_vals) + np.sum((1/t) * b * h_vals)
            
            epsilon = 0.3 * past_e + t_dist.ppf(q, 3)
            y_i = beta_i + delta + epsilon[:m[i]]
            
            past_b = 0.5 * past_b + (1/t) * b
            past_e = epsilon
            
            X_list.append(X_i.reshape(-1, 1))
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        return X_full, y_full
    
    def sample(self, X):
        n = X.shape[0]
        m = self._generate_m(n)
        M = m.max()
        
        for i in range(1, n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = where.sum()
                X[i, :n_keep] = X[i-1, :m[i-1]][where]
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        past_b = np.zeros(25)
        past_e = np.zeros(M)
        
        for i in tqdm(range(n), desc='Sample generation', leave=False):
            X_i = X[i, :m[i], 0]
            beta_i = self.noiseless(X_i)
            f_bar = beta_i.mean()
            beta_i = beta_i - f_bar
            
            b = t_dist.rvs(2, size=25)
            delta = np.zeros(m[i])
            for j in range(m[i]):
                h_vals = self._h_function(X_i[j], t)
                delta[j] = np.sum(0.5 * past_b * h_vals) + np.sum((1/t) * b * h_vals)
            
            epsilon = 0.3 * past_e + t_dist.rvs(3, size=M)
            y_i = beta_i + delta + epsilon[:m[i]]
            
            past_b = 0.5 * past_b + (1/t) * b
            past_e = epsilon
            
            X_list.append(X_i.reshape(-1, 1))
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        return X_full, y_full

class STScenario12(UnivariateSTScenario):
    def __init__(self, m_mult=1, tau=0.5):
        super().__init__(m_mult, tau)
        self.label = 'ST Scenario 12 (Piecewise Linear)'
    
    def noiseless(self, X):
        return np.where(X <= 0.4, 2.5 * X,
               np.where(X <= 0.6, 45 * X - 17,
               np.where(X <= 0.8, -40 * X + 34, 30 * X - 22)))
    
    def quantile(self, X, q):
        n = X.shape[0]
        m = self._generate_m(n)
        M = m.max()
        
        for i in range(1, n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = where.sum()
                X[i, :n_keep] = X[i-1, :m[i-1]][where]
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        past_b = np.zeros(25)
        past_e = np.zeros(M)
        
        for i in tqdm(range(n), desc='Quantile computation', leave=False):
            X_i = X[i, :m[i], 0]
            beta_i = self.noiseless(X_i)
            f_bar = beta_i.mean()
            beta_i = beta_i - f_bar
            
            b = t_dist.ppf(q, 3, size=25)
            delta = np.zeros(m[i])
            for j in range(m[i]):
                h_vals = self._h_function(X_i[j], t)
                delta[j] = np.sum(0.5 * past_b * h_vals) + np.sum((1/t) * b * h_vals)
            
            epsilon = 0.3 * past_e + t_dist.ppf(q, 2)
            y_i = beta_i + delta + epsilon[:m[i]]
            
            past_b = 0.5 * past_b + (1/t) * b
            past_e = epsilon
            
            X_list.append(X_i.reshape(-1, 1))
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        return X_full, y_full
    
    def sample(self, X):
        n = X.shape[0]
        m = self._generate_m(n)
        M = m.max()
        
        for i in range(1, n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = where.sum()
                X[i, :n_keep] = X[i-1, :m[i-1]][where]
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        past_b = np.zeros(25)
        past_e = np.zeros(M)
        
        for i in tqdm(range(n), desc='Sample generation', leave=False):
            X_i = X[i, :m[i], 0]
            beta_i = self.noiseless(X_i)
            f_bar = beta_i.mean()
            beta_i = beta_i - f_bar
            
            b = t_dist.rvs(3, size=25)
            delta = np.zeros(m[i])
            for j in range(m[i]):
                h_vals = self._h_function(X_i[j], t)
                delta[j] = np.sum(0.5 * past_b * h_vals) + np.sum((1/t) * b * h_vals)
            
            epsilon = 0.3 * past_e + t_dist.rvs(2, size=M)
            y_i = beta_i + delta + epsilon[:m[i]]
            
            past_b = 0.5 * past_b + (1/t) * b
            past_e = epsilon
            
            X_list.append(X_i.reshape(-1, 1))
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        return X_full, y_full

class STScenario13(UnivariateSTScenario):
    def __init__(self, m_mult=1, tau=0.5):
        super().__init__(m_mult, tau)
        self.label = 'ST Scenario 13 (Heterogeneous Sine)'
    
    def noiseless(self, X):
        beta = 1.5 * np.sin(4 * np.pi * X)
        beta[X > 0.5] += np.sin(16 * np.pi * X[X > 0.5])
        return beta
    
    def quantile(self, X, q):
        n = X.shape[0]
        m = self._generate_m(n)
        M = m.max()
        
        for i in range(1, n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = where.sum()
                X[i, :n_keep] = X[i-1, :m[i-1]][where]
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        past_b = np.zeros(25)
        past_e = np.zeros(M)
        
        for i in tqdm(range(n), desc='Quantile computation', leave=False):
            X_i = X[i, :m[i], 0]
            beta_i = self.noiseless(X_i)
            f_bar = beta_i.mean()
            beta_i = beta_i - f_bar
            
            b = t_dist.ppf(q, 3, size=25)
            delta = np.zeros(m[i])
            for j in range(m[i]):
                h_vals = self._h_function(X_i[j], t)
                delta[j] = np.sum(0.5 * past_b * h_vals) + np.sum((1/t) * b * h_vals)
            
            epsilon = 0.1 * past_e + cauchy.ppf(q, 0, 1)
            y_i = beta_i + delta + epsilon[:m[i]]
            
            past_b = 0.5 * past_b + (1/t) * b
            past_e = epsilon
            
            X_list.append(X_i.reshape(-1, 1))
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        return X_full, y_full
    
    def sample(self, X):
        n = X.shape[0]
        m = self._generate_m(n)
        M = m.max()
        
        for i in range(1, n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = where.sum()
                X[i, :n_keep] = X[i-1, :m[i-1]][where]
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        past_b = np.zeros(25)
        past_e = np.zeros(M)
        
        for i in tqdm(range(n), desc='Sample generation', leave=False):
            X_i = X[i, :m[i], 0]
            beta_i = self.noiseless(X_i)
            f_bar = beta_i.mean()
            beta_i = beta_i - f_bar
            
            b = t_dist.rvs(3, size=25)
            delta = np.zeros(m[i])
            for j in range(m[i]):
                h_vals = self._h_function(X_i[j], t)
                delta[j] = np.sum(0.5 * past_b * h_vals) + np.sum((1/t) * b * h_vals)
            
            epsilon = 0.1 * past_e + cauchy.rvs(0, 1, M)
            y_i = beta_i + delta + epsilon[:m[i]]
            
            past_b = 0.5 * past_b + (1/t) * b
            past_e = epsilon
            
            X_list.append(X_i.reshape(-1, 1))
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        return X_full, y_full

class STScenario14(UnivariateSTScenario):
    def __init__(self, m_mult=1, tau=0.5):
        super().__init__(m_mult, tau)
        self.label = 'ST Scenario 14 (Lagrange Polynomial)'
    
    def noiseless(self, X):
        return -2125/6 * X**4 + 2050/3 * X**3 - 2465/6 * X**2 + 248/3 * X
    
    def quantile(self, X, q):
        n = X.shape[0]
        m = self._generate_m(n)
        M = m.max()
        
        for i in range(1, n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = where.sum()
                X[i, :n_keep] = X[i-1, :m[i-1]][where]
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        past_b = np.zeros(25)
        past_e = np.zeros(M)
        
        for i in tqdm(range(n), desc='Quantile computation', leave=False):
            X_i = X[i, :m[i], 0]
            beta_i = self.noiseless(X_i)
            f_bar = beta_i.mean()
            beta_i = beta_i - f_bar
            
            b = t_dist.ppf(q, 3, size=25)
            delta = np.zeros(m[i])
            for j in range(m[i]):
                h_vals = self._h_function(X_i[j], t)
                delta[j] = np.sum(0.5 * past_b * h_vals) + np.sum((1/t) * b * h_vals)
            
            epsilon = 0.3 * past_e + norm.ppf(q, 0, 1)
            y_i = beta_i + delta + epsilon[:m[i]]
            
            past_b = 0.5 * past_b + (1/t) * b
            past_e = epsilon
            
            X_list.append(X_i.reshape(-1, 1))
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        return X_full, y_full
    
    def sample(self, X):
        n = X.shape[0]
        m = self._generate_m(n)
        M = m.max()
        
        for i in range(1, n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = where.sum()
                X[i, :n_keep] = X[i-1, :m[i-1]][where]
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        past_b = np.zeros(25)
        past_e = np.zeros(M)
        
        for i in tqdm(range(n), desc='Sample generation', leave=False):
            X_i = X[i, :m[i], 0]
            beta_i = self.noiseless(X_i)
            f_bar = beta_i.mean()
            beta_i = beta_i - f_bar
            
            b = t_dist.rvs(3, size=25)
            delta = np.zeros(m[i])
            for j in range(m[i]):
                h_vals = self._h_function(X_i[j], t)
                delta[j] = np.sum(0.5 * past_b * h_vals) + np.sum((1/t) * b * h_vals)
            
            epsilon = 0.3 * past_e + np.random.normal(0, 1, M)
            y_i = beta_i + delta + epsilon[:m[i]]
            
            past_b = 0.5 * past_b + (1/t) * b
            past_e = epsilon
            
            X_list.append(X_i.reshape(-1, 1))
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        return X_full, y_full

class STScenario15(UnivariateSTScenario):
    def __init__(self, m_mult=1, tau=0.5):
        super().__init__(m_mult, tau)
        self.label = 'ST Scenario 15 (Piecewise Constant)'
    
    def noiseless(self, X):
        breakpoints = np.linspace(0, 1, 7)**2
        interval = np.minimum(np.searchsorted(breakpoints, X, side='right') - 1, len(breakpoints) - 2)
        return ((interval % 2) == 1).astype(float) * 2 - 1
    
    def quantile(self, X, q):
        n = X.shape[0]
        m = self._generate_m(n)
        M = m.max()
        
        for i in range(1, n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = where.sum()
                X[i, :n_keep] = X[i-1, :m[i-1]][where]
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        past_b = np.zeros(25)
        past_e = np.zeros(M)
        
        for i in tqdm(range(n), desc='Quantile computation', leave=False):
            X_i = X[i, :m[i], 0]
            beta_i = self.noiseless(X_i)
            f_bar = beta_i.mean()
            beta_i = beta_i - f_bar
            
            b = norm.ppf(q, 0, 3, size=25)
            delta = np.zeros(m[i])
            for j in range(m[i]):
                h_vals = self._h_function(X_i[j], t)
                delta[j] = np.sum(0.5 * past_b * h_vals) + np.sum((1/t) * b * h_vals)
            
            epsilon = 0.3 * past_e + t_dist.ppf(q, 2)
            y_i = beta_i + delta + epsilon[:m[i]]
            
            past_b = 0.5 * past_b + (1/t) * b
            past_e = epsilon
            
            X_list.append(X_i.reshape(-1, 1))
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        return X_full, y_full
    
    def sample(self, X):
        n = X.shape[0]
        m = self._generate_m(n)
        M = m.max()
        
        for i in range(1, n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = where.sum()
                X[i, :n_keep] = X[i-1, :m[i-1]][where]
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        past_b = np.zeros(25)
        past_e = np.zeros(M)
        
        for i in tqdm(range(n), desc='Sample generation', leave=False):
            X_i = X[i, :m[i], 0]
            beta_i = self.noiseless(X_i)
            f_bar = beta_i.mean()
            beta_i = beta_i - f_bar
            
            b = np.random.normal(0, 3, 25)
            delta = np.zeros(m[i])
            for j in range(m[i]):
                h_vals = self._h_function(X_i[j], t)
                delta[j] = np.sum(0.5 * past_b * h_vals) + np.sum((1/t) * b * h_vals)
            
            epsilon = 0.3 * past_e + t_dist.rvs(2, size=M)
            y_i = beta_i + delta + epsilon[:m[i]]
            
            past_b = 0.5 * past_b + (1/t) * b
            past_e = epsilon
            
            X_list.append(X_i.reshape(-1, 1))
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        return X_full, y_full
