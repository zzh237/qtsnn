'''
Spatial-temporal data generation scenarios.
Converted from R code in QTS-main/Multivariate/scenarios.R
'''
import numpy as np
from scipy.stats import t as t_dist, norm, cauchy
from tqdm import tqdm

class SpatialTemporalScenario:
    def __init__(self, m_mult=1, d=2, tau=0.5):
        self.m_mult = m_mult
        self.d = d
        self.tau = tau
        self.n_in = d
        
    def _generate_m(self, n=None):
        self.n = n
        m = np.zeros(n, dtype=int)
        m[:n//4] = 16 * self.m_mult
        m[n//4:n//2] = 24 * self.m_mult
        m[n//2:3*n//4] = 20 * self.m_mult
        m[3*n//4:] = 10 * self.m_mult
        return m
    
    def _h_function(self, x, t):
        products = np.array([(1/np.sqrt(2)) * np.pi * np.sin(t_val * x) for t_val in t])
        return np.prod(products, axis=1)
    
    def _flatten_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 3:
            n, M, d = X.shape
            if d != self.d:
                raise ValueError(f"X last dim must be {self.d}, got {d}.")
            m = self._generate_m(n)
            return np.vstack([X[i, :m[i], :] for i in range(n)])  # (N_flat, d)
        elif X.ndim == 2:
            if X.shape[1] != self.d:
                raise ValueError(f"X must have shape (N, {self.d}), got {X.shape}.")
            return X
        else:
            raise ValueError(f"X must be 2D or 3D, got {X.ndim}D.")

class STScenario6(SpatialTemporalScenario):
    def __init__(self, m_mult=1, d=2, tau=0.5):
        super().__init__(m_mult, d, tau)
        self.label = 'ST Scenario 6 (R1)'

         # caches created by sample()
        self._cache_X_full = None
        self._cache_shift = None   # deterministic shift term per flattened point
        self._cache_var = None     # conditional variance per flattened point (given past)

    
    def noiseless(self, X):
        center1 = np.ones(self.d) * 0.25
        center2 = np.ones(self.d) * 0.75
        dist1 = np.linalg.norm(X - center1, axis=1)
        dist2 = np.linalg.norm(X - center2, axis=1)
        return np.where(dist1 < dist2, 1, -1)
    
    
    def _h_matrix(self, X, t):
        X = np.asarray(X, dtype=float)
        t = np.asarray(t, dtype=float)
        sin_part = np.sin(X[:, None, :] * t[None, :, None])  # (N, T, d)
        prod_over_d = np.prod(sin_part, axis=2)              # (N, T)
        const = ((np.pi / np.sqrt(2.0)) ** X.shape[1])
        return const * prod_over_d    

    def quantile(self, X, q):
        """
        Oracle quantile if sample() was called and X matches cached X_full row order.
        Otherwise, fall back to a stationary/independence approximation (only depends on X).
        """
        if not (0.0 < q < 1.0):
            raise ValueError("q must be in (0, 1).")
        z = norm.ppf(q)

        X_flat = self._flatten_X(X)          # (N_flat, d)
        beta = self.noiseless(X_flat)        # (N_flat,)

        # --- oracle branch: use cached shift/var from the last sample() ---
        if (
            hasattr(self, "_cache_shift") and hasattr(self, "_cache_var") and hasattr(self, "_cache_nflat")
            and self._cache_shift is not None and self._cache_var is not None
            and X_flat.shape[0] == self._cache_nflat
        ):
            return beta + self._cache_shift + z * np.sqrt(self._cache_var)

        # --- fallback: steady-state approximation (no history available) ---
        t = np.arange(1, 26, dtype=float)
        H = self._h_matrix(X_flat, t)

        eps_var_ss = (0.5 ** 2) / (1.0 - 0.3 ** 2)
        delta_var_ss = (4.0 / 3.0) * np.sum((H / t[None, :]) ** 2, axis=1)

        return beta + z * np.sqrt(delta_var_ss + eps_var_ss)

    def sample(self, X):
        """
        X expected shape: (n, M, d) where M >= max(m)
        """
        n = X.shape[0]
        m = self._generate_m(n)
        M = m.max()

        # keep mechanism (in-place modify X!)
        for i in range(1, n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = where.sum()
                X[i, :n_keep, :] = X[i-1, :m[i-1], :][where]

        X_list, y_list = [], []
        shift_list, var_list = [], []   # <-- NEW

        t = np.arange(1, 26, dtype=float)  # (25,)
        past_b = np.zeros(25)
        past_e = np.zeros(M)

        for i in tqdm(range(n), desc='Sample generation', leave=False):
            X_i = X[i, :m[i], :]              # (m_i, d)
            beta_i = self.noiseless(X_i)      # (m_i,)

            # ---- NEW: oracle shift & var given current history ----
            H = self._h_matrix(X_i, t)  # (m_i, 25)

            shift_i = H @ (0.5 * past_b) + 0.3 * past_e[:m[i]]           # (m_i,)
            var_i = np.sum((H / t[None, :]) ** 2, axis=1) + (0.5 ** 2)   # (m_i,)

            # ---- your original randomness ----
            b = np.random.normal(0, 1, 25)

            # random part from b: sum (1/t)*b*h
            delta_rand = H @ (b / t)  # (m_i,)

            epsilon = 0.3 * past_e + np.random.normal(0, 0.5, M)
            # y = beta + (0.5*past_b*h + (b/t)*h) + epsilon
            y_i = beta_i + shift_i + delta_rand + (epsilon[:m[i]] - 0.3 * past_e[:m[i]])

            # update states
            past_b = 0.5 * past_b + (1/t) * b
            past_e = epsilon

            X_list.append(X_i)
            y_list.append(y_i)
            shift_list.append(shift_i)   # <-- NEW
            var_list.append(var_i)       # <-- NEW

        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)

        # ---- NEW: cache aligned with X_full row order ----
        self._cache_shift = np.concatenate(shift_list)
        self._cache_var = np.concatenate(var_list)
        self._cache_nflat = X_full.shape[0]

        print(f"X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")
        return X_full, y_full
        # if len(X) <= len(X_full):
        #     indices = np.random.choice(len(X_full), len(X), replace=False)
        #     return y_full[indices]
        # else:
        #     indices = np.random.choice(len(X_full), len(X), replace=True)
        #     return y_full[indices]

class STScenario7(SpatialTemporalScenario):
    def __init__(self, m_mult=1, d=2, tau=0.5):
        super().__init__(m_mult, d, tau)
        self.label = 'ST Scenario 7 (R2)'
    
    def noiseless(self, X):
        return ((5/4) * X[:, 0] + (3/4) * X[:, 1] > 1).astype(float)
    
    def quantile(self, X, q):
        n = X.shape[0]
        m = self._generate_m(n)
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        for i in tqdm(range(n), desc='Quantile computation', leave=False):
            X_i = X[i, :m[i]]
            beta_i = self.noiseless(X_i)

            b = t_dist.ppf(q, 2)
            delta = np.zeros(m[i])
            for j in range(m[i]):
                h_vals = self._h_function(X_i[j], t)
                delta[j] = np.sum((1/t) * b * h_vals)

            # y_i = beta_i + cauchy.ppf(q, 0, 1) * np.ones(m[i])
            y_i = beta_i + delta + cauchy.ppf(q, 0, 1) * np.ones(m[i])

            X_list.append(X_i)
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        print(f"X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")
        
        return y_full
        # if len(X) <= len(X_full):
        #     indices = np.random.choice(len(X_full), len(X), replace=False)
        #     return y_full[indices]
        # else:
        #     indices = np.random.choice(len(X_full), len(X), replace=True)
        #     return y_full[indices]
    
    def sample(self, X):
        n = X.shape[0]
        m = self._generate_m(n)
        M = m.max()
        
        X_list, y_list = [], []
        t = np.arange(1, 26)
        past_b = np.zeros(25)
        past_e = np.zeros(M)
        
        for i in tqdm(range(n), desc='Sample generation', leave=False):
            X_i = X[i, :m[i]]
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
        print(f"X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")
        return X_full, y_full
        # if len(X) <= len(X_full):
        #     indices = np.random.choice(len(X_full), len(X), replace=False)
        #     return y_full[indices]
        # else:
        #     indices = np.random.choice(len(X_full), len(X), replace=True)
        #     return y_full[indices]

class STScenario8(SpatialTemporalScenario):
    def __init__(self, m_mult=1, d=2, tau=0.5):
        super().__init__(m_mult, d, tau)
        self.label = 'ST Scenario 8 (R3)'
    
    def noiseless(self, X):
        center1 = np.ones(self.d) * 0.25
        center2 = np.ones(self.d) * 0.75
        dist1 = np.linalg.norm(X - center1, axis=1)
        dist2 = np.linalg.norm(X - center2, axis=1)
        return np.where(dist1 < dist2, 1, -1)
    
    def quantile(self, X, q):
        n = X.shape[0]
        m = self._generate_m(n)
        
        for i in range(1, n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = where.sum()
                X[i, :n_keep] = X[i-1, :m[i-1]][where]
        
        X_list, y_list = [], []
        
        for i in tqdm(range(n), desc='Quantile computation', leave=False):
            X_i = X[i, :m[i]]
            beta_i = self.noiseless(X_i) + norm.ppf(self.tau, 0, 1)
            y_i = beta_i + t_dist.ppf(q, 3) * np.ones(m[i])
            
            X_list.append(X_i)
            y_list.append(y_i)
        
        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)
        print(f"X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")
        
        return y_full
        # if len(X) <= len(X_full):
        #     indices = np.random.choice(len(X_full), len(X), replace=False)
        #     return y_full[indices]
        # else:
        #     indices = np.random.choice(len(X_full), len(X), replace=True)
        #     return y_full[indices]
    
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
            X_i = X[i, :m[i]]
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
        print(f"X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")
        return X_full, y_full
        # if len(X) <= len(X_full):
        #     indices = np.random.choice(len(X_full), len(X), replace=False)
        #     return y_full[indices]
        # else:
        #     indices = np.random.choice(len(X_full), len(X), replace=True)
        #     return y_full[indices]
