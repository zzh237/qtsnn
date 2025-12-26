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
        
    def _h_matrix(self, X, t):
        """
        Vectorized h(x,t):
        X: (N,d), t: (25,)
        return H: (N,25), H[n,k] = prod_{r=1..d} ((pi/sqrt(2))*sin(t_k * x_r))
        """
        X = np.asarray(X, dtype=float)
        t = np.asarray(t, dtype=float)  # (25,)

        # (N,25,d)
        sin_part = np.sin(X[:, None, :] * t[None, :, None])
        prod_over_d = np.prod(sin_part, axis=2)  # (N,25)
        const = ((np.pi / np.sqrt(2.0)) ** X.shape[1])
        return const * prod_over_d

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
         # caches created by sample()
        self._cache_X_full = None
        self._cache_shift = None   # deterministic shift term per flattened point
        self._cache_var = None     # conditional variance per flattened point (given past)

        # MC settings (you can tune)
        self.mc_R = 1024              # number of MC samples
        self.mc_chunk = 2000          # chunk size over points
        self._mc_B = None             # (R,25) cached b draws
        self._mc_U = None             # (R,) cached cauchy draws

    def noiseless(self, X):
        return ((5/4) * X[:, 0] + (3/4) * X[:, 1] > 1).astype(float)
    
    def quantile(self, X, q):
        """
        Oracle quantile after sample():
          Q_q(Y | X, history) = beta(X) + shift + Q_q(W b + U)
        where b~t2 iid, U~Cauchy(0,1).
        If cache not available, falls back to Monte Carlo with shift=0 and uses W computed from X.
        """
        if not (0.0 < q < 1.0):
            raise ValueError("q must be in (0,1).")

        X_flat = self._flatten_X(X)
        beta = self.noiseless(X_flat).astype(np.float32)

        # use cached oracle if available and lengths match
        if (self._cache_W is not None and self._cache_shift is not None and self._cache_nflat is not None
            and X_flat.shape[0] == self._cache_nflat):
            W = self._cache_W
            shift = self._cache_shift
            B = self._mc_B
            U = self._mc_U
        else:
            # fallback: no history => shift=0, still MC the random part
            t = np.arange(1, 26, dtype=float)
            H = self._h_matrix(X_flat, t)
            W = (H / t[None, :]).astype(np.float32)
            shift = np.zeros(X_flat.shape[0], dtype=np.float32)
            if self._mc_B is None or self._mc_U is None:
                B = t_dist.rvs(df=5, size=(self.mc_R, 25)).astype(np.float32)
                U = t_dist.rvs(df=5, size=self.mc_R).astype(np.float32)
            else:
                B = self._mc_B
                U = self._mc_U

        # MC quantile by chunks to control memory
        N = W.shape[0]
        out = np.empty(N, dtype=np.float32)

        for s in range(0, N, self.mc_chunk):
            e = min(N, s + self.mc_chunk)
            Wc = W[s:e, :]                       # (C,25)
            # samples: (R,C) = (R,25) @ (25,C) + (R,1)
            samples = B @ Wc.T                   # (R,C)
            samples = samples + U[:, None]       # add Cauchy innovation
            out[s:e] = np.quantile(samples, q, axis=0).astype(np.float32)

        return (beta + shift + out).astype(float)
    
    def sample(self, X):
        """
        X expected in your benchmark as: (n, M, d)
        Return: X_full (N_flat,d), y_full (N_flat,)
        Also caches shift and W aligned with X_full rows for oracle quantile.
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        m = self._generate_m(n)
        M = m.max()

        X_list, y_list = [], []
        shift_list, W_list = [], []

        t = np.arange(1, 26, dtype=float)  # (25,)
        past_b = np.zeros(25)
        past_e = np.zeros(M)

        for i in tqdm(range(n), desc='Sample generation', leave=False):
            X_i = X[i, :m[i], :]                 # <-- IMPORTANT FIX
            beta_i = self.noiseless(X_i)

            H = self._h_matrix(X_i, t)           # (m_i,25)
            W = H / t[None, :]                   # (m_i,25)

            # deterministic shift given history
            shift_i = H @ (0.5 * past_b) + 0.3 * past_e[:m[i]]   # (m_i,)

            # randomness
            b = t_dist.rvs(df=5, size=25)        # (25,)
            delta_rand = W @ b                   # (m_i,)  since W = H/t

            epsilon = 0.3 * past_e + t_dist.rvs(df=5, size=M)
            u = epsilon[:m[i]] - 0.3 * past_e[:m[i]]             # innovation part, (m_i,)

            y_i = beta_i + shift_i + delta_rand + u

            # update states
            past_b = 0.5 * past_b + (1.0 / t) * b
            past_e = epsilon

            X_list.append(X_i)
            y_list.append(y_i)
            shift_list.append(shift_i)
            W_list.append(W.astype(np.float32))  # save memory

        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)

        # cache for oracle quantile (aligned with X_full rows)
        self._cache_shift = np.concatenate(shift_list).astype(np.float32)
        self._cache_W = np.vstack(W_list)        # (N_flat,25)
        self._cache_nflat = X_full.shape[0]

        # prepare MC draws once (reuse for different q calls)
        self._mc_B = t_dist.rvs(df=5, size=(self.mc_R, 25)).astype(np.float32)
        self._mc_U = t_dist.rvs(df=5, size=self.mc_R).astype(np.float32)

        print(f"X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")
        return X_full, y_full

class STScenario8(SpatialTemporalScenario):
    def __init__(self, m_mult=1, d=2, tau=0.5):
        super().__init__(m_mult, d, tau)
        self.label = 'ST Scenario 8 (R3)'
         # caches created by sample()
        self._cache_X_full = None
        self._cache_shift = None   # deterministic shift term per flattened point
        self._cache_var = None     # conditional variance per flattened point (given past)

        # (optional) MC draws cache for quantile() if you use MC there
        self.mc_R = 1024
        self._mc_Z = None
        self._mc_T = None

    def noiseless(self, X):
        center1 = np.ones(self.d) * 0.25
        center2 = np.ones(self.d) * 0.75
        dist1 = np.linalg.norm(X - center1, axis=1)
        dist2 = np.linalg.norm(X - center2, axis=1)
        return np.where(dist1 < dist2, 1, -1)
    
    def quantile(self, X, q):
        """
        Oracle quantile (MC) for Scenario 8.

        Requires that sample() has been called right before so caches are aligned.
        If caches not available, fallback will ignore history shift (less accurate).
        """
        if not (0.0 < q < 1.0):
            raise ValueError("q must be in (0,1).")

        X_flat = self._flatten_X(X)  # parent 提供；把 (n,M,d) -> (N_flat,d)
        base = self.noiseless(X_flat) + norm.ppf(self.tau, loc=0, scale=1)

        # ---------- use oracle cache if available ----------
        use_cache = (
            hasattr(self, "_cache_nflat")
            and self._cache_nflat is not None
            and X_flat.shape[0] == self._cache_nflat
            and getattr(self, "_cache_shift", None) is not None
            and getattr(self, "_cache_sigma", None) is not None
            and getattr(self, "_cache_tscale", None) is not None
        )

        if use_cache:
            shift  = self._cache_shift.astype(np.float32)
            sigma  = self._cache_sigma.astype(np.float32)
            tscale = self._cache_tscale.astype(np.float32)
        else:
            # fallback: 没有 history，shift 只能当 0；sigma/tscale 用当前 i>0 的近似常数
            # （你 benchmark 是 sample 后立刻 quantile，基本不会走到这）
            t = np.arange(1, 26, dtype=float)
            H = self._h_matrix(X_flat, t)
            W = H / t[None, :]
            scale_delta = np.sqrt(0.5**2 + 1.0**2)
            scale_eps   = np.sqrt(0.3**2 + 1.0**2)
            shift  = np.zeros(X_flat.shape[0], dtype=np.float32)
            sigma  = (np.sqrt(np.sum(W**2, axis=1)) / scale_delta).astype(np.float32)
            tscale = np.full(X_flat.shape[0], 1.0/scale_eps, dtype=np.float32)

        # ---------- MC for quantile of Normal(0,sigma^2) + t3*tscale ----------
        R = getattr(self, "mc_R", 1024)
        chunk = getattr(self, "mc_chunk", 4000)

        # 复用 MC draws（每次 q 不同也能复用同一批样本）
        if getattr(self, "_mc_Z", None) is None or getattr(self, "_mc_T", None) is None or self._mc_Z.shape[0] != R:
            self._mc_Z = np.random.normal(0.0, 1.0, size=R).astype(np.float32)   # (R,)
            self._mc_T = t_dist.rvs(df=3, size=R).astype(np.float32)            # (R,)

        Z = self._mc_Z
        T = self._mc_T

        N = X_flat.shape[0]
        out = np.empty(N, dtype=np.float32)

        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            sig = sigma[s:e]     # (C,)
            sca = tscale[s:e]    # (C,)

            # samples: (R,C) = Z[:,None]*sig[None,:] + T[:,None]*sca[None,:]
            samples = Z[:, None] * sig[None, :] + T[:, None] * sca[None, :]
            out[s:e] = np.quantile(samples, q, axis=0).astype(np.float32)

        return (base + shift + out).astype(float)

    
    def sample(self, X):
        """
        X: expected shape (n, M, d), where M >= max(m)
        returns:
            X_full: (N_flat, d)
            y_full: (N_flat,)
        Also caches oracle terms aligned with X_full rows:
            self._cache_shift, self._cache_sigma, self._cache_tscale, self._cache_nflat
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 3 or X.shape[2] != self.d:
            raise ValueError(f"X must have shape (n, M, {self.d}), got {X.shape}.")

        n = X.shape[0]
        m = self._generate_m(n)
        M = m.max()

        # --- keep mechanism (in-place modify X) ---
        for i in range(1, n):
            where = np.random.uniform(0, 1, m[i-1]) < 0.1
            if where.any():
                n_keep = int(where.sum())
                # copy selected previous points into the front of current list
                X[i, :n_keep, :] = X[i-1, :m[i-1], :][where]

        X_list, y_list = [], []
        shift_list, sigma_list, tscale_list = [], [], []

        t = np.arange(1, 26, dtype=float)  # (25,)
        past_b = np.zeros(25, dtype=float)
        past_e = np.zeros(M, dtype=float)

        # constant shift for beta (tau-quantile of N(0,1))
        beta_shift = norm.ppf(self.tau, loc=0.0, scale=1.0)

        for i in tqdm(range(n), desc='Sample generation', leave=False):
            mi = int(m[i])
            X_i = X[i, :mi, :]  # (mi, d)  <-- IMPORTANT FIX

            # beta part
            beta_i = self.noiseless(X_i) + beta_shift  # (mi,)

            # scaling (your original)
            scale_delta   = 1.0 if i == 0 else np.sqrt(0.5**2 + 1.0**2)
            scale_epsilon = 1.0 if i == 0 else np.sqrt(0.3**2 + 1.0**2)

            # compute H and W
            H = self._h_matrix(X_i, t)         # (mi, 25)
            W = H / t[None, :]                  # (mi, 25)

            # ---- oracle cache pieces (given current history) ----
            # delta deterministic part from past_b: H @ (0.5*past_b) / scale_delta
            # epsilon deterministic part from past_e: (0.3*past_e_j) / scale_epsilon
            shift_i = (H @ (0.5 * past_b)) / scale_delta + (0.3 * past_e[:mi]) / scale_epsilon  # (mi,)

            # delta random part is Normal with std = ||W|| / scale_delta
            sigma_i = (np.sqrt(np.sum(W**2, axis=1)) / scale_delta)  # (mi,)

            # epsilon innovation is t3 scaled by 1/scale_epsilon
            tscale_i = np.full(mi, 1.0 / scale_epsilon, dtype=float)  # (mi,)

            # ---- generate randomness exactly as your code ----
            # b = (0.5*past_b + (1/t)*N(0,1)) / scale_delta
            innov_b = (1.0 / t) * np.random.normal(0.0, 1.0, 25)      # (25,)
            b = (0.5 * past_b + innov_b) / scale_delta                 # (25,)

            # delta = sum_k b_k * h_k(x) = H @ b
            delta = H @ b                                               # (mi,)

            # epsilon = (0.3*past_e + t3) / scale_epsilon
            epsilon = (0.3 * past_e + t_dist.rvs(df=3, size=M)) / scale_epsilon  # (M,)

            y_i = beta_i + delta + epsilon[:mi]                         # (mi,)

            # update states (your original)
            past_b = b
            past_e = epsilon

            # collect
            X_list.append(X_i)
            y_list.append(y_i)
            shift_list.append(shift_i)
            sigma_list.append(sigma_i)
            tscale_list.append(tscale_i)

        X_full = np.vstack(X_list)
        y_full = np.concatenate(y_list)

        # cache aligned with X_full rows
        self._cache_shift = np.concatenate(shift_list).astype(np.float32)
        self._cache_sigma = np.concatenate(sigma_list).astype(np.float32)
        self._cache_tscale = np.concatenate(tscale_list).astype(np.float32)
        self._cache_nflat = X_full.shape[0]

        # (optional) pre-draw MC samples for quantile() if you use MC there
        self._mc_Z = np.random.normal(0.0, 1.0, size=self.mc_R).astype(np.float32)
        self._mc_T = t_dist.rvs(df=3, size=self.mc_R).astype(np.float32)

        print(f"X_full shape: {X_full.shape}, y_full shape: {y_full.shape}")
        return X_full, y_full
