'''
Neural Models
'''
import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
import itertools
import math

from utils import NormalizeImg, Flatten, _get_out_shape

import warnings

try:
    from tigramite.data_processing import DataFrame as TGDataFrame
    from tigramite.pcmci import PCMCI as TGPCMCI
    from tigramite.independence_tests.parcorr import ParCorr as TGParCorr
    _TIGRAMITE = True
except Exception:
    _TIGRAMITE = False


class Encoder(nn.Module):
    def __init__(self, in_channels, num_channels, img_size, latent_dim):
        super().__init__()
        
        # convolut
        conv_layers = nn.Sequential(
            NormalizeImg(),
            nn.Conv2d(in_channels, num_channels, kernel_size=7, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2), nn.ReLU()
        )
        
        # Calculate the output size of the convolutional layers
        # This is needed to know the input size for the linear layer
        conv_out_shape = _get_out_shape((in_channels, img_size, img_size), conv_layers)
        
        # Combine all the layers into a single network
        self.encoder = nn.Sequential(
            conv_layers,
            Flatten(),
            # nn.Linear(torch.prod(conv_out_shape), latent_dim)
            nn.Linear(np.prod(conv_out_shape), latent_dim)
        )

    def forward(self, x):
        # The forward pass is as simple as calling the sequential container
        return self.encoder(x)


class _PCMCI():
    """
    Lightweight placeholder for PCMCI-style graph refinement.
    Here we use absolute Pearson correlations between inputs U=[z_t, a_t]
    and targets V=z_{t+1} to prune edges. Replace with a proper PCMCI if desired.
    """
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    @torch.no_grad()
    def estimate(self, z_t: torch.Tensor, a_t: torch.Tensor, z_t_1: torch.Tensor) -> torch.Tensor:
        # z_t: [B, latent_dim], a_t: [B, action_dim], z_t_1: [B, latent_dim]
        U = torch.cat([z_t, a_t], dim=-1)  # [B, U_dim]
        U = U - U.mean(dim=0, keepdim=True) # mean centering
        V = z_t_1 - z_t_1.mean(dim=0, keepdim=True)
        # corr(U_i, V_j) over batch
        num = U.T @ V  # [U_dim, latent_dim]
        u_norm = torch.sqrt(U.pow(2).sum(dim=0) + 1e-8)        # [U_dim]
        v_norm = torch.sqrt(V.pow(2).sum(dim=0) + 1e-8)        # [latent_dim]
        denom = u_norm.view(-1, 1) @ v_norm.view(1, -1)          # [U_dim, latent_dim]
        corr = num / (denom + 1e-8)
        adj = (corr.abs() >= self.threshold).float()
        return adj
    

class PCMCI():
    """
    Proper (simplified) PCMCI for edges U_t -> V_{t+1}.
    - Condition selection: iterative PC-style pruning with ParCorr CI tests.
    - MCI step: confirm each remaining parent with CI test conditioned on the other parents.
    - FDR (Benjamini–Hochberg) per target variable.
    Notes:
      - This variant assumes lag=1 only (U_t -> V_{t+1}). Extendable via max_lag.
      - Linear-Gaussian CI via partial correlation. Works best if relationships are approx. linear.
    """
    def __init__(
        self,
        alpha: float = 0.05,       # Significance level for edge selection
        pc_alpha: float | None = 0.2,  # Skeleton pruning alpha (None disables PC step)
        use_fdr: bool = True,
        standardize: bool = True,
    ):
        self.alpha = alpha
        self.pc_alpha = pc_alpha
        self.use_fdr = use_fdr
        self.standardize = standardize
        if not _TIGRAMITE:
            raise ImportError(
                "tigramite is required for PCMCI. Install it with `pip install tigramite` "
                "and ensure its dependencies (numpy, scipy, networkx, statsmodels) are available."
            )


    @torch.no_grad()
    def estimate(self, z_t: torch.Tensor, a_t: torch.Tensor, z_t_1: torch.Tensor) -> torch.Tensor:
        # Inputs: [N, Z], [N, A], [N, Z]
        U = torch.cat([z_t, a_t], dim=-1).detach().cpu().double().numpy()  # [N, U_dim]
        V = z_t_1.detach().cpu().double().numpy()                           # [N, Z]
        U = self._standardize(U)
        V = self._standardize(V)

        N, U_dim = U.shape
        _, Z_dim = V.shape

        # Parent selection (PC-style) for each V_j
        parents = [list(range(U_dim)) for _ in range(Z_dim)]
        for j in range(Z_dim):
            # Iterative increase of conditioning set size
            for l in range(0, self.max_cond_set + 1):
                changed = False
                Pj = parents[j].copy()
                for i in Pj:
                    # build candidate conditioning pool excluding i
                    pool = [p for p in parents[j] if p != i]
                    if len(pool) < l:
                        continue
                    # Try subsets of size l; remove edge if any subset yields independence
                    remove_i = False
                    for C in itertools.combinations(pool, l):
                        pval = self._parcorr_pvalue(U[:, i], V[:, j], U[:, C] if len(C) > 0 else None)
                        if pval > self.alpha:
                            parents[j].remove(i)
                            changed = True
                            remove_i = True
                            break
                    if remove_i:
                        continue
                if not changed:
                    break

        # MCI confirmation step + FDR
        adj = np.zeros((U_dim, Z_dim), dtype=np.float32)
        for j in range(Z_dim):
            Pj = parents[j]
            if len(Pj) == 0:
                continue
            pvals = []
            edges = []
            for i in Pj:
                cond = [k for k in Pj if k != i]
                pval = self._parcorr_pvalue(U[:, i], V[:, j], U[:, cond] if len(cond) > 0 else None)
                pvals.append(pval)
                edges.append(i)
            if self.use_fdr:
                keep_mask = self._bh_fdr(np.array(pvals), self.alpha)
            else:
                keep_mask = np.array(pvals) <= self.alpha
            for keep, i in zip(keep_mask, edges):
                if keep:
                    adj[i, j] = 1.0

        return torch.from_numpy(adj)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-12
        return (X - mu) / sd

    def _parcorr_pvalue(self, x: np.ndarray, y: np.ndarray, Z: np.ndarray | None) -> float:
        """
        Partial-correlation test via residual correlation with ridge regression.
        Returns two-sided p-value using Fisher z approx (Normal).
        """
        # Ensure shapes [N,], [N,], [N,k]
        if Z is None or Z.size == 0:
            r = self._pearsonr(x, y)
            n_eff = x.shape[0]
            if n_eff <= 3:
                return 1.0
            z = 0.5 * math.log((1 + r + 1e-12) / (1 - r + 1e-12)) * math.sqrt(max(n_eff - 3, 1))
            p = self._normal_2sided_p_from_z(z)
            return p

        # Regress x and y on Z with ridge, take residuals, correlate
        xr = x - self._ridge_proj(Z, x, self.ridge)
        yr = y - self._ridge_proj(Z, y, self.ridge)
        r = self._pearsonr(xr, yr)
        n, k = Z.shape
        n_eff = n - k - 3
        if n_eff <= 1:
            return 1.0
        z = 0.5 * math.log((1 + r + 1e-12) / (1 - r + 1e-12)) * math.sqrt(max(n_eff, 1))
        p = self._normal_2sided_p_from_z(z)
        return p

    def _ridge_proj(self, Z: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
        # Compute Z @ beta (predictions) using ridge: beta = (Z^T Z + λI)^-1 Z^T y
        # Avoid explicit inverse: solve (G + λI) beta = Z^T y
        G = Z.T @ Z
        G.flat[::G.shape[0] + 1] += ridge
        rhs = Z.T @ y
        beta = np.linalg.solve(G, rhs)
        return Z @ beta

    def _pearsonr(self, x: np.ndarray, y: np.ndarray) -> float:
        x = x - x.mean()
        y = y - y.mean()
        num = float((x * y).sum())
        den = math.sqrt(float((x * x).sum()) * float((y * y).sum())) + 1e-12
        r = max(min(num / den, 1.0), -1.0)
        return r

    def _normal_2sided_p_from_z(self, z: float) -> float:
        # p = 2*(1 - Phi(|z|)) using erfc
        return float(math.erfc(abs(z) / math.sqrt(2)))

    def _bh_fdr(self, pvals: np.ndarray, alpha: float) -> np.ndarray:
        # Benjamini–Hochberg per-target
        m = len(pvals)
        order = np.argsort(pvals)
        ranked = pvals[order]
        thresh = alpha * (np.arange(1, m + 1) / m)
        below = ranked <= thresh
        if not np.any(below):
            return np.zeros(m, dtype=bool)
        k = np.max(np.where(below)[0])
        cutoff = ranked[k]
        return pvals <= cutoff


    @torch.no_grad()
    def estimate(self, z_t: torch.Tensor, a_t: torch.Tensor, z_t_1: torch.Tensor) -> torch.Tensor:
        # z_t: [N,Z], a_t: [N,A], z_t_1: [N,Z]
        N, Z = z_t.shape
        A = a_t.shape[-1]
        U_dim, V_dim = Z + A, Z

        if not _TIGRAMITE:
            # Shouldn't happen because constructor already enforces tigramite presence
            raise ImportError("tigramite must be installed to use PCMCI. No fallback available.")

        # Build a time series X of shape [T, Z+A] such that:
        #   X[t, :Z]   = z_t
        #   X[t, Z:]   = a_t
        #   X[t+1, :Z] = z_{t+1}
        # This aligns parents at lag 1 with children at current time.
        T = N + 1
        X = np.zeros((T, Z + A), dtype=np.float64)
        X[:-1, :Z] = z_t.detach().cpu().numpy()
        X[:-1, Z:] = a_t.detach().cpu().numpy()
        X[1:, :Z] = z_t_1.detach().cpu().numpy()

        if self.standardize:
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, keepdims=True) + 1e-12
            X = (X - mu) / sd

        df = TGDataFrame(X)
        ci_test = TGParCorr(significance="analytic")
        pcmci = TGPCMCI(dataframe=df, cond_ind_test=ci_test)

        results = pcmci.run_pcmci(
            tau_max=1,
            pc_alpha=self.pc_alpha,
            fdr_method="fdr_bh" if self.use_fdr else None,
        )

        # p/q-matrices: shape [Nvars, Nvars, tau_max]
        # Entry [i, j, 1] corresponds to X_i(t-1) -> X_j(t)
        mat = results.get("q_matrix", None)
        if mat is None:
            mat = results["p_matrix"]

        lag1 = mat[:, :, 1]  # [Nvars, Nvars]
        sig = (lag1 <= self.alpha)

        # Build adjacency [U_dim, V_dim]; parents i in [0..Z+A-1], targets j in [0..Z-1]
        adj = np.zeros((U_dim, V_dim), dtype=np.float32)
        for i in range(Z + A):
            for j in range(Z):
                adj[i, j] = 1.0 if sig[i, j] else 0.0

        # Enforce no self-loops z_i(t) -> z_i(t+1)
        for j in range(Z):
            adj[j, j] = 0.0

        return torch.from_numpy(adj).to(dtype=torch.float32)


# import torch
# import numpy as np
# from scipy.stats import pearsonr

# # Example: Partial correlation function
# def partial_correlation(x, y, z):
#     """
#     Compute partial correlation between x and y given z.
#     """
#     x, y, z = torch.tensor(x), torch.tensor(y), torch.tensor(z)
#     x_residual = x - torch.matmul(torch.linalg.pinv(z), z.T @ x)
#     y_residual = y - torch.matmul(torch.linalg.pinv(z), z.T @ y)
#     return pearsonr(x_residual.numpy(), y_residual.numpy())[0]

# # PCMCI Algorithm (Simplified)
# def pcmci(data, max_lag=2, alpha=0.05):
#     """
#     Perform PCMCI causal discovery on time-series data.
#     Args:
#         data: Time-series data (NumPy array or PyTorch tensor).
#         max_lag: Maximum lag to consider.
#         alpha: Significance level for independence tests.
#     Returns:
#         Causal graph (adjacency matrix).
#     """
#     n_vars = data.shape[1]
#     causal_graph = torch.zeros((n_vars, n_vars, max_lag + 1))

#     for i in range(n_vars):
#         for j in range(n_vars):
#             if i == j:
#                 continue
#             for lag in range(1, max_lag + 1):
#                 x = data[lag:, i]
#                 y = data[:-lag, j]
#                 z = data[:-lag, :]  # All other variables as conditioning set
#                 corr = partial_correlation(x, y, z)
#                 if abs(corr) > alpha:  # Threshold for significance
#                     causal_graph[i, j, lag] = corr

#     return causal_graph

# # Example usage
# data = np.random.rand(100, 3)  # 100 time points, 3 variables
# causal_graph = pcmci(data)
# print("Causal Graph:", causal_graph)





class CausalGraph():
    """
    Causal Graph
    Data structure for constraining the latent causal relationships between 
    elements of the latent state/observation vector.
    """
    
    def __init__(self, latent_state_dim, latent_action_dim):
        self.U_dim = latent_state_dim + latent_action_dim 
        self.V_dim = latent_state_dim 
        # init latent causal graph G as a complete directed graph (U -> V)
        # Algorithm 1 starts from a fully connected graph, so initialize all ones.
        self.adjacency_matrix = torch.ones((self.U_dim, self.V_dim), dtype=torch.float32)

    def to(self, device):
        self.adjacency_matrix = self.adjacency_matrix.to(device)
        return self

    def update(self, new_adj: torch.Tensor):
        assert new_adj.shape == self.adjacency_matrix.shape
        self.adjacency_matrix.copy_(new_adj)

    def __call__(self):
        return self.adjacency_matrix

    



class LatentTransitionModel(nn.Module):
    """ 
    Latent Transition Model
    Learns a mapping (z, a) --> (z') constraint by a causal graph G

    models:
        z_i' = f_i(PA(z_i), a, N_i), whereby N_i is a stochastic Gaussian ~N(0, I)
    """
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 64, rng: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        in_dim = latent_dim + action_dim
        # construct on FNN for each latent dimension
        self.fnns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            ) for _ in range(latent_dim)
        ])
        # learnable log-std per latent dim (σ_j = exp(log_std[j]))
        self.log_std = nn.Parameter(torch.full((latent_dim, ), -2.0))

        # Seed RNG for reproducibility (accept integer seeds)
        try:
            torch.manual_seed(int(rng))
        except Exception:
            # if rng is not an int, fall back to no-op
            pass


    def __call__(self, latent_state, action, causal_graph):
        return self.forward(latent_state, action, causal_graph)


    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        # z_t: [B, latent_dim], a_t: [B, action_dim], graph: [U_dim, V_dim]
        # B = z_t.shape[0]
        U = torch.cat([z_t, a_t], dim=-1)  # [B, U_dim]
        outs = []
        # iterate through |V| columns of adjacency list [|U|, |V|]
        for j in range(self.latent_dim): 
            col_mask = graph[:, j]  # [U_dim]
            # select which factors of U may have a causal effect on V
            masked = U * col_mask.unsqueeze(0)  # [B, U_dim]
            out_j = self.fnns[j](masked) # each column is fed through its dedicated FNN f_j
            std_j = self.log_std[j].exp() # exponentiate log-std
            out_j = out_j + torch.randn_like(out_j) * std_j  # add Gaussian noise N(0, 1) scaled by learned std
            outs.append(out_j)
        z_t_1 = torch.cat(outs, dim=-1)  # [B, latent_dim]
        return z_t_1




class RewardModel(nn.Module):
    """
    r_t = r(z_t, a_t)
    """
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        in_dim = latent_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_t, a_t], dim=-1)
        return self.net(x).squeeze(-1)







if __name__ == "__main__":
    from env import DMCEnv

    env = DMCEnv(domain_name='cartpole', task_name='swingup', from_pixels=True, pixels_only=True, img_size=64)
    env.reset()
    encoder = Encoder(in_channels=3, num_channels=32, img_size=64, latent_dim=8)
    graph = CausalGraph(latent_state_dim=8, latent_action_dim=4)
    print(graph())
    while True:
        action = env.sample_action()
        # print(action) # scalar
        next_state, reward, done, _ = env.step(action)
        print(next_state.shape) 
        latent_next_state = encoder(next_state)
        if done:
            break