"""
REINFORCE agent with Gaussian policy for missile navigation.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pure NumPy — no external ML frameworks required.

Architecture
------------
- Policy network:  obs → raw_mean  (tanh-squashed → action ∈ [-1,1]³)
- Value network:   obs → V(s)       (scalar baseline for variance reduction)
- Shared log_std:  learnable, per-action-dim, not input-dependent

Update rule: REINFORCE with advantage = G_t - V(s_t).
"""

import warnings
import numpy as np


# ── Small feedforward network ─────────────────────────────────────────────────

class _Net:
    """Two-hidden-layer fully-connected network (ReLU activations)."""

    def __init__(self, in_dim: int, out_dim: int,
                 hidden: int = 64, out_scale: float = 0.01,
                 rng: np.random.Generator = None):
        rng = rng or np.random.default_rng(0)
        he = lambda m: np.sqrt(2.0 / m)   # He initialisation for ReLU

        self.W1 = rng.normal(0, he(in_dim),  (hidden, in_dim))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, he(hidden),  (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0, out_scale,   (out_dim, hidden))
        self.b3 = np.zeros(out_dim)

    def forward(self, x: np.ndarray):
        """x: (in_dim,) → (out, cache)"""
        # RuntimeWarning suppressed: BLAS may raise spurious FPU flags
        # on subnormal intermediates; the results are numerically correct.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            z1 = self.W1 @ x + self.b1
            h1 = np.maximum(0.0, z1)
            z2 = self.W2 @ h1 + self.b2
            h2 = np.maximum(0.0, z2)
            out = self.W3 @ h2 + self.b3
        return out, (x, z1, h1, z2, h2)

    def backward(self, cache, d_out: np.ndarray):
        """Backprop; returns tuple of weight gradients in parameter order."""
        x, z1, h1, z2, h2 = cache
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d_h2 = self.W3.T @ d_out
            d_z2 = d_h2 * (z2 > 0)
            d_h1 = self.W2.T @ d_z2
            d_z1 = d_h1 * (z1 > 0)
        return (
            np.outer(d_z1, x), d_z1,
            np.outer(d_z2, h1), d_z2,
            np.outer(d_out, h2), d_out,
        )

    def apply_grads(self, grads, lr: float):
        self.W1 += lr * grads[0]
        self.b1 += lr * grads[1]
        self.W2 += lr * grads[2]
        self.b2 += lr * grads[3]
        self.W3 += lr * grads[4]
        self.b3 += lr * grads[5]

    def _zero_grads(self):
        return [np.zeros_like(p)
                for p in (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)]


# ── REINFORCE agent ───────────────────────────────────────────────────────────

class REINFORCEAgent:
    """Gaussian-policy REINFORCE agent with value-function baseline.

    Parameters
    ----------
    obs_dim     : observation dimensionality
    act_dim     : action dimensionality
    hidden      : hidden layer size (same for policy and value nets)
    lr_policy   : policy network learning rate
    lr_value    : value network learning rate
    gamma       : discount factor
    init_log_std: initial log σ for the Gaussian policy (σ = exp(log_std))
    seed        : RNG seed for reproducibility
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 64,
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        init_log_std: float = -0.5,
        entropy_coeff: float = 0.01,   # entropy bonus — prevents policy collapse
        l2_output: float = 1e-4,       # L2 on output weights — anchors near initial
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        # Zero-init the output layer so the policy starts as a pure-bias
        # policy (fly in the initial direction, cancel gravity).  The hidden
        # layers still have He-random weights so gradients break symmetry.
        self.policy = _Net(obs_dim, act_dim, hidden, out_scale=0.0, rng=rng)
        self.value  = _Net(obs_dim, 1,       hidden, out_scale=1.0,  rng=rng)
        self.log_std = np.full(act_dim, init_log_std, dtype=float)

        self.lr_policy     = lr_policy
        self.lr_value      = lr_value
        self.gamma         = gamma
        self.act_dim       = act_dim
        self.entropy_coeff = entropy_coeff
        self.l2_output     = l2_output

        # tanh⁻¹(G / (max_g * G)) = tanh⁻¹(1/15) ≈ 0.067 — cancels gravity
        self._gz_action = 0.067

        # RL output is scaled down so it acts as a small correction on top
        # of the proportional-navigation baseline rather than the full action.
        self._correction_scale = 0.8   # large enough for visible terrain deviations

    # ── Inference ────────────────────────────────────────────────────────────

    def _pn_baseline(self, obs: np.ndarray) -> np.ndarray:
        """Proportional navigation baseline.

        Flies toward the target in XY (obs[15:18] = unit vector to target)
        while holding cruise altitude (gravity-cancel on z-axis).
        This gives the agent a free ride to the target; the RL network only
        needs to learn *when* to deviate for terrain avoidance.
        """
        target_dir = obs[21:24].copy()          # unit vector to target (ENU)
        target_dir[2] = self._gz_action         # override z: cancel gravity
        return np.clip(target_dir, -1.0, 1.0)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """PN baseline + learned correction."""
        pn = self._pn_baseline(obs)

        raw, _ = self.policy.forward(obs)
        correction = np.tanh(raw) * self._correction_scale

        action = pn + correction
        if not deterministic:
            std = np.exp(self.log_std)
            action = action + std * np.random.randn(self.act_dim)
        return np.clip(action, -1.0, 1.0)

    # ── Training ─────────────────────────────────────────────────────────────

    def update(self, trajectories: list, grad_clip: float = 1.0) -> dict:
        """Update from a batch of episode trajectories.

        Each trajectory is a dict with:
            "obs"     : np.ndarray (T, obs_dim)
            "actions" : np.ndarray (T, act_dim)
            "rewards" : array-like (T,)

        Returns are normalised *across the full batch* (not per-episode)
        to keep gradients stable regardless of episode length.
        Gradient norms are clipped to `grad_clip`.
        """
        # ── Pass 1: compute all discounted returns ────────────────────────
        all_returns = []
        traj_returns = []
        for traj in trajectories:
            rew_seq = np.asarray(traj["rewards"], dtype=float)
            T = len(rew_seq)
            returns = np.zeros(T)
            G = 0.0
            for t in reversed(range(T)):
                G = rew_seq[t] + self.gamma * G
                returns[t] = G
            traj_returns.append(returns)
            all_returns.extend(returns.tolist())

        # Batch-level normalisation
        all_ret = np.array(all_returns)
        ret_mean = all_ret.mean()
        ret_std  = all_ret.std() + 1e-8
        traj_returns = [(r - ret_mean) / ret_std for r in traj_returns]

        # ── Pass 2: accumulate gradients ─────────────────────────────────
        p_grads  = self.policy._zero_grads()
        v_grads  = self.value._zero_grads()
        ls_grad  = np.zeros_like(self.log_std)

        total_p_loss = 0.0
        total_v_loss = 0.0
        total_steps  = 0

        std = np.exp(self.log_std)

        for traj, returns in zip(trajectories, traj_returns):
            obs_seq = traj["obs"]
            act_seq = traj["actions"]
            T = len(returns)

            for t in range(T):
                obs = obs_seq[t]
                act = act_seq[t]
                ret = float(returns[t])

                # Value baseline
                val_raw, v_cache = self.value.forward(obs)
                baseline  = float(val_raw[0])
                advantage = np.clip(ret - baseline, -10.0, 10.0)

                d_val = 2.0 * np.array([baseline - ret])
                for i, g in enumerate(self.value.backward(v_cache, d_val)):
                    v_grads[i] -= g
                total_v_loss += (baseline - ret) ** 2

                # Policy gradient
                raw, p_cache = self.policy.forward(obs)
                mean = np.tanh(raw)

                d_log_pi_d_mean = (act - mean) / (std ** 2 + 1e-8)
                d_raw = advantage * d_log_pi_d_mean * (1.0 - mean ** 2)
                # Clip per-step gradient signal before backprop
                d_raw = np.clip(d_raw, -1.0, 1.0)

                for i, g in enumerate(self.policy.backward(p_cache, d_raw)):
                    p_grads[i] += g

                ls_grad += advantage * ((act - mean) ** 2 / (std ** 2 + 1e-8) - 1.0)
                # Entropy bonus: ∂H/∂log_σ = 1  →  encourages maintained exploration
                ls_grad += self.entropy_coeff

                total_p_loss -= float(
                    np.sum(-0.5 * ((act - mean) / (std + 1e-8)) ** 2 - self.log_std)
                )
                total_steps += 1

        if total_steps == 0:
            return {}

        inv = 1.0 / total_steps

        # Gradient clipping (global norm)
        def _clip_and_apply(net, grads, lr, sign=1.0):
            grads = [g * inv for g in grads]
            # L2 on output weights and bias — anchors both toward initialisation
            grads[4] = grads[4] - self.l2_output * net.W3
            grads[5] = grads[5] - self.l2_output * net.b3
            total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
            if total_norm > grad_clip:
                grads = [g * grad_clip / total_norm for g in grads]
            net.apply_grads([sign * g for g in grads], lr)

        _clip_and_apply(self.policy, p_grads,  self.lr_policy, sign=1.0)
        _clip_and_apply(self.value,  v_grads,  self.lr_value,  sign=-1.0)

        ls_update = np.clip(self.lr_policy * ls_grad * inv, -0.1, 0.1)
        self.log_std += ls_update
        self.log_std  = np.clip(self.log_std, -3.0, 1.0)

        return {
            "policy_loss":  total_p_loss * inv,
            "value_loss":   total_v_loss * inv,
            "steps":        total_steps,
            "log_std_mean": float(self.log_std.mean()),
        }
