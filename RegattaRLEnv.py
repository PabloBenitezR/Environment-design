import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


# ---------------------- Config ----------------------
@dataclass
class RegattaConfig:
    distance_total: float = 500.0      # meters to finish
    dt: float = 0.1                    # seconds per env step (one stroke decision)
    mass: float = 85.0                 # kg (rower + shell effective)
    Fmax: float = 900.0                # N, max push at full catch & max effort
    k_force_vel: float = 0.15          # force-velocity coefficient (higher => less force at high v)
    beta_tired: float = 1.0            # energy->force exponent; >1 makes fatigue harsher
    cd_drag: float = 0.06              # quadratic water drag
    cd_boost_t: float = 2.0            # seconds with extra start drag
    cd_start: float = 0.10             # drag during first cd_boost_t seconds
    Emax: float = 100.0                # energy budget units
    eff_cost_c0: float = 0.03          # activation cost coefficient
    efficiency: float = 0.25           # mech->metabolic efficiency
    recovery_per_step: float = 0.10    # energy recovered when not pushing 
    reward_mode: str = "time"          # "time", "time_energy", or "vector"
    lambda_energy: float = 0.01        # weight for energy penalty in scalar reward
    max_steps: int = 10000


# ---------------------- Environment ----------------------
class RegattaRLEnv(gym.Env):
    """
    Hybrid-control regatta environment for RL / MORL.

    - Agent chooses both catch depth and effort each step.
    - Each env step represents one stroke decision window of length dt.
    - Physics produces thrust only during the drive implied by chosen depth & effort.
    - Rewards can be scalar (time, time+energy) or vector (time, energy).
    """

    metadata = {"render_modes": [None]}

    def __init__(self, cfg: Optional[RegattaConfig] = None):
        super().__init__()
        self.cfg = cfg or RegattaConfig()

        # Action: 2D Box (depth selector, effort)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: distance, velocity, energy, last_depth, time
        high = np.array([
            np.inf,  # distance
            np.inf,  # velocity
            self.cfg.Emax,  # energy
            1.0,     # last_depth
            np.inf,  # time
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=high, dtype=np.float32)

        # Internal state
        self.distance: float = self.cfg.distance_total
        self.velocity: float = 0.0
        self.energy: float = self.cfg.Emax
        self.time_s: float = 0.0
        self.last_depth: float = 0.5
        self.steps: int = 0

    # ---------------- Gym API ----------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.distance = float(self.cfg.distance_total)
        self.velocity = 0.0
        self.energy = float(self.cfg.Emax)
        self.time_s = 0.0
        self.last_depth = 0.5
        self.steps = 0
        obs = self._obs()
        info = {}
        return obs, info

    def step(self, action):
        self.steps += 1
        dt = self.cfg.dt
        self.time_s += dt

        # Parse action
        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, 0.0, 1.0)
        depth_idx = int(np.floor(a[0] * 3.0))  # 0,1,2 for [0,1)
        if depth_idx >= 3: depth_idx = 2
        depth = [0.5, 0.75, 1.0][depth_idx]
        effort = float(a[1])

        # Force-velocity effect
        k = self.cfg.k_force_vel
        tired = (self.energy / self.cfg.Emax) ** self.cfg.beta_tired
        baseF = effort * self.cfg.Fmax * depth * tired / (1.0 + k * self.velocity)
        F = max(0.0, baseF)

        # Drag (higher at the beginning for cd_boost_t seconds)
        cd = self.cfg.cd_start if self.time_s <= self.cfg.cd_boost_t else self.cfg.cd_drag
        a_lin = (F / self.cfg.mass) - cd * (self.velocity ** 2)
        self.velocity = max(0.0, self.velocity + a_lin * dt)

        # Progress
        self.distance = max(0.0, self.distance - self.velocity * dt)

        # Energy cost: mechanical work / efficiency + activation cost
        work = F * (self.velocity * dt)  # J
        energy_cost = work / max(self.cfg.efficiency, 1e-6) + self.cfg.eff_cost_c0 * (effort ** 2)
        # Recovery if very low effort (coast)
        if effort < 0.05:
            rec = self.cfg.recovery_per_step * (self.cfg.Emax - self.energy)
        else:
            rec = 0.0
        self.energy = float(np.clip(self.energy - energy_cost + rec, 0.0, self.cfg.Emax))

        # Rewards
        r_time = -dt
        r_energy = -energy_cost

        if self.cfg.reward_mode == "vector":
            reward = np.array([r_time, r_energy], dtype=np.float32)
        elif self.cfg.reward_mode == "time_energy":
            reward = float(r_time + self.cfg.lambda_energy * r_energy)  # scalarized
        else:  # "time"
            reward = float(r_time)

        terminated = self.distance <= 0.0
        truncated = self.steps >= self.cfg.max_steps

        self.last_depth = depth
        obs = self._obs()
        info = {
            "force": F,
            "energy_cost": energy_cost,
            "depth": depth,
            "effort": effort,
        }
        return obs, reward, bool(terminated), bool(truncated), info

    def _obs(self) -> np.ndarray:
        return np.array([
            self.distance,
            self.velocity,
            self.energy,
            self.last_depth,
            self.time_s,
        ], dtype=np.float32)

    def render(self):
        # Headless env for RL training; integrate pygame later if desired
        pass

    def close(self):
        pass


# ---------------------- Demo ----------------------
if __name__ == "__main__":
    # Vector reward demo
    env = RegattaRLEnv(RegattaConfig(distance_total=200.0, reward_mode="vector"))
    obs, info = env.reset()
    done = False
    Rt = np.zeros(2, dtype=np.float64)
    while not done:
        # naive heuristic: deeper catch when slow, lighter when fast
        depth_u = 0.0 if obs[1] > 4.0 else (0.5 if obs[1] > 2.0 else 0.9)
        effort = 1.0 if obs[2] > env.cfg.Emax * 0.2 else 0.4
        a = np.array([depth_u, effort], dtype=np.float32)
        obs, r, term, trunc, info = env.step(a)
        if isinstance(r, np.ndarray):
            Rt += r
        else:
            Rt[0] += r
        done = term or trunc
    print("Finished in", env.time_s, "s, returns=", Rt, "energy left=", env.energy)
