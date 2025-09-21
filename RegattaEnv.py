# packages 
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import pygame
import time
import cv2

@dataclass
class RegattaConfig:
    distance: float = 500.0         # total distance to row (meters)
    dt: float = 0.1                 # time step
    mass: float = 85.0              # mass of the rower (kg)
    Fmax: float = 900.0             # maximum force applied during drive phase (N)
    k_force_vel: float = 0.15       # force-velocity coefficient (higher => less force at high v)
    beta_tired: float = 1.0         # energy->force exponent; >1 makes fatigue harsher
    cd_drag: float = 0.06           # quadratic water drag
    cd_boost_t: float = 2.0         # seconds with extra start drag
    cd_start: float = 0.10          # drag during first cd_boost_t seconds
    Emax: float = 100.0             # energy available 
    eff_cost_c0: float = 0.03       # activation cost coefficient
    efficiency: float = 0.25        # mech->metabolic efficiency
    recovery_per_step: float = 0.10 # energy recovered when not pushing (applied to (Emax-E))
    reward_mode: str = "time"       
    lambda_energy: float = 0.01     # weight for energy penalty in scalar reward
    max_steps: int = 10000

class RegattaEnv(gym.Env):
    """
    Rowing regatta environment for reinforcement learning.
    
    - Common terminations:
        - Rower: the agent.
        - Boat: Single (1X) one rower. Further you will choose the pair (2X,2-), four (4X,4-), and eight (8-).
        - Oars: the oars used by the rower to propel the boat. 
        - Distance: the boat has covered the total distance (e.g., 500m).
        - Time: the boat has been rowing for a maximum time (e.g., 200).
        - Pace: time to takes cover a specific distance (e.g., 2:00/500m).
        - Stroke: the rowing cycle, which consists of a drive phase (pushing the oars) 
                  and a recovery phase (returning the oars to the starting position).
        -Tiredness: the rower gets tired over time, which affects their pace.
        - Recovery: the rower takes a little recover in the recovery phase. 
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: Optional[RegattaConfig] = None, render_mode=None):
        super().__init__()
        self.cfg = cfg or RegattaConfig()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(2)

        # Observation: [distance_remaining, velocity, pace]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        # State
        self.velocity = 0.0
        self.distance_remaining = distance
        self.time = 0
        self.pace = 0.0 # strokes per minute (SPM)
        self.phase = 0.0 # stroke cycle phase [0,1)

        # Pygame
        self.screen = None
        self.clock = None
        self.row_images = [pygame.image.load(f"position_{i}.png") for i in range(1, 5)]
        self.current_frame = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.velocity = 0.0
        self.distance_remaining = self.distance_total
        self.time = 0
        self.pace = 0.0
        self.phase = 0.0
        self.last_time = time.time()
        obs = np.array([self.distance_remaining, self.velocity, self.pace], dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.time += 1

        dt = 1
        
        if action == 1:
            self.pace = + 0.1
        else:
            self.pace *= 0.98  # cansao

        self.pace = np.clip(self.pace, 0, 3)

        #now = time.time()
        #dt = now - self.last_time
        #self.last_time = now

        self.phase += self.pace * dt   
        if self.phase >= 1.0:
            self.phase %= 1.0

        # push phase
        if 0.5 <= self.phase < 1.0:
            self.velocity += 1 * self.pace

        self.velocity *= 0.98 # friction
        
        self.distance_remaining -= self.velocity
        self.distance_remaining = max(self.distance_remaining, 0.0)

        # reward
        reward = -1.0 
        terminated = self.distance_remaining <= 0.0
        truncated = self.time >= 2000

        obs = np.array([self.distance_remaining, self.velocity], dtype=np.float32)
        return obs, reward, terminated, truncated, {"time": self.time, "phase": self.phase}

    def render(self):
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((400, 300))
            self.clock = pygame.time.Clock()

        self.screen.fill((0, 100, 200))  # water 

        # draw boat 
        frame_idx = int(self.phase * 4) % 4
        boat_img = self.row_images[frame_idx]
        rect = boat_img.get_rect(center=(200, 150))
        self.screen.blit(boat_img, rect)

        # text overlay
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"Dist: {int(self.distance_remaining)} Vel: {self.velocity:.2f} Pace: {self.pace*60:.1f} SPM Time: {self.time}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)

        # record frame
        if self.record_video:
            frame = pygame.surfarray.array3d(self.screen)  # (width, height, 3)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self.video_writer is None:
                h, w = frame.shape[:2]
                self.video_writer = cv2.VideoWriter(
                    self.video_path, cv2.VideoWriter_fourcc(*"XVID"), 30, (w, h)
                )
            self.video_writer.write(frame)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None