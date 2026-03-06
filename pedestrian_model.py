"""
pedestrian_model.py
-------------------
Stochastic Pedestrian Model for Self-Driving Car Simulation
Models random pedestrian arrival, crossing decisions, and walking speeds.

Key stochastic components:
  - Poisson process    → random pedestrian arrival times
  - Logistic function  → probability of crossing after waiting t seconds
  - Truncated Gaussian → realistic walking speed distribution (Weidmann model)
  - State machine      → WAITING → CROSSING → EXITED
"""

import random
import math
from dataclasses import dataclass
from scipy.stats import truncnorm


# -----------------------------
# Model Parameters
# -----------------------------

ARRIVAL_RATE       = 0.08   # λ: avg pedestrians per second (1 every ~12s)
CROSSING_TIPPING   = 8.0    # t0: wait time (s) where crossing prob = 50%
CROSSING_STEEPNESS = 2.5    # k: how sharply probability rises around t0
WALK_SPEED_MEAN    = 1.34   # m/s (Weidmann model mean)
WALK_SPEED_STD     = 0.26   # m/s standard deviation
WALK_SPEED_MIN     = 0.5    # m/s minimum (slow pedestrian)
WALK_SPEED_MAX     = 2.5    # m/s maximum (rushing pedestrian)

ROAD_LEFT_EDGE  = -4.0   # left curb (where pedestrians start)
ROAD_RIGHT_EDGE =  4.0   # right curb (where pedestrians exit)
CROSSWALK_X_MIN = 10.0   # start of crosswalk zone
CROSSWALK_X_MAX = 15.0   # end of crosswalk zone


# -----------------------------
# Pedestrian State
# -----------------------------

@dataclass
class Pedestrian:
    id:           int
    x:            float    # position along road axis (meters)
    y:            float    # position across road (meters)
    vx:           float = 0.0     # velocity along road axis (usually 0)
    vy:           float = 0.0     # velocity across road (m/s when crossing)
    state:        str   = "WAITING"
    waiting_time: float = 0.0     # seconds spent waiting at curb
    spawn_time:   float = 0.0     # sim clock time when spawned
    walk_speed:   float = 1.34    # m/s, sampled once at spawn
    crossing:     bool  = False   # True when state == CROSSING
    active:       bool  = True    # False when state == EXITED


# -----------------------------
# Random Sampling Functions
# -----------------------------

def sample_walk_speed() -> float:
    """
    Sample a walking speed from a Truncated Gaussian (Weidmann model).
    Mean = 1.34 m/s, Std = 0.26 m/s, clipped to [0.5, 2.5] m/s.
    Unlike random.gauss + clamping, this keeps the distribution shape clean.
    """
    a = (WALK_SPEED_MIN - WALK_SPEED_MEAN) / WALK_SPEED_STD
    b = (WALK_SPEED_MAX - WALK_SPEED_MEAN) / WALK_SPEED_STD
    return float(truncnorm.rvs(a, b, loc=WALK_SPEED_MEAN, scale=WALK_SPEED_STD))


def crossing_probability(wait_time: float) -> float:
    """
    Logistic function: probability of crossing given how long the pedestrian waited.
    - At t=0s  → ~4%  (unlikely to immediately cross)
    - At t=8s  → 50%  (coin flip, the tipping point)
    - At t=16s → ~96% (very likely crossing soon)
    """
    exponent = -(wait_time - CROSSING_TIPPING) / CROSSING_STEEPNESS
    return 1.0 / (1.0 + math.exp(exponent))


def time_until_next_arrival() -> float:
    """
    Sample the gap until the next pedestrian arrives.
    Uses exponential distribution — the natural gap for a Poisson process.
    """
    return random.expovariate(ARRIVAL_RATE)


def random_crosswalk_x() -> float:
    """Pick a random x position along the crosswalk."""
    return random.uniform(CROSSWALK_X_MIN, CROSSWALK_X_MAX)


# -----------------------------
# Pedestrian Manager
# -----------------------------

class PedestrianManager:
    """
    Manages the full lifecycle of all pedestrians in the simulation.

    Every simulation tick, call:
        manager.update(current_time, dt)

    To get pedestrians for rendering or sensors:
        manager.get_active_pedestrians()   → WAITING + CROSSING
        manager.get_crossing_pedestrians() → CROSSING only
    """

    def __init__(self):
        self.pedestrians = []
        self.next_id = 0
        self.time_until_spawn = time_until_next_arrival()

    # ── Public API (used by renderer and Student B's sensor module) ──

    def update(self, current_time: float, dt: float) -> None:
        self.spawn_if_needed(current_time, dt)
        self.update_pedestrians(dt)
        self.remove_exited()

    def get_active_pedestrians(self) -> list:
        """Returns all pedestrians still in the scene (WAITING or CROSSING)."""
        return [p for p in self.pedestrians if p.active]

    def get_crossing_pedestrians(self) -> list:
        """Returns only pedestrians actively crossing. Used by vehicle decision algorithm."""
        return [p for p in self.pedestrians if p.active and p.state == "CROSSING"]

    # ── Internal methods ──

    def spawn_if_needed(self, current_time: float, dt: float) -> None:
        self.time_until_spawn -= dt
        if self.time_until_spawn <= 0:
            ped = Pedestrian(
                id         = self.next_id,
                x          = random_crosswalk_x(),
                y          = ROAD_LEFT_EDGE,
                walk_speed = sample_walk_speed(),
                spawn_time = current_time
            )
            self.pedestrians.append(ped)
            self.next_id += 1
            print(f"[{current_time:.1f}s] Pedestrian {ped.id} spawned | speed={ped.walk_speed:.2f} m/s")
            self.time_until_spawn = time_until_next_arrival()

    def update_pedestrians(self, dt: float) -> None:
        for ped in self.pedestrians:
            if not ped.active:
                continue
            if ped.state == "WAITING":
                self.update_waiting(ped, dt)
            elif ped.state == "CROSSING":
                self.update_crossing(ped, dt)

    def update_waiting(self, ped: Pedestrian, dt: float) -> None:
        ped.waiting_time += dt
        p_cross = crossing_probability(ped.waiting_time) * dt
        if random.random() < p_cross:
            ped.state    = "CROSSING"
            ped.crossing = True
            ped.vy       = ped.walk_speed
            print(f"  → Pedestrian {ped.id} CROSSING | waited {ped.waiting_time:.1f}s")

    def update_crossing(self, ped: Pedestrian, dt: float) -> None:
        ped.y += ped.vy * dt
        if ped.y >= ROAD_RIGHT_EDGE:
            ped.state    = "EXITED"
            ped.crossing = False
            ped.active   = False
            ped.vy       = 0.0
            print(f"  → Pedestrian {ped.id} EXITED")

    def remove_exited(self) -> None:
        self.pedestrians = [p for p in self.pedestrians if p.active]


# -----------------------------
# Quick Test
# -----------------------------

if __name__ == "__main__":
    print("=== Pedestrian Model Test (30 second simulation) ===\n")

    manager  = PedestrianManager()
    sim_time = 0.0
    dt       = 0.1
    end_time = 30.0

    while sim_time < end_time:
        manager.update(sim_time, dt)
        sim_time += dt

    print(f"\n=== Done. Total pedestrians spawned: {manager.next_id} ===")