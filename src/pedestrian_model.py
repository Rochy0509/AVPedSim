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


ARRIVAL_RATE       = 0.08   # λ: avg pedestrians per second (1 every ~12s)
CROSSING_TIPPING   = 2.24   # mean time-to-crossing-decision (Nagulapati et al., 2025
CROSSING_STEEPNESS = 0.8    # k: calibrated to observed range 0.81–3.76s
WALK_SPEED_MEAN    = 1.34   # m/s (Weidmann model mean)
WALK_SPEED_STD     = 0.26   # m/s standard deviation
WALK_SPEED_MIN     = 0.5    # m/s minimum (slow pedestrian)
WALK_SPEED_MAX     = 2.5    # m/s maximum (rushing pedestrian)

WALKING_SPEED = 1.2 #m/s stable crossing pace
RUNNING_SPEED = 2.2 #m/s rushed crossing under perceived danger
TTC_DANGER_THRESHOLD = 3.0 # s TTC triggers RUNNING

ROAD_LEFT_EDGE  = -4.0   # left curb (where pedestrians start)
ROAD_RIGHT_EDGE =  4.0   # right curb (where pedestrians exit)
CROSSWALK_X_MIN = 10.0   # start of crosswalk zone
CROSSWALK_X_MAX = 15.0   # end of crosswalk zone

# Jaywalking parameters
JAYWALK_PROBABILITY = 0.25  # 25% of crossers are jaywalkers
JAYWALK_X_MIN = -5.0        # earliest jaywalking start along road
JAYWALK_X_MAX = 30.0        # latest jaywalking start along road

# Sidewalk walker parameters
SIDEWALK_ARRIVAL_RATE = 0.05   # avg sidewalk walkers per second
SIDEWALK_SPEED_MEAN   = 1.0    # m/s along road axis
SIDEWALK_Y_OFFSET     = 1.5    # how far from road edge sidewalk walkers stay
SIDEWALK_X_MIN        = -30.0  # spawn range
SIDEWALK_X_MAX        =  35.0  # despawn range

# Social Force repulsion parameters (Helbing & Molnár, via Yang et al., 2024)
SFM_A = 2.0   # repulsion strength m/s^2
SFM_B = 3.0   # repulsion decay distance m


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
    crossing_start_time: float = 0.0 #sim time when crossing 
    ttc_at_crossing_start: float = 999.0
    current_ttc: float = 999.0 # default = no threat
    jaywalker:   bool  = False   # True if crossing outside crosswalk
    sidewalk_walker: bool = False # True if just walking along sidewalk


def sample_walk_speed() -> float:
    """
    Sample a walking speed from a Truncated Gaussian (Weidmann model).
    Mean = 1.34 m/s, Std = 0.26 m/s, clipped to [0.5, 2.5] m/s.
    Unlike random.gauss + clamping, this keeps the distribution shape clean. (I first tried clamping, but it creates a weird  max speeds.)
    """
    a = (WALK_SPEED_MIN - WALK_SPEED_MEAN) / WALK_SPEED_STD
    b = (WALK_SPEED_MAX - WALK_SPEED_MEAN) / WALK_SPEED_STD
    return float(truncnorm.rvs(a, b, loc=WALK_SPEED_MEAN, scale=WALK_SPEED_STD))


def crossing_probability(wait_time: float) -> float:
    """
    Logistic function: probability of crossing given how long the pedestrian waited.
    Calibrated to Nagulapati et al. (2025) VR study.
    - At t=0.81s → ~15%  (earliest observed crossings)
    - At t=2.24s → 50%   (mean time-to-crossing-decision)
    - At t=3.76s → ~90%  (upper range of observed decisions)
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
        self.recently_exited  = []
        self.time_until_spawn = time_until_next_arrival()
        self.time_until_sidewalk_spawn = random.expovariate(SIDEWALK_ARRIVAL_RATE)

    # Public API

    def update(self, current_time: float, dt: float,
               vehicle_pos: tuple = (0.0, 0.0)) -> None:
        self.spawn_if_needed(current_time, dt)
        self.spawn_sidewalk_walker(current_time, dt)
        self.update_pedestrians(dt, current_time, vehicle_pos)
        self.remove_exited()

    def get_active_pedestrians(self) -> list:
        """Returns all pedestrians still in the scene (WAITING or CROSSING)."""
        return [p for p in self.pedestrians if p.active]

    def get_crossing_pedestrians(self) -> list:
        """Returns only pedestrians actively crossing. Used by vehicle decision algorithm."""
        return [p for p in self.pedestrians
            if p.active and p.state in ("CROSSING", "RUNNING")]

    #Internal methods

    def spawn_if_needed(self, current_time: float, dt: float) -> None:
        self.time_until_spawn -= dt
        if self.time_until_spawn <= 0:
            is_jaywalker = random.random() < JAYWALK_PROBABILITY
            if is_jaywalker:
                # Jaywalker: random x outside the crosswalk zone
                x_pos = random.choice([
                    random.uniform(JAYWALK_X_MIN, CROSSWALK_X_MIN - 3.0),
                    random.uniform(CROSSWALK_X_MAX + 3.0, JAYWALK_X_MAX)
                ])
            else:
                x_pos = random_crosswalk_x()

            ped = Pedestrian(
                id         = self.next_id,
                x          = x_pos,
                y          = ROAD_LEFT_EDGE,
                walk_speed = sample_walk_speed(),
                spawn_time = current_time,
                jaywalker  = is_jaywalker
            )
            self.pedestrians.append(ped)
            self.next_id += 1
            tag = "JAYWALKER" if is_jaywalker else "crosswalk"
            print(f"[{current_time:.1f}s] Pedestrian {ped.id} spawned ({tag}) | speed={ped.walk_speed:.2f} m/s")
            self.time_until_spawn = time_until_next_arrival()

    def spawn_sidewalk_walker(self, current_time: float, dt: float) -> None:
        self.time_until_sidewalk_spawn -= dt
        if self.time_until_sidewalk_spawn <= 0:
            # Pick a random side (top or bottom sidewalk)
            side = random.choice(["top", "bottom"])
            if side == "top":
                y_pos = ROAD_RIGHT_EDGE + SIDEWALK_Y_OFFSET
                direction = random.choice([-1, 1])
            else:
                y_pos = ROAD_LEFT_EDGE - SIDEWALK_Y_OFFSET
                direction = random.choice([-1, 1])

            x_start = SIDEWALK_X_MIN if direction > 0 else SIDEWALK_X_MAX

            ped = Pedestrian(
                id             = self.next_id,
                x              = x_start,
                y              = y_pos,
                vx             = direction * SIDEWALK_SPEED_MEAN,
                walk_speed     = SIDEWALK_SPEED_MEAN,
                spawn_time     = current_time,
                state          = "WALKING",
                sidewalk_walker = True,
            )
            self.pedestrians.append(ped)
            self.next_id += 1
            self.time_until_sidewalk_spawn = random.expovariate(SIDEWALK_ARRIVAL_RATE)

    def update_pedestrians(self, dt: float, current_time: float,
                           vehicle_pos: tuple = (0.0, 0.0)) -> None:
        for ped in self.pedestrians:
            if not ped.active:
                continue
            if ped.state == "WALKING":
                self.update_sidewalk_walker(ped, dt)
            elif ped.state == "WAITING":
                self.update_waiting(ped, dt, current_time)
            elif ped.state == "CROSSING":
                self.update_crossing(ped, dt)
            elif ped.state == "RUNNING":
                self.update_running(ped, dt, vehicle_pos)

    def update_waiting(self, ped: Pedestrian, dt: float, current_time: float) -> None:
        ped.waiting_time += dt
        p_cross = crossing_probability(ped.waiting_time) * dt
        if random.random() < p_cross:
            ped.state    = "CROSSING"
            ped.crossing = True
            ped.vy       = ped.walk_speed
            ped.crossing_start_time = current_time
            ped.ttc_at_crossing_start  = ped.current_ttc 
            print(f"Pedestrian {ped.id} CROSSING | waited {ped.waiting_time:.1f}s")

    def update_crossing(self, ped: Pedestrian, dt: float) -> None:

        if ped.current_ttc < TTC_DANGER_THRESHOLD:
            ped.state = "RUNNING"
            print(f"Pedestrian {ped.id} RUNNING | TTC={ped.current_ttc:.1f}s")
            return

        ped.vy  = WALKING_SPEED
        ped.y  += ped.vy * dt
        if ped.y >= ROAD_RIGHT_EDGE:
            ped.state    = "EXITED"
            ped.crossing = False
            ped.active   = False
            ped.vy       = 0.0
            print(f"Pedestrian {ped.id} EXITED")
    
    def update_running(self, ped: Pedestrian, dt: float,
                       vehicle_pos: tuple = (0.0, 0.0)) -> None:
        # Relax back to walking if danger has passed
        if ped.current_ttc >= TTC_DANGER_THRESHOLD:
            ped.state = "CROSSING"
            return

        boost = compute_sfm_repulsion(ped, vehicle_pos)
        ped.vy = min(RUNNING_SPEED + boost, WALK_SPEED_MAX)
        ped.y += ped.vy * dt

        if ped.y >= ROAD_RIGHT_EDGE:
            ped.state    = "EXITED"
            ped.crossing = False
            ped.active   = False
            ped.vy       = 0.0
            print(f"Pedestrian {ped.id} EXITED")

    def update_sidewalk_walker(self, ped: Pedestrian, dt: float) -> None:
        ped.x += ped.vx * dt
        # Remove when walked off the scene
        if ped.x > SIDEWALK_X_MAX or ped.x < SIDEWALK_X_MIN:
            ped.state  = "EXITED"
            ped.active = False

    def remove_exited(self) -> None:
        # Save exited peds before removing — simulation.py reads this
        self.recently_exited = [p for p in self.pedestrians if not p.active]
        self.pedestrians     = [p for p in self.pedestrians if p.active]

def compute_sfm_repulsion(ped: Pedestrian, vehicle_pos: tuple) -> float:
    dx = ped.x - vehicle_pos[0]
    dy = ped.y - vehicle_pos[1]
    distance = math.sqrt(dx**2 + dy**2)

    #avoid division by zero
    if distance < 0.1:
        distance = 0.1

    repulsion = SFM_A * math.exp(-distance / SFM_B)
    return repulsion



def compute_safety_margin(ped: Pedestrian, vehicle_ttc: float, current_time: float) -> float:
    crossing_duration = current_time - ped.crossing_start_time
    return vehicle_ttc - crossing_duration

