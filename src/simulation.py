import sys 
import os
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pedestrian_model import PedestrianManager, compute_safety_margin
from sensor_fusion import SensorFusion
from vehicle_controller import VehicleController, compute_ttc

SIM_DURATION = 120.0
DT = 0.1
NUM_TRIALS = 100
VEHICLE_START = -100.0

CROSSWALK_CENTRE = 12.5
CROSSWALK_X_MAX = 15.0

CRUISE_SPEED = 8.33

def run_trial(seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    #initialize all modules
    manager = PedestrianManager()
    fusion = SensorFusion()
    controller = VehicleController(start_x=VEHICLE_START)

    #per-trial metrics
    safety_margins = []
    crossing_durations = []
    waiting_times = []
    hard_brake_count = 0
    prev_speed = controller.speed
    exited_peds = set()

    sim_time = 0.0

    while sim_time < SIM_DURATION:

        vehicle_pos = (controller.x, 0.0)

        #update pedestrian model
        manager.update(sim_time, DT, vehicle_pos)

        #run sensor fusion
        estimates = fusion.update(vehicle_pos, manager.get_active_pedestrians())

        crossing_peds = manager.get_crossing_pedestrians()
        confirmed_ids = {e["pedestrian_id"] for e in estimates}
        visible_crossing = [
            p for p in crossing_peds
            if p.id in confirmed_ids
        ]

        est_map = {e["pedestrian_id"]: e for e in estimates}
        for ped in manager.get_active_pedestrians():
            if ped.id in est_map:
                est_x = est_map[ped.id]["x_est"]
                ttc = compute_ttc(controller.x, controller.speed, est_x)
                ped.current_ttc = ttc
            else:
                ped.current_ttc = compute_ttc(controller.x, controller.speed, ped.x) 
        
        controller.update(visible_crossing, DT, vehicle_pos)

        if controller.x >= CROSSWALK_X_MAX + 5.0:
            controller.x     = VEHICLE_START
            controller.speed = CRUISE_SPEED

        if controller.speed < prev_speed - (4.5 * DT):
            hard_brake_count += 1

        prev_speed = controller.speed

       

        #check all peds including those about to be removed 
        for ped in manager.recently_exited:
            if ped.id not in exited_peds:
                exited_peds.add(ped.id)
                crossing_durations.append(sim_time - ped.crossing_start_time)
                waiting_times.append(ped.waiting_time)

                if ped.ttc_at_crossing_start < 15.0:
                    margin = compute_safety_margin(
                        ped, ped.ttc_at_crossing_start, sim_time
                    )
                    safety_margins.append(margin)

        sim_time = round(sim_time + DT, 6)

    return {
        "seed":               seed,
        "peds_spawned":       manager.next_id,
        "safety_margins":     safety_margins,
        "crossing_durations": crossing_durations,
        "waiting_times":      waiting_times,
        "hard_brake_count":   hard_brake_count,
        "near_miss_count":    sum(1 for m in safety_margins if m < 0),
        "mean_safety_margin": np.mean(safety_margins) if safety_margins else None,
    }

def run_monte_carlo(num_trials: int = NUM_TRIALS) -> dict:

    all_safety_margins     = []
    all_crossing_durations = []
    all_waiting_times      = []
    all_hard_brakes        = []
    all_near_misses        = []
    all_peds_spawned       = []

    for i in range(num_trials):
        result = run_trial(seed=i)

        all_safety_margins.extend(result["safety_margins"])
        all_crossing_durations.extend(result["crossing_durations"])
        all_waiting_times.extend(result["waiting_times"])
        all_hard_brakes.append(result["hard_brake_count"])
        all_near_misses.append(result["near_miss_count"])
        all_peds_spawned.append(result["peds_spawned"])

        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{num_trials} trials...")

    def report(label, values, benchmark=None, unit=""):
        if not values:
            print(f"{label}: no  data")

            return
        
        arr = np.array(values)
        line = (f"  {label}: "
                f"mean={arr.mean():.3f}{unit}  "
                f"std={arr.std():.3f}{unit}  "
                f"min={arr.min():.3f}{unit}  "
                f"max={arr.max():.3f}{unit}")
        
        if benchmark is not None:
            line += f"[benchmark: {benchmark}{unit}]"
        print(line)

    report("Safety Margin",     all_safety_margins,
    benchmark=0.58,  unit="s")
    report("Crossing Duration", all_crossing_durations,
        benchmark=2.35,  unit="s")
    report("Waiting Time",      all_waiting_times,
        unit="s")
    report("Hard Brakes/trial", all_hard_brakes,
        unit=" events")
    report("Near Misses/trial", all_near_misses,
        unit=" events")
    report("Peds Spawned/trial",all_peds_spawned,
        unit=" peds")
    
    # Near-miss rate as a percentage
    total_crossings = len(all_safety_margins)
    total_near_misses = sum(all_near_misses)
    if total_crossings > 0:
        rate = 100.0 * total_near_misses / total_crossings
        print(f"\n  Near-miss rate: {rate:.1f}% of crossings"
            f"  [across {total_crossings} total crossings]")

    return {
        "safety_margins":     all_safety_margins,
        "crossing_durations": all_crossing_durations,
        "waiting_times":      all_waiting_times,
        "hard_brakes":        all_hard_brakes,
        "near_misses":        all_near_misses,
        "peds_spawned":       all_peds_spawned,
    }


if __name__ == "__main__":
    results = run_monte_carlo(NUM_TRIALS)

