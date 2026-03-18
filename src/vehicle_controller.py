from pedestrian_model import Pedestrian, compute_safety_margin

CRUISE_SPEED = 8.33 #m/s
SOFT_BRAKE_DECEL = 2.0 # m/s^2 deceleration 
HARD_BRAKE_DECEL = 5.0 # m/s^2 deceleration

COMFORT_TTC = 5.0 # s
BRAKE_TTC = 3.0 # s


def compute_ttc(vehicle_x: float, vehicle_speed: float,
                crosswalk_x: float) -> float:
    
    if vehicle_speed <= 0:
        return 999.0 # stopped, no collision possible
    
    distance = crosswalk_x - vehicle_x
    if distance <= 0:
        return 0.0 #vehicle already at or past crosswalk
    
    return distance / vehicle_speed

class VehicleController:

    def __init__(self, start_x: float = 0.0):
        self.x = start_x #vehicle position
        self.speed = CRUISE_SPEED
        self.safety_margins = []

    def update(self, crossing_peds: list, dt: float,
               vehicle_pos: tuple = None) -> None:
        
        if vehicle_pos is None:
            vehicle_pos = (self.x, 0.0)

        min_ttc = 999.0
        for ped in crossing_peds:
            ttc = compute_ttc(self.x, self.speed, ped.x)
            ped.current_ttc = ttc #feedback into pedestrian model
            if ttc < min_ttc:
                min_ttc = ttc

        #Decide speed based on most urgent ttc
        self._apply_speed_decision(min_ttc, dt)

        #Advance vehicle position
        self.x += self.speed * dt

    def _apply_speed_decision(self, ttc: float, dt: float) -> None:

        if ttc > COMFORT_TTC:
            #Zone 1, all clear
            self.speed = min(self.speed + SOFT_BRAKE_DECEL * dt, CRUISE_SPEED)

        elif ttc > BRAKE_TTC:
            #Zone 2, Caution
            self.speed = max(self.speed - SOFT_BRAKE_DECEL * dt, 0.0)
            print(f"Vehicle soft braking | TTC={ttc:.1f}s | speed={self.speed:.1f}m/s")
        
        else: 
            #Zone 3, Danger
            self.speed = max(self.speed - HARD_BRAKE_DECEL * dt, 0.0)
            print(f"Vehicle hard braking | TTC={ttc:.1f}s | speed={self.speed:.1}m/s")


    def record_safety_margin(self, ped: Pedestrian, current_time: float) -> None:
        margin = compute_safety_margin(ped, ped.ttc_at_crossing_start, current_time)
        self.safety_margins.append(margin)
        print(f"Vehicle safety margin: {margin:.2f}s [mean target: 0.58s]")

    def get_mean_safety_margin(self) -> float:
        if not self.safety_margins:
            return 0.0
        return sum(self.safety_margins) / len(self.safety_margins)
    
# -----------------------------
# Quick Test
# -----------------------------

if __name__ == "__main__":
    from dataclasses import dataclass

    print("=== Vehicle Controller Test ===\n")

    controller = VehicleController(start_x=0.0)

    ped = Pedestrian(id=0, x=12.0, y=-4.0,
                     state="CROSSING", crossing=True,
                     walk_speed=1.34)
    ped.ttc_at_crossing_start = 12.0 / CRUISE_SPEED 

    sim_time = 0.0
    dt       = 0.1
    end_time = 10.0

    while sim_time < end_time:
        ped.y += ped.walk_speed * dt  # move pedestrian manually
        controller.update([ped], dt)
        sim_time += dt
        if ped.y >= 4.0:
            print(f"\nPedestrian exited at t={sim_time:.1f}s")
            controller.record_safety_margin(ped, sim_time)
            break

    print(f"\nMean safety margin: {controller.get_mean_safety_margin():.2f}s")
    print(f"Target (Nagulapati et al.): 0.58s")
    print(f"\n=== Done ===")