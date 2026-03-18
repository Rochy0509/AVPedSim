import numpy as np

class Lidar():

    def __init__(self, num_beams=36, max_range=20.0, base_noise=0.03):
        
        self.num_beams = num_beams
        self.max_range = max_range
        self.base_noise = base_noise # scales with distance (Espineira et al., 2021)

        self.beam_angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)
        self.angle_tolerance = (2 * np.pi / num_beams) / 2

    def detect(self, car_pos, pedestrians):
        
        detections = []

        for pedestrian in pedestrians:

            if not pedestrian.active:
                continue 
            
            #calculating distance using pythagorean theorem
            dx = pedestrian.x - car_pos[0]
            dy = pedestrian.y - car_pos[1]
            distance = np.sqrt(dx**2 + dy**2)

            #calculating angle from car to pedestrian
            angle = np.arctan2(dy, dx)

            #calculating how close the pedestrian is from the car
            angle_diffs = np.abs(self.beam_angles - angle)
            #wrapping angles around
            angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
            min_diff = np.min(angle_diffs)

            if min_diff <= self.angle_tolerance and distance <= self.max_range:
                noise_std = self.base_noise * distance
                noisy_distance = distance + np.random.normal(0, noise_std)
                x_measured = car_pos[0] + noisy_distance * np.cos(angle)
                y_measured = car_pos[1] + noisy_distance * np.sin(angle)

                detections.append({
                    "pedestrian_id": pedestrian.id,
                    "x_measured": x_measured,
                    "y_measured" : y_measured,
                    "detected" : True
                })
        
        return detections
    



