import numpy as np

class Camera:

    def __init__(self, fov_deg, max_range, noise_std, heading_deg):

        self.max_range = max_range
        self.noise_std = noise_std

        # convert FOV and heading from degrees to radians for angle math
        self.fov_rad = np.deg2rad(fov_deg)
        self.heading_rad = np.deg2rad(heading_deg)

    def detect(self, car_pos, pedestrians):
        # stores all detections this tick
        detections = []

        for pedestrian in pedestrians:
            # skip pedestrians that have left the scene
            if not pedestrian.active:
                continue
            
            # vector from car to pedestrian
            dx = pedestrian.x - car_pos[0]
            dy = pedestrian.y - car_pos[1]

            # straight-line distance between car and pedestrian
            distance = np.sqrt(dx**2 + dy**2)

            # absolute angle to pedestrian in world frame
            angle = np.arctan2(dy, dx)

            # angle to pedestrian relative to camera's facing direction
            relative_angle = angle - self.heading_rad

            # normalize to [-pi, pi] to handle angle wrap-around
            normalized_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))

            # pedestrian is detected only if inside FOV cone and within range
            if abs(normalized_angle) <= self.fov_rad / 2 and distance <= self.max_range:
                # add gaussian noise to position
                x_measured = pedestrian.x + np.random.normal(0, self.noise_std)
                y_measured = pedestrian.y + np.random.normal(0, self.noise_std)

                # store noisy detection result
                detections.append({
                    "pedestrian_id": pedestrian.id,
                    "x_measured": x_measured,
                    "y_measured": y_measured,
                    "detected": True
                })
        
        return detections


        
