import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sensors.lidar import Lidar
from sensors.camera import Camera
from kalman_filter import KalmanFilter

#Fusion Parameters
KF_DT = 0.1            #timestep
KF_PROCESS_NOISE = 0.5 #how much to trust the motion model
KF_MEASUREMENT_NOISE = 1.0 #how much to trust the sensor readings


class SensorFusion:
    
    def __init__(self):
        self.lidar = Lidar(
            num_beams=36,
            max_range=20.0,
            base_noise=0.02
        )

        self.cameras = [
            # Front main — narrow, long range
            Camera(fov_deg=47,  max_range=150.0, noise_std=0.3,  heading_deg=0),
            # Front wide — wide, medium range
            Camera(fov_deg=120, max_range=60.0,  noise_std=0.5,  heading_deg=0),
            # Front telephoto — very narrow, very long range
            Camera(fov_deg=22,  max_range=250.0, noise_std=0.2,  heading_deg=0),
            # B-pillar left — side forward
            Camera(fov_deg=90,  max_range=40.0,  noise_std=0.4,  heading_deg=90),
            # B-pillar right — side forward
            Camera(fov_deg=90,  max_range=40.0,  noise_std=0.4,  heading_deg=-90),
        ]

        #KalmanFilter per tracked pedestrian
        self.tracks = {}

    def update(self, car_pos: tuple, active_pedestrians: list) -> list:

        #Running sensors
        lidar_detections = self.lidar.detect(car_pos, active_pedestrians)

        camera_detections = []
        for cam in self.cameras:
            camera_detections.extend(cam.detect(car_pos, active_pedestrians))


        #Merge to build one detection per pedestrian_id
        merged = self._merge_detections(lidar_detections, camera_detections)

        #predict all tracks to update with detections
        detected_ids = set(merged.keys())

        for ped_id, kf in self.tracks.items():
            kf.predict()
            if ped_id in detected_ids:
                det = merged[ped_id]
                kf.update(det["x_measured"], det["y_measured"])

        #create new tracks for pedestrian seen for first time
        for ped_id, det in merged.items():
            if ped_id not in self.tracks:
                kf = KalmanFilter(
                    dt = KF_DT,
                    process_noise = KF_PROCESS_NOISE,
                    measurement_noise = KF_MEASUREMENT_NOISE
                )
                kf.initialize(det["x_measured"], det["y_measured"])
                self.tracks[ped_id] = kf

        self.tracks = {
            pid: kf for pid, kf in self.tracks.items() if kf.active
        }

        #return confirmed estimates
        estimates = []
        for ped_id, kf in self.tracks.items():
            if kf.is_valid():
                estimates.append(kf.get_estimate(ped_id))

        return estimates
    
    def _merge_detections(self,
                          lidar_dets: list,
                          camera_dets: list) -> dict:
        merged = {}
        
        #camera detections
        for det in camera_dets:
            pid = det["pedestrian_id"]
            merged[pid] = {
                "x_measured": det["x_measured"],
                "y_measured": det["y_measured"],
                "source": "camera"
            }

        #LiDAR
        for det in lidar_dets:
            pid = det["pedestrian_id"]
            merged[pid] = {
                "x_measured": det["x_measured"],
                "y_measured": det["y_measured"],
                "source": "camera"
            }
        
        return merged
    
