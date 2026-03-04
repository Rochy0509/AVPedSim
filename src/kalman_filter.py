import numpy as np

class KalmanFilter():

    def __init__(self, dt, process_noise, measurement_noise):

        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        #State vector Kalman filter
        self.x = np.zeros((4, 1)) #column vector shape (4,1) for matrix math

        #Transition matrix from Kalman filter
        self.F = np.array([
            [1, 0, self.dt, 0      ],
            [0, 1, 0,       self.dt],
            [0, 0, 1,       0      ],
            [0, 0, 0,       1      ]
        ])

        #Observation matrix, sensor only measures x and y
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        #Process noise, uncertainty in motion model
        self.Q = np.eye(4) * (self.process_noise**2)

        #Measurement Noise, uncertainty in sensor readings
        self.R = np.eye(2) * (self.measurement_noise**2)

        #Error covariance, starts large because initial state is unknown
        self.P = np.eye(4) * 1000

    def predict(self):
        
        #  move state forward one timestep using motion model
        self.x = self.F @ self.x

        # grow uncertainty
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, x_measured, y_measured):

        z = np.array([[x_measured], [y_measured]])

        #compute innovation
        y = z - self.H @ self.x

        #compute innovation covariance 
        S = self.H @ self.P @ self.H.T + self.R

        #compute Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S) 

        #Correct state estimate 
        self.x = self.x + K @ y

        #reduce uncertainty after new information
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

    # simple getter method
    def get_estimate(self, pedestrian_id):
        return {
            "pedestrian_id": pedestrian_id,
            "x_est":  self.x[0][0],
            "y_est":  self.x[1][0],
            "vx_est": self.x[2][0],
            "vy_est": self.x[3][0]
        }
    
    def initialize(self, x0, y0):
        # set initial position from first detection, velocity unknown so starts at 0
        self.x = np.array([[x0], [y0], [0.0], [0.0]])
        # reset uncertainty to large value for fresh start
        self.P = np.eye(4) * 1000


