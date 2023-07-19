import asyncio
import math
import pandas as pd
import numpy as np 
import time
import xml.etree.cElementTree as ET
from threading import Thread

import qtm
from scipy.spatial.transform import Rotation

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem import Poly4D
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

# ... (rest of the imports and setup)

# Define positions of leaders
leader1_pos = [0.5, 0, 0.5]  # [x, y, z] in meters
leader2_pos = [-0.5, 0, 0.5]  # [x, y, z] in meters

# ... (rest of the setup)

# class QtmWrapper(Thread):
#     def __init__(self, body_name):
#         # ... (rest of the code)

#     # ... (rest of the code)

# # ... (rest of the code)

class PDController:
    def __init__(self, kp, kd, ts):
        self.kp = kp
        self.kd = kd
        self.ts = ts
        self.previous_error = [0.0, 0.0]

    def control(self, current_state, desired_state):
        error = [desired_state[i] - current_state[i] for i in range(2)]
        derivative = [(error[i] - self.previous_error[i]) / self.ts for i in range(2)]
        output = [self.kp * error[i] + self.kd * derivative[i] for i in range(2)]
        self.previous_error = error
        return output

# Initialize the PD controller
pd_controller = PDController(KP, KD, TS)

# Function to get current state
def get_current_state(cf):
    position = cf.position()
    return [position.x, position.y]

# Function to choose leader
def choose_leader(current_state):
    dist1 = math.sqrt((leader1_pos[0] - current_state[0]) ** 2 + (leader1_pos[1] - current_state[1]) ** 2)
    dist2 = math.sqrt((leader2_pos[0] - current_state[0]) ** 2 + (leader2_pos[1] - current_state[1]) ** 2)
    if dist1 < dist2:
        return leader1_pos
    else:
        return leader2_pos

# ... (rest of the code)

if __name__ == '__main__':
    # ... (rest of the code)

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        trajectory_id = 1

        # Set up a callback to handle data from QTM
        qtm_wrapper.on_pose = lambda pose: send_extpose_rot_matrix(
            cf, pose[0], pose[1], pose[2], pose[3])

        adjust_orientation_sensitivity(cf)
        activate_kalman_estimator(cf)

        # Get current state
        current_state = get_current_state(cf)

        # Choose leader
        leader_pos = choose_leader(current_state)

        # Calculate control output
        control_output = pd_controller.control(current_state, leader_pos)

        # Apply control output
        cf.commander.send_hover_setpoint(control_output[0], control_output[1], 0, current_state[2])

        # ... (rest of the code)

    qtm_wrapper.close()
