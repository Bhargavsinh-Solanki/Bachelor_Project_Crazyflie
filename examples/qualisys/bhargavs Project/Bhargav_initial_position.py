
import asyncio
import math
import numpy as np 
import time
from threading import Thread
import xml.etree.cElementTree as ET

import qtm
from scipy.spatial.transform import Rotation

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

# URI to the Crazyflie to connect to
uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

# The name of the rigid body in QTM that represents the Crazyflie
rigid_body_name = 'cf'

# True: send position and orientation; False: send position only
send_full_pose = False

# PD controller parameters
TS = 0.05
KP = 2.3
KD = 0.557

# Define positions of leaders
leader1_pos = [0.5, 0, 0.5]  # [x, y, z] in meters
leader2_pos = [-0.5, 0, 0.5]  # [x, y, z] in meters

class QtmWrapper(Thread):
    def __init__(self, body_name):
        Thread.__init__(self)

        self.body_name = body_name
        self.on_pose = None
        self.connection = None
        self.qtm_6DoF_labels = []
        self._stay_open = True

        self.start()

    def close(self):
        self._stay_open = False
        self.join()

    def run(self):
        asyncio.run(self._life_cycle())

    async def _life_cycle(self):
        await self._connect()
        while (self._stay_open):
            await asyncio.sleep(1)
        await self._close()

    async def _connect(self):
        qtm_instance = await self._discover()
        host = qtm_instance.host
        print('Connecting to QTM on ' + host)
        self.connection = await qtm.connect(host)

        params = await self.connection.get_parameters(parameters=['6d'])
        xml = ET.fromstring(params)
        self.qtm_6DoF_labels = [label.text.strip() for index, label in enumerate(xml.findall('*/Body/Name'))]

        await self.connection.stream_frames(
            components=['6D'],
            on_packet=self._on_packet)

    async def _discover(self):
        async for qtm_instance in qtm.Discover('0.0.0.0'):
            return qtm_instance

    def _on_packet(self, packet):
        header, bodies = packet.get_6d()

        if bodies is None:
            return

        if self.body_name not in self.qtm_6DoF_labels:
            print('Body ' + self.body_name + ' not found.')
        else:
            index = self.qtm_6DoF_labels.index(self.body_name)
            temp_cf_pos = bodies[index]
            x = temp_cf_pos[0][0] / 1000
            y = temp_cf_pos[0][1] / 1000
            z = temp_cf_pos[0][2] / 1000

            r = temp_cf_pos[1].matrix
            rot = [
                [r[0], r[3], r[6]],
                [r[1], r[4], r[7]],
                [r[2], r[5], r[8]],
            ]

            if self.on_pose:
                # Make sure we got a position
                if math.isnan(x):
                    return

                self.on_pose([x, y, z, rot])

    async def _close(self):
        await self.connection.stream_frames_stop()
        self.connection.disconnect()

# Define the PD controller
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

def send_extpose_rot_matrix(cf, x, y, z, rot):
    """
    Send the current Crazyflie X, Y, Z position and attitude as a (3x3)
    rotaton matrix. This is going to be forwarded to the Crazyflie's
    position estimator.
    """
    quat = Rotation.from_matrix(rot).as_quat()

    if send_full_pose:
        cf.extpos.send_extpose(x, y, z, quat[0], quat[1], quat[2], quat[3])
    else:
        cf.extpos.send_extpos(x, y, z)

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

# QtmWrapper class and functions remain the same

if __name__ == '__main__':
    cflib.crtp.init_drivers()

    # Connect to QTM
    qtm_wrapper = QtmWrapper(rigid_body_name)

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf

        # Set up a callback to handle data from QTM
        qtm_wrapper.on_pose = lambda pose: send_extpose_rot_matrix(
            cf, pose[0], pose[1], pose[2], pose[3])

        # Get current state
        current_state = get_current_state(cf)

        # Choose leader
        leader_pos = choose_leader(current_state)

        # Calculate control output
        control_output = pd_controller.control(current_state, leader_pos)

        # Apply control output
        cf.commander.send_hover_setpoint(control_output[0], control_output[1], 0, current_state[2])

    qtm_wrapper.close()
