import asyncio
import math
import xml.etree.ElementTree as ET
import numpy as np
import time
from threading import Thread
from itertools import count
import random
import matplotlib.pyplot as plt
import pandas as pd
import qtm

from distutils.ccompiler import gen_preprocess_options

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

# URI to the Crazyflie to connect to
uri = uri_helper.uri_from_env(default='radio://0/100/2M/E7E7E7E701')

# The name of the rigid body in QTM that represents the Crazyflie
rigid_body_name = 'cf1'

# True: send position and orientation; False: send position only
send_full_pose = False

class QtmWrapper(Thread):
    def __init__(self, body_name):
        Thread.__init__(self)

        self.body_name = body_name
        self.on_pose = None
        self.connection = None
        self.qtm_6DoF_labels = []
        self._stay_open = True
        self.pose = []

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
        host = '192.168.1.102'
        print('Connecting to QTM on ' + host)
        self.connection = await qtm.connect(host)

        params = await self.connection.get_parameters(parameters=["6d"])
        xml = ET.fromstring(params)
        self.qtm_6DoF_labels = [label.text.strip() for index, label in enumerate(xml.findall('*/BodyName'))]

        await self.connection.stream_frames(components=['6D', '6dEuler'], on_packet=self._on_packet)

        async def _discover():
            async for qtm_instance in qtm.discover('0.0.0.0'):
                return qtm_instance
            
        def _on_packet(packet):
            header, bodies = packet.get_6d()
            header_eulor, bodies_eulor = packet.get_6d_euler()

            if bodies is None:
                return
            
            if self.body_name not in self.qtm_6DoF_labels:
                print('Body' + self.body_name + 'not found in QTM')
            else:
                index = self.qtm_6DoF_labels.index(self.body_name)
                temp_cf_pose = bodies[index]
                self.pose = bodies_eulor[index]
                x = temp_cf_pose[0][0] / 1000
                y = temp_cf_pose[0][1] / 1000
                z = temp_cf_pose[0][2] / 1000

                r = temp_cf_pose[1].matrix
                rot = [r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], r[2][0], r[2][1], r[2][2]]
            
                if self.on_pose:
                    #Make sure we get a position
                    if math.isnan(x):
                        return
                    self.on_pose([x, y, z, rot]) 

    async def _close(self):
        await self.connection.stream_frames_stop()
        self.connection.disconnect()


class PDController:
    def __init__(self):
        self.KP = 2.3
        self.KD = 0.557
        
        # Position controller Parameters
        self.Kp_x = 1.2
        self.Kp_xy = 1.2
        self.RX = 0.07

        self.last_pos = [0.0, 0.0, 0.0]
        self.waypoint_time = 0.1

    def get_vx_from_PD(self, CurrentPos, P1, P2, y_L=0.0):
        x_e1 = CurrentPos[0] - P1[0]
        x_e2 = CurrentPos[0] - P2[0]

        xedt1 = (x_e1 - (self.last_pos[0] + P1[0])) / self.waypoint_time
        xedt2 = (x_e2 - (self.last_pos[0] + P2[0])) / self.waypoint_time

        y_e = CurrentPos[1] - y_L

        vx = -((self.KP*x_e1+self.KD*xedt1)*np.exp(-x_e1**2/(2*self.RX**2)) + (self.KP*x_e2+self.KD*xedt2)*np.exp(-x_e2**2/(2*self.RX**2)))

        return vx




def wait_for_position_estimator(scf):
    print('Waiting for estimator to find position...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            print("{} {} {}".
                  format(max_x - min_x,
                         max_y - min_y,
                         max_z - min_z))

            if ((max_x - min_x) < threshold and
                (max_y - min_y) < threshold and
                (max_z - min_z) < threshold):
                break


def _sqrt(a):
    if a < 0.0:
        return 0.0
    return math.sqrt(a)

def send_extpose_rot_matrix(cf, x, y, z, rot):

    qw = _sqrt(1 + rot[0][0] + rot[1][1] + rot[2][2]) / 2
    qx = _sqrt(1 + rot[0][0] - rot[1][1] - rot[2][2]) / 2
    qy = _sqrt(1 - rot[0][0] + rot[1][1] - rot[2][2]) / 2
    qz = _sqrt(1 - rot[0][0] - rot[1][1] + rot[2][2]) / 2

    #Normalize the quarternion
    ql = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)

    if send_full_pose:
        cf.extpos.send_extpose(x, y, z, qw / ql, qx / ql, qy / ql, qz / ql)
    else:
        cf.extpos.send_extpose(x, y, z)

def go_to_height(cf, position, desired_z, pd):
    # Go to the desired height
    actual_z = position[2]
    K_p = pd.Kp_x
    K_p_xy = pd.Kp_xy
    vz = K_p * (desired_z - actual_z)

    K_p_xy = 1.0

    #print("Actul height : {} Desired height : {} vz : {}".format(actual_z, desired_z, vz))

    cf.commander.send_velocity_world_setpoint(0.0 - K_p_xy * position[0], 0.0 - K_p_xy * position[1], vz, 0.01 * position[3])


def reset_estimator(cf):
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')

    wait_for_position_estimator(cf)

def activate_kalman_estimator(cf):
    cf.param.set_value('stabilizer.estimator', '2')
    cf.param.set_value('locSrv.extQuatStdDev', '0.06')


index  = count()
yVals = []
xVals = []

def animate(i):
    xVals.append(next(index))
    yVals.append(random.randint(0, 10))

    plt.cla()
    plt.plot(xVals, yVals)




if __name__ == '__main___':
    cflib.crtp.init_drivers()

    # Connect to QTM
    qtm_wrapper = QtmWrapper(rigid_body_name)

    with SyncCrazyflie(uri, cf = Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        trajectory_id = 1 
        DEFAULT_HEIGHT = 0.3

    # Set up a callback to handle data from QTM

    qtm_wrapper.on_pose = lambda pose: send_extpose_rot_matrix(cf, pose[0], pose[1], pose[2], pose[3])

    activate_kalman_estimator(cf)
    reset_estimator(cf)

    current_x = qtm_wrapper.pose[0][0]/1000.
    current_y = qtm_wrapper.pose[0][1]/1000.
    current_z = qtm_wrapper.pose[0][2]/1000.
    current_yaw = qtm_wrapper.pose[1][0]

    start_time = time.time()

    pd = PDController()

    state = "null"
    XvVals = []

    while (True):
        # Set up a callback to handle data from QTM

        current_x = qtm_wrapper.pose[0][0]/1000.
        current_y = qtm_wrapper.pose[0][1]/1000.
        current_z = qtm_wrapper.pose[0][2]/1000.
        current_yaw = qtm_wrapper.pose[1][0]

        current_time = time.time() - start_time

        desired_z = 1.0


        if current_time < 3:    # try to take off and hover for the first 3 seconds
                state = "takeoff"
                go_to_height(cf, [current_x, current_y, current_z, current_yaw], desired_z=desired_z, pd=pd)

        elif current_time < 30:# elif
                state = "flight"
                #print("Flight")
                #
                # Calculate v_x from Liang's method
                factor = 10
                p1 = [0.1, 0.0, desired_z]
                p2 = [-0.1, 0.0, desired_z]
                v_x = pd.get_vx_from_BS([current_x, current_y, current_z, current_yaw], p1, p2)
                
                
                # just make the y- and z-velocity zero (also Yaw) with a P-controller
                #
                v_y = 0.0-pd.k_P_xy*current_y
                v_z = pd.k_P_z*(desired_z - current_z)
                v_yaw = -0.01*current_yaw
                cf.commander.send_velocity_world_setpoint(v_x, 0*v_y, 0*v_z, 0*v_yaw)

                pd.last_pos = [current_x, current_y, current_z, current_yaw]

                target_reached = 0.90*p1[0] < current_x < p1[0]* 0.95 or abs(0.90*p2[0]) < abs(current_x) < abs(p2[0]* 0.95)
            
                if not target_reached:
        
                    XvVals.append(current_x ) # HEEEEEEEEEEEEEEEERE FAAAAAAAAACTOOOOOOOOOOOOOR
                if target_reached:
                    print("Target reached")
                    
                    for i in range(100):
                        go_to_height(cf, [current_x, current_y, current_z, current_yaw], desired_z=desired_z, pd=pd)
                        time.sleep(0.01)
                    
                    # break
                     # Land the drone
                cf.commander.send_stop_setpoint()  # Stop any movement
                cf.commander.send_setpoint(0, 0, 0, 0)  # Send zero setpoints to stop movement
                cf.commander.send_setpoint(0, 0, 0, 0, relative=False, yaw_rate=0.0, is_land=True)  # Land the drone


        qtm_wrapper.close()

        plt.plot(XvVals)
        plt.tight_layout()
        plt.show()

