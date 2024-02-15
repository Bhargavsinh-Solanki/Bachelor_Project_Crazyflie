import asyncio
import math
import time
import xml.etree.cElementTree as ET
from threading import Thread

import numpy as np
import pandas as pd
import qtm
from distutils.ccompiler import gen_preprocess_options

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem import Poly4D
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

# URI to the Crazyflie to connect to
uri = uri_helper.uri_from_env(default='radio://0/100/2M/E7E7E7E701')

# The name of the rigid body in QTM that represents the Crazyflie
rigid_body_name = 'cf1'

# True: send position and orientation; False: send position only
send_full_pose = False 

p1 = [1,0,1]
p2 = [-1,0,1]

controller = PDController()

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
        host = qtm_instance.host
        print('Connecting to QTM on ' + host)
        self.connection = await qtm.connect(host)

        params = await self.connection.get_parameters(parameters=['6d'])
        xml = ET.fromstring(params)
        self.qtm_6DoF_labels = [label.text.strip() for index, label in enumerate(xml.findall('*/Body/Name'))]

        await self.connection.stream_frames(
            components=['6D', '6dEuler'],
            on_packet=self._on_packet)

    async def _discover(self):
        async for qtm_instance in qtm.Discover('0.0.0.0'):
            return qtm_instance

    def _on_packet(self, packet):

        header, bodies = packet.get_6d()
        header_eulor, bodies_eulor = packet.get_6d_euler()

        if bodies is None:
            return

        if self.body_name not in self.qtm_6DoF_labels:
            print('Body ' + self.body_name + ' not found.')
        else:
            index = self.qtm_6DoF_labels.index(self.body_name)
            temp_cf_pos = bodies[index]
            self.pose = bodies_eulor[index]
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

            # print("{} {} {}".
            #       format(max_x - min_x, max_y - min_y, max_z - min_z))

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break


def _sqrt(a):
    """
    There might be rounding errors making 'a' slightly negative.
    Make sure we don't throw an exception.
    """
    if a < 0.0:
        return 0.0
    return math.sqrt(a)


def send_extpose_rot_matrix(cf, x, y, z, rot):
    """
    Send the current Crazyflie X, Y, Z position and attitude as a (3x3)
    rotaton matrix. This is going to be forwarded to the Crazyflie's
    position estimator.
    """
    qw = _sqrt(1 + rot[0][0] + rot[1][1] + rot[2][2]) / 2
    qx = _sqrt(1 + rot[0][0] - rot[1][1] - rot[2][2]) / 2
    qy = _sqrt(1 - rot[0][0] + rot[1][1] - rot[2][2]) / 2
    qz = _sqrt(1 - rot[0][0] - rot[1][1] + rot[2][2]) / 2

    # Normalize the quaternion
    ql = math.sqrt(qx ** 2 + qy ** 2 + qz ** 2 + qw ** 2)

    if send_full_pose:
        cf.extpos.send_extpose(x, y, z, qx / ql, qy / ql, qz / ql, qw / ql)
    else:
        cf.extpos.send_extpos(x, y, z)


# Here the drone goes to the particular height and it stabiles with the help of the P Controller
def go_to_height(cf, position, des_z):
    act_z = position[2]
    K_p = 1.2
    vz = K_p*(des_z - act_z)

    k_P_xy = 1.0

    print("Actual height: ", act_z)
    cf.commander.send_velocity_world_setpoint(0.0-k_P_xy*position[0], 0.0-k_P_xy*position[1], vz, -0.01*position[3])


#Always give 

def reset_estimator(cf):
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')

    # time.sleep(1)
    wait_for_position_estimator(cf)


def activate_kalman_estimator(cf):
    cf.param.set_value('stabilizer.estimator', '2')

    # Set the std deviation for the quaternion data pushed into the
    # kalman filter. The default value seems to be a bit too low.
    cf.param.set_value('locSrv.extQuatStdDev', 0.06)


if __name__ == '__main__':
    cflib.crtp.init_drivers()

    # Connect to QTM
    qtm_wrapper = QtmWrapper(rigid_body_name)

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        trajectory_id = 1

        DEFAULT_HEIGHT = 0.3
        TS = 0.1
        KP = 2.3
        KD = 0.557
        RX = 0.14
        RY = 0.14

        # Set up a callback to handle data from QTM
        qtm_wrapper.on_pose = lambda pose: send_extpose_rot_matrix(
            cf, pose[0], pose[1], pose[2], pose[3])

        activate_kalman_estimator(cf)
        reset_estimator(cf)

        current_x = qtm_wrapper.pose[0][0]/1000.
        current_y = qtm_wrapper.pose[0][1]/1000.
        current_z = qtm_wrapper.pose[0][2]/1000.
        current_yaw = qtm_wrapper.pose[1][0]

        start_time = time.time()
        prev_time = start_time

        # for i in range(10000):
        while(True):
            # Set up a callback to handle data from QTM

            current_x = qtm_wrapper.pose[0][0]/1000.
            current_y = qtm_wrapper.pose[0][1]/1000.
            current_z = qtm_wrapper.pose[0][2]/1000.
            current_yaw = qtm_wrapper.pose[1][0]

            cur_time = time.time() - start_time

            # print(cur_time)

            if cur_time < 5:    # try to take off and hover for the first 5 seconds
                go_to_height(cf, [current_x, current_y, current_z, current_yaw], des_z=1.0)

            else:  # land smoothly to 0.2, and then shut down the motors
                if current_z>0.2:
                    go_to_height(cf, [current_x, current_y, current_z, current_yaw], des_z=-0.05)
                else:
                    cf.commander.send_velocity_world_setpoint(0, 0, 0, 0) # set all the motors off
                    break
            # print(qtm_wrapper.pose[0][2]/1000)

            # time.sleep(0.1)

        # qtm_wrapper.close()




        #take_off(cf, [current_x, current_y, current_z+0.5, current_yaw])
        # current_pose = np.array([current_x, current_y, current_yaw])

        # Desired = prepare_desired_trajactory(current_pose=current_pose, SimulationTime=50, track='sine')

        # state_sim = np.array([current_pose])

        # xe_last = 0
        # ye_last = 0
        # # states_debug = np.empty((1,15))
        # for ii, desired in enumerate(Desired):
        #     start_time = time.time()

        #     current_x = qtm_wrapper.pose[0][0]/1000.
        #     current_y = qtm_wrapper.pose[0][1]/1000.
        #     current_z = qtm_wrapper.pose[0][2]/1000.
        #     current_yaw = qtm_wrapper.pose[1][0]
        #     current_pose = np.array([[current_x, current_y, current_yaw]])

        #     state_sim = np.append(state_sim, current_pose, 0)
        #     xe_in = state_sim[-1, 0]-Desired[ii, 0]
        #     ye_in = state_sim[-1, 1]-Desired[ii, 1]
        #     ang_in = Desired[ii, 2]

        #     ang_temp = -ang_in
        #     xe_o = xe_in*np.cos(ang_temp) - ye_in*np.sin(ang_temp)
        #     ye_o = xe_in*np.sin(ang_temp) + ye_in*np.cos(ang_temp)

        #     xedt = (xe_o-xe_last)/TS
        #     yedt = (ye_o-ye_last)/TS

        #     vx = -(KP*xe_o+KD*xedt)*np.exp(-xe_o**2/(2*RX**2))
        #     vy = -(KP*ye_o+KD*yedt)*np.exp(-ye_o**2/(2*RY**2))

        #     print(xe_o, vx)
        #     print(current_x)

        #     vx_o = np.sign(vx)*np.min([np.abs(vx), 1])
        #     vy_o = np.sign(vy)*np.min([np.abs(vy), 1])
        #     vyaw = 0

            # vyaw = np.sign(vyaw)*np.min([np.abs(vyaw),30])


            # vx_o = np.sign(vx_o)*np.min([np.abs(vx_o),0.5])
            # vy_o = np.sign(vy_o)*np.min([np.abs(vy_o),0.5])
            # vyaw = np.sign(vyaw)*np.min([np.abs(vyaw),30])
            # print(vx_o, vy_o, vyaw)
            #cf.commander.send_hover_setpoint(vx_o, vy_o, vyaw, DEFAULT_HEIGHT)

            # xe_last = xe_o
            # ye_last = ye_o

            # states_temp = [[desired[0], desired[1], current_x, current_y, xe_in,
            #                 ye_in, ang_in, xe_o, ye_o, xedt, yedt, vx, vy, vx_o, vy_o]]
            # if ii == 0:
            #     states_debug = states_temp
            # else:
            #     states_debug = np.append(states_debug, states_temp, 0)
            # if TS-(time.time() - start_time) > 0:
            #     time.sleep(TS-(time.time() - start_time))





        # take_off(cf, [current_x, current_y, current_z, current_yaw])
        #

        
        
        #land(cf, [current_x, current_y, current_z])

    # indexes = ['desired_x', 'desired_y', 'current_x', 'current_y', 'xe_in', 'ye_in',
    #            'ang_in', 'xe_o', 'ye_o', 'xedt', 'yedt', 'vx', 'vy', 'vx_o', 'vy_o']
    # df = pd.DataFrame(data=states_debug, columns=indexes)
    # df.to_csv('./debug.csv')
    qtm_wrapper.close()
