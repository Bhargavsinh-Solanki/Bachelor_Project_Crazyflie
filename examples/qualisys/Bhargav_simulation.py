import math as m
import numpy as np
import matplotlib.pyplot as plt
# def desiredTrajectory(current_pose, duration, ticks):
#     #TODO



class PDController:
    def __init__(self):
        self.KP = 2.3
        self.KD = 0.557
        self.RX = 0.14
        self.RY = 0.14

        self.lastPos = [0, 0, 1]
        self.waypoint_time = 0.1



    # def calculateOptimal(self, currentPos, p1, p2):
    #     dPc2 = m.sqrt(pow(p2[0] - currentPos[0], 2) + pow(p2[1] - currentPos[1], 2) + pow(p2[2] - currentPos[2], 2))
    #     dpc1 = m.sqrt(pow(p1[0] - currentPos[0], 2) + pow(p1[1] - currentPos[1], 2) + pow(p1[2] - currentPos[2], 2))
    #     return dpc1, dPc2


    def xVelocity(self, currentPos, p1, p2):
        # twoDistances = self.calculateOptimal(currentPos, p1, p2)

        x_e1 = currentPos[0] - p1[0]
        x_e2 = currentPos[0] - p2[0]

        # if x_e1 or x_e2 > self.RX:
        #     return # land

        xedt1 = (x_e1 - self.lastPos[0])/self.waypoint_time
        xedt2 = (x_e2 - self.lastPos[0]) / self.waypoint_time

        vx = -((self.KP*x_e1+self.KD*xedt1)*np.exp(-x_e1**2/(2*self.RX**2)) + (self.KP*x_e2+self.KD*xedt2)*np.exp(-x_e2**2/(2*self.RX**2)))

        return vx

    def droneUpdate(self, currentPos, p1, p2, t2, t1, Rn):
        vx = self.xVelocity(currentPos, p1, p2)
        xv = currentPos[0] + vx * (t2 - t1) + Rn

        self.lastPos[0] = xv

        return xv



pd = PDController()


def simulation():
    time = np.arange(0, 100, 0.1)
    currentPos = (0.01, 0, 1)
    X = []
    for i in range(0, 1000-1):
        # p1 = (np.random.default_rng().random(), 0 , 1)
        # p2 = (np.random.default_rng().random(), 0 , 1)
        p1 = (-0.1, 0 , 1)
        p2 = (0.1, 0 , 1)
        Rn = np.random.default_rng(). random()*0.001
        xv = pd.droneUpdate(currentPos, p1, p2 , time[i+1], time[i],Rn)
        currentPos = (xv, 0 , 1)
        X.append(xv)

    # print(X)
    plt.plot(X)
    plt.show()

simulation()
