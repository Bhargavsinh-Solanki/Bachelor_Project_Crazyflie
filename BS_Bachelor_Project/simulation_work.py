import numpy as np
import matplotlib.pyplot as plt
import pandas as pandas



class PDController:
    def __init__(self):
        self.KP = 2.3
        self.KD = 0.557
        self.RX = 0.07
        self.RY = 0.07

        self._lastPos = [0, 0, 1]
        self.waypoint_time = 0.1

    def setLastPos(self, lastPos):
        self._lastPos = lastPos
    
    def getLastPos(self):
        return self._lastPos


    def xVelocity(self, currentPos, p1, p2):
        

        x_e1 = currentPos[0] - p1[0]
        x_e2 = currentPos[0] - p2[0]


        xedt1 = (x_e1 - self.getLastPos()[0]+p1[0])/self.waypoint_time
        xedt2 = (x_e2 - self.getLastPos()[0]+p2[0])/self.waypoint_time


        vx = -((self.KP*x_e1+self.KD*xedt1)*np.exp(-x_e1**2/(2*self.RX**2)) + (self.KP*x_e2+self.KD*xedt2)*np.exp(-x_e2**2/(2*self.RX**2)))

        return vx

    def droneUpdate(self, currentPos, p1, p2, Rn):
        vx = self.xVelocity(currentPos, p1, p2)
        xv = currentPos[0] + vx * self.waypoint_time + Rn
        self.setLastPos([xv,0,1])

        return xv

    
pd = PDController()

def simulation():
    time = np.arange(0, 100, 0.1)
    rng = np.random.default_rng()
    currentPos = (rng.uniform(-0.008, 0.008), 0, 1)
    X = []
    vxVals = []
    current_x = []
    timeVals = []
    P1 = []
    P2 = []


    for i in range(0, len(time)-1):
        # p1 = (np.random.default_rng().random(), 0 , 1)
        # p2 = (np.random.default_rng().random(), 0 , 1)
        p1 = (-0.1, 0 , 0)
        p2 = (0.1, 0 , 0)
        # Rn = np.random.default_rng().random()*0.001 # the default range is between 0 and 1
        Rn = 0
        v_x = pd.xVelocity(currentPos, p1, p2) 
        xv = pd.droneUpdate(currentPos, p1, p2,Rn)
        currentPos = (xv, 0 , 1)
        
        X.append(xv)
        vxVals.append(v_x)
        current_x.append(pd.getLastPos()[0])
        timeVals.append(time[i])
        P1.append(p1[0])
        P2.append(p2[0])



        
        
        if 0.90*p1[0] < currentPos[0] < p1[0]* 0.95 or abs(0.90*p2[0]) < abs(currentPos[0]) < abs(p2[0]* 0.95):
            print('The time is:', i)
            
            break
        
            

    test = 0.90*p1[0] < 0.0912 < p1[0]* 0.95
    print('What is it printing : ' , p1[0]* 0.95)

    print('The Velcity of the drone is :' , vxVals)
    print(test)
    print('The current position of the drone is :', current_x)
    print('Print the current position of the drone per timeVals: ' , X)
    plt.plot(X)
    plt.show()

simulation()