
"""simloop.py: Simulation loop management """

__author__ = "Pablo Alvarado"
__copyright__ = "Copyright 2020, Pablo Alvarado"
__license__ = "BSD 3-Clause License (Revised)"

import time

from enum import Enum
from PyQt5 import QtCore


class SimState(Enum):
    """Constants for each simulation state"""
    Stopped = 0
    Stopping = 1
    Idle = 2
    Running = 3
    Stepping = 4
    Restarting = 5
    StStepping = 6
    Quitting = 7


class DummyStepper:
    """Dummy stepper class to be used if nothing else is specified"""
    def init(self):
        print("DummyStepper.init() called")

    def reset(self):
        print("DummyStepper.reset() called")

    def step(self, iteration):
        print("DummyStepper.step(", iteration, ")")
        return


class SimTimer(QtCore.QRunnable):
    """Timer handled by an independent thread"""

    def defaultFunction():
        print("SimTimer default function called.")
        return

    def __init__(self, fct=defaultFunction):
        super(SimTimer, self).__init__()

        self.state = SimState.Stopped
        self.interval = -1
        self.wait = QtCore.QSemaphore(0)
        self.timeoutFunction = fct

    def setTimeoutFunction(self, fct):
        self.timeoutFunction = fct

    @QtCore.pyqtSlot()
    def run(self):
        while True:

            # print("[DBG]   SimTimer.state=", self.state)

            if (self.state == SimState.Stopped):
                self.wait.acquire()
            elif (self.state == SimState.Running):

                t0 = time.perf_counter()
                self.timeoutFunction()
                deltat = time.perf_counter() - t0
                # print("[DBG] deltat=",deltat)

                left = self.interval-deltat

                # print("[DBG] left=",left)

                if (left > 0):
                    # print("[DBG]   SimTimer sleep for ",left,"seconds")
                    time.sleep(left)
                elif (self.interval < 0):
                    self.state = SimState.Stopped

            elif (self.state == SimState.Quitting):
                break

    def start(self, interval=-1):

        # print("[DBG] SimTimer.start with interval", interval, "s")
        self.interval = interval
        self.state = SimState.Running
        self.wait.release(1)

    def stop(self):
        self.state = SimState.Stopped

    def isActive(self):
        return self.state == SimState.Running

    def quit(self):
        self.state = SimState.Quitting
        self.wait.release(1)

    def setInterval(self, interval):
        # print("[DBG] SimTimer.setInterval ", interval)
        self.interval = interval


class SimLoop(QtCore.QRunnable):
    """Simulation loop running in a secondary Qt thread"""

    def __init__(self, stepper=DummyStepper(), stepInt_s=0.2):
        """The SimLoop requires a stepper object, that will be called
           at each single step.  This object must have a method step(iter)
           and a method init() which should reset the state of the stepper
           after the simulator has been stopped and restarted.
           This object needs also the step interval in s.
        """

        super(SimLoop, self).__init__()

        self.speed = 1
        self.stepper = stepper
        self.state = SimState.Stopped
        self.wait = QtCore.QSemaphore(0)
        self.lock = QtCore.QMutex()
        self.iteration = 0
        self.stepInterval = stepInt_s

        self.timer = SimTimer(self.playStep)
        self.stepEnabled = True
        
    @QtCore.pyqtSlot()
    def run(self):

        while True:

            # print("[DBG] SimLoop.state=", self.state)

            if self.state == SimState.Stopping:
                self.timer.stop()

                # Cannot use lockers because in Python, scope here is that of
                # the function
                self.lock.lock()
                self.state = SimState.Stopped
                self.lock.unlock()

            elif self.state == SimState.Stopped:
                self.wait.acquire()  # Blocks until some other button pressed

            elif self.state == SimState.Idle:
                self.timer.stop()
                self.wait.acquire()

            elif self.state == SimState.Running:

                if (self.timer.isActive()):
                    self.timer.stop()

                self.timer.start(self.stepInterval)
                self.wait.acquire()  # Blocks until something else happens

            elif self.state == SimState.Stepping:
                self.playStep()
                self.wait.acquire()

            elif self.state == SimState.Restarting:
                self.stepper.init()
                self.timer.stop()
                self.lock.lock()
                self.iteration = 0
                self.stepEnabled = True
                self.state = SimState.Running
                self.lock.unlock()

            elif self.state == SimState.StStepping:
                self.stepper.init()
                self.timer.stop()
                self.lock.lock()
                self.iteration = 0
                self.stepEnabled = True
                self.state = SimState.Stepping
                self.lock.unlock()
                self.wait.acquire()

            elif self.state == SimState.Quitting:
                self.timer.quit()
                self.lock.lock()
                self.state = SimState.Stopped
                self.lock.unlock()
                break

    def setStepper(self, stepper):
        self.stepper = stepper

    # The following methods will be executed within the sim thread

    def playStep(self):
        if self.stepEnabled:
            self.stepper.step(self.iteration)

        locker = QtCore.QMutexLocker(self.lock)  # noqa: F841
        self.iteration = self.iteration+1

    # The following methods will be executed in the gui thread

    def play(self):
        # print("[DBG] SimLoop.play")
        
        locker = QtCore.QMutexLocker(self.lock)  # noqa: F841
        if (self.state != SimState.Running):
            # print("[DBG] In play state is ", self.state)

            if (self.state == SimState.Stopped):
                self.state = SimState.Restarting
            else:
                self.state = SimState.Running

            self.stepEnabled = True
                
            self.wait.release(1)

    def pause(self):
        # print("[DBG] SimLoop.pause")
        locker = QtCore.QMutexLocker(self.lock)  # noqa: F841
        if (self.state != SimState.Idle):
            self.state = SimState.Idle
            self.wait.release(1)

    def step(self):
        # print("[DBG] SimLoop.step")
        locker = QtCore.QMutexLocker(self.lock)  # noqa: F841

        self.stepEnabled = True
        
        if (self.state == SimState.Running):
            return

        if (self.state == SimState.Idle or self.state == SimState.Stepping):
            self.state = SimState.Stepping
        elif (self.state == SimState.Stopped):
            self.state = SimState.StStepping

        self.wait.release(1)

    def stop(self):
        # print("[DBG] SimLoop.stop")
        locker = QtCore.QMutexLocker(self.lock)  # noqa: F841
        if (self.state != SimState.Stopped):
            self.state = SimState.Stopping
            self.wait.release(1)

    def restart(self):
        # print("[DBG] SimLoop.restart")
        locker = QtCore.QMutexLocker(self.lock)  # noqa: F841
        if (self.state != SimState.Restarting):
            self.state = SimState.Restarting
            self.wait.release(1)

    def quit(self):
        # print("[DBG] SimLoop.quit")

        locker = QtCore.QMutexLocker(self.lock)  # noqa: F841
        self.state = SimState.Quitting
        self.wait.release(1)
        
    def setInterval(self, interval):
        self.stepInterval = interval
        self.timer.setInterval(self.stepInterval)

    # These methods are called from anywhere
    def disableExecution(self):
        locker = QtCore.QMutexLocker(self.lock)  # noqa: F841
        self.stepEnabled = False
