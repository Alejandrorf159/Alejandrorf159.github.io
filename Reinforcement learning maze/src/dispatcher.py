
"""dispatcher.py:

   Distribute the information among all modules:
   - gui takes care of the window managment
   - environment holds the maze and agent
   - simloop controls the threads for running the simulation
"""

__author__ = "Pablo Alvarado"
__copyright__ = "Copyright 2020, Pablo Alvarado"
__license__ = "BSD 3-Clause License (Revised)"

import sys
from PyQt5 import QtWidgets
import gui


class Dispatcher:
    """Main controller for the whole program.

       This class holds the main actors in the application and
       ensures that the GUI and the environment keep in touch.
    """

    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = gui.Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)
        self.app.aboutToQuit.connect(self.ui.quit)

    def setEnvironment(self, env):
        self.ui.setEnvironment(env)

    def env(self):
        return self.ui.env

    def setStepper(self, stepper):
        self.ui.setStepper(stepper)

    def stepper(self):
        return self.ui.stepper()

    def render(self):
        self.ui.renderSignal.emit()

    def pause(self):
        # print("[DBG] Pause invoqued")

        # When the signal is emitted, the simloop
        # keeps executing for a while.  We must
        # flag an immediate stop while the threads
        # really stop
        self.ui.disableExecution()  # Stop now!
        self.ui.pauseSignal.emit()

    def stop(self):
        # print("[DBG] Stop invoqued")
        self.ui.disableExecution()  # Stop now!
        self.ui.stopSignal.emit()

    def restart(self):
        # print("[DBG] Restart invoqued")
        self.ui.disableExecution()  # Stop now!
        self.ui.restartSignal.emit()

    def run(self):
        self.ui.startThreads()
        self.MainWindow.show()
        self.ui.fitInCanvasView()
        sys.exit(self.app.exec_())
