
import pyqtgraph as pg
import numpy as np
try:
    from modules.ThorlabsCamera.uc480 import uc480
except ImportError:
    from ThorlabsCamera.uc480 import uc480
import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
import sys
from PyQt5 import QtGui, uic
import PyQt5
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from pyqtgraph.Qt import QtGui, QtCore
import time as t
from scipy.optimize import curve_fit
from matplotlib import cm

CONVERSION_PX_TO_MIKROM = 5.2


class ThorlabsCam:

    def __init__(self, camsToUse):
        self.cam = uc480()

    def openCommunications(self):
        self.cam.connect()

    def setCameraParameters(self, exposureTime):
        self.cam.set_exposure(exposureTime)

    def startAcquisition(self):
        print('Start')

    def stopAcquisition(self):
        self.cam.disconnect()

    def getImage(self):
        imageArray = self.cam.acquire(1)
        return imageArray, 1

    def getIntLimits(self):
        '''
        for i in self.camList:
            cam = self.cameras[i]
            min = cam.ExposureTimeRaw.Min
            max = cam.ExposureTimeRaw.Max
        '''
        min, max, increment = self.cam.get_exposure_limits()
        return min, max

    def setIntegrationTime(self, exposureTime):
        self.cam.set_exposure(exposureTime)

    def getPixelSize(self, camera):
        return (CONVERSION_PX_TO_MIKROM, CONVERSION_PX_TO_MIKROM)


def main():
    cam = ThorlabsCam(0)
    cam.openCommunications()
    cam.setCameraParameters(60)
    img = cam.getImage()
    print(img)

if __name__ == '__main__':
        main()