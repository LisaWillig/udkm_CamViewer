import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
import sys
import os
from datetime import datetime
from PyQt5 import uic
import PyQt5
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from pyqtgraph.Qt import QtGui, QtCore
from matplotlib import cm
from scipy.optimize import curve_fit
from pypylon import genicam
import config
import logging, traceback

if config.CamCompany == 'Basler':
    from modules.BaslerCommunication import BaslerCam as Camera
elif config.CamCompany == 'Thorlabs':
    from modules.ThorlabsCommunication import ThorlabsCam as Camera
from modules.saveGUISettings import GUISettings
from modules.analysis import *

imageCalculations = dict()

pixelFormat = 'Mono8'

colormaps = {0: 'viridis',
             1: 'plasma',
             2: 'inferno',
             3: 'Greys',
             4: 'seismic',
             5: 'copper',
             6: 'gist_gray',
             }

class MyWindow(PyQt5.QtWidgets.QMainWindow):

    def __init__(self, parent=None):

        super(MyWindow, self).__init__(parent)
        self.ui = uic.loadUi('\\GUI\\udkm_CamViewer.ui', self)

        self.ui.setWindowTitle("udkm_CamViewer")
        self.init_ui()

        self.cam = Camera()
        self.bcamera = None
        #
        self.average = 1
        self.bZoom = False
        self.init = 1
        self.bFitX = False
        self.bFitY = False
        self.IntTime = None
        self.bAutoExp = False
        self.Gain = 0
        # rotation angle, possible values either 0, 90, 180, 270 Degree
        self.angleRot = 0
        # default 1: no inverting, -1: inverting
        # list means [x, y]
        self.imgInvert = [1, 1]
        self.turnCount = 4

        self.FList = []
        self.fwhmXList = []
        self.fwhmYList = []
        self.markerListX = []
        self.markerListY = []

        self.populateColormaps()

        self.Setting = GUISettings()
        self.Setting.guirestore(self.ui, QtCore.QSettings('./GUI/saved.ini', QtCore.QSettings.IniFormat))
        #try:
        #    self.markerListX, self.markerListY = self.Setting.readLines('./GUI/savedLines.txt')
        #except FileNotFoundError:
        #    pass
        self.initMarker()

        self.show()

        self.btn_Exit.clicked.connect(self.close)
        self.btn_addMarker.clicked.connect(self.addMarker)
        self.btn_removeMarker.clicked.connect(self.removeMarker)
        self.line_expTime.returnPressed.connect(self.setIntTime)
        self.line_average.returnPressed.connect(self.setAverage)
        self.slide_expTime.valueChanged.connect(self.changeIntText)
        self.btn_Save.clicked.connect(self.saveAnalysis)

        self.btn_turnLeft.clicked.connect(self.turnLeft)
        self.btn_turnRight.clicked.connect(self.turnRight)
        self.btn_invertH.toggled.connect(self.invertHorizontal)
        self.btn_invertV.toggled.connect(self.invertVertical)

        self.check_chopper.toggled.connect(self.chopperText)
        self.setupCamList()
        self.combo_Cam.activated.connect(self.initCam)
        self.combo_colourmap.activated.connect(self.Plot_CamImage)
        self.combo_gain.activated.connect(self.setGain)
        self.Image.scene().sigMouseClicked.connect(self.mouseClicked)
        self.check_analysis.toggled.connect(self.analysisEnable)
        self.tab_analysis.currentChanged.connect(self.clearCurves)

        self.btn_autoExposure.toggled.connect(self.autoExposure)
        self.btn_autoGain.toggled.connect(self.autoGain)
        self.Main()

    def autoExposure(self):
        if self.btn_autoExposure.isChecked():
            self.bAutoExp = True
            self.line_expTime.setDisabled(True)
            self.slide_expTime.setDisabled(True)
            self.setAutoExposure(True)
        else:
            self.bAutoExp = False
            self.setAutoExposure(False)
            self.line_expTime.setDisabled(False)
            self.slide_expTime.setDisabled(False)
            self.line_expTime.setText(str(self.cam.getIntTime()/1000))
            self.setIntTime()

    def autoGain(self):
        if self.btn_autoGain.isChecked():
            self.combo_gain.setDisabled(True)
            self.setAutoGain(True)
        else:
            self.setAutoGain(False)
            self.combo_gain.setDisabled(False)
            newGain = self.cam.getGain()
            new = self.combo_gain.findText(str(newGain))
            self.combo_gain.setCurrentIndex(new)
            self.setGain()

    def setAutoExposure(self, bstate):
        self.cam.autoExposure(bstate)

    def setAutoGain(self, bstate):
        self.cam.autoGain(bstate)

    def clearCurves(self):
        self.removeFitPlots()
        self.addFitPlots()

    def populateColormaps(self):
        for i in range(len(colormaps)):
            self.combo_colourmap.addItem(colormaps[i])

    def setIntTime(self):
        self.IntTime = float(self.line_expTime.text())
        if not self.bAutoExp:
            self.cam.setIntegrationTime(self.IntTime)

    def setGain(self):
        try:
            self.Gain = float(self.combo_gain.currentText())
            self.cam.setGain(self.Gain)
        except ValueError:
            pass

    def analysisEnable(self):

        if self.check_analysis.isChecked():
            self.enableAnalysis()
        else:
            self.disableAnalysis()

    def getCurrentCam(self):
        return {0: str(self.combo_Cam.currentText())}

    def setupCamList(self):
        """
        Read the Analog Input Channels from all connected NI DAQ Devices.
        :return: List of channels and devices in GUI
        """
        self.combo_Cam.clear()
        CamList = self.cam.returnCamList()
        for cam in CamList:
            self.combo_Cam.addItem(cam)

    def invertHorizontal(self):
        if self.btn_invertH.isChecked():
            self.imgInvert[0] = -1
        else:
            self.imgInvert[0] = 1
        self.resetImage()

    def invertVertical(self):
        if self.btn_invertV.isChecked():
            self.imgInvert[1] = -1
        else:
            self.imgInvert[1] = 1
        self.resetImage()

    def turnLeft(self):
        if self.turnCount == 0:
            self.turnCount = 3
        else:
            self.turnCount -= 1
        self.resetImage()

    def turnRight(self):
        self.turnCount += 1
        self.resetImage()

    def resetImage(self):
        self.currentAvg = 0
        imageCalculations["Image"] = 0

    def mouseClicked(self, evt):
        pos = evt.pos() ## using signal proxy turns original arguments into a tuple
        if self.vb.sceneBoundingRect().contains(pos):
            mousePoint = self.vb.mapSceneToView(pos)
            self.markerListY[0].setPos(mousePoint.y())
            self.markerListX[0].setPos(mousePoint.x())


    def chopperText(self):
        if self.check_chopper.isChecked():
            self.check_chopper.setText("with Chopper")
        else:
            self.check_chopper.setText("w/o Chopper")

    @staticmethod
    def createTimeStamp_Date():
        """
        Main folder for data saving is named after Timestamp to ensure
        unique name tha tcannot be overriden accidently
        :return: date
        """

        return str(datetime.now().strftime("%Y%m%d"))

    @staticmethod
    def createTimeStamp_Time():
        """
        Main folder for data saving is named after Timestamp to ensure
        unique name tha tcannot be overriden accidently
        :return: time
        """

        return str(datetime.now().strftime("%H%M%S"))

    def saveAnalysis(self):

        date = self.createTimeStamp_Date()
        path = config.SavingDestination+date

        if not os.path.exists(path):
            os.makedirs(path)

        name = self.createTimeStamp_Time()

        np.savetxt(path+"\\"+date + "_" + name +"_image.txt", imageCalculations["Image"])

        self.createOrOpenListFile(config.SavingDestination, name, date)
        self.zoomToROI()

        minX = float(self.xMin)*float(self.size[0])
        maxX = self.xMax*float(self.size[0])
        minY = self.yMin*float(self.size[1])
        maxY = self.yMax*float(self.size[1])
        
        analysis(imageCalculations["Image"], self.size[0], self.size[1],
                 minX, maxX, minY, maxY, path, date, name)

    def createOrOpenListFile(self, path, name, date):

        self.fn = path + "\\BeamprofileList.txt"
        header = False
        file = open(self.fn, 'a+')
        if os.stat(self.fn).st_size == 0:
            header = True
        if header:
            file.write('#date\t time\t FWHMX(µm)\t FWHMY(µm)\t FWHMX_Slice(µm)\t FWHMY_Slice(µm)\n')
        file.write(date + "\t")
        file.write(name + "\t")
        try:
            file.write(str(np.round(imageCalculations["FWHM_X"], 2))+"\t")
            file.write(str(np.round(imageCalculations["FWHM_Y"], 2)) + "\t")
        except KeyError:
            file.write(str("0\t"))
            file.write(str("0\t"))
        try:
            file.write(str(np.round(imageCalculations["SliceFWHM_X"], 2)) + "\t")
            file.write(str(np.round(imageCalculations["SliceFWHM_Y"], 2)) + "\n")
        except KeyError:
            file.write(str("0\t"))
            file.write(str("0\t"))

        file.close()

    def zoomToROI(self):
        self.xMin = imageCalculations["Center_GaussFitX"] - len(imageCalculations["SumX"])/3
        self.xMax = imageCalculations["Center_GaussFitX"] + len(imageCalculations["SumX"])/3

        self.yMin = imageCalculations["Center_GaussFitY"] - len(imageCalculations["SumY"])/3
        self.yMax = imageCalculations["Center_GaussFitY"] + len(imageCalculations["SumY"])/3

        if self.btn_Zoom.isChecked():
            self.bZoom = True
        else:
            self.bZoom = False

    def setAverage(self):
        if self.line_average.text() == '':
            self.line_average.setText('1')
        return int(self.line_average.text())

    def initIntegrationTime(self):

        intLimits = self.getIntLimits()
        self.slide_expTime.setMinimum(intLimits[0]*1000)
        self.slide_expTime.setMaximum(intLimits[1]*1000)

        self.turnCount, self.imgInvert = \
            self.Setting.guirestore(self.ui,
                                    QtCore.QSettings('./GUI/saved.ini', QtCore.QSettings.IniFormat),
                                    str(self.getCurrentCam()[0]))

        try:
            self.IntTime = float(self.line_expTime.text())
        except ValueError:
            self.IntTime = 1
        self.line_expTime.setText(str(self.IntTime))
        self.slide_expTime.setValue(self.calcNewSlider())


    def initGain(self):

        try:
            old = int(self.combo_gain.currentText())
        except ValueError:
            old = None
        gain = self.getGainLimits()
        self.combo_gain.clear()
        for i in range(gain[0], gain[1]+1):
            self.combo_gain.addItem(str(i))
        if old:
            ind = self.combo_gain.findText(str(old))
            self.combo_gain.setCurrentIndex(ind)

    def calcNewSlider(self):
        return self.IntTime*10**3

    def getIntTime(self):
        return self.exposureTime

    def initMarker(self):

        for entry in self.markerListX:
            self.vb.addItem(entry)
        for entry in self.markerListY:
            self.vb.addItem(entry)

    def addMarker(self):

        self.markerListX.append(pg.InfiniteLine(pen=(215, 255, 26), angle=90, movable=True))
        self.markerListY.append(pg.InfiniteLine(pen=(215, 255, 0), angle=0, movable=True))

        self.vb.addItem(self.markerListX[-1])
        self.vb.addItem(self.markerListY[-1])

    def removeMarker(self):
        try:
            self.vb.removeItem(self.markerListX[-1])
            self.vb.removeItem(self.markerListY[-1])

            del self.markerListX[-1]
            del self.markerListY[-1]
        except IndexError:
            pass

    def changeIntText(self):
        self.line_expTime.setText(str(np.round(self.slide_expTime.value()*10**(-3), 3)))
        self.setIntTime()

    def getIntLimits(self):
         return self.cam.getIntLimits()

    def getGainLimits(self):
        return self.cam.getGainRange()

    def init_ui(self):

        # Left Image
        self.vb = self.ImageBox.addViewBox(row=0, col=0)
        self.vb.setAspectLocked(True)
        self.Image = pg.ImageItem()

        # Lines
        self.xCenter = pg.InfiniteLine(pen=(215, 0, 26), angle=90)
        self.yCenter = pg.InfiniteLine(pen=(215, 0, 26), angle=0)

        # Add Items to Viewbox
        self.vb.addItem(self.Image)

        # Plots

        self.ImageBox.ci.layout.setColumnMaximumWidth(1, 100)
        self.ImageBox.ci.layout.setRowMaximumHeight(1, 100)


    def enableAnalysis(self):

        self.PlotY = self.ImageBox.addPlot(row=0, col=1)
        self.PlotX = self.ImageBox.addPlot(row=1, col=0)
        self.PlotX.setXLink(self.Image.getViewBox())
        self.PlotY.setYLink(self.Image.getViewBox())

        # Set Layout
        self.ImageBox.ci.layout.setColumnMaximumWidth(1, 100)
        self.ImageBox.ci.layout.setRowMaximumHeight(1, 100)

        self.vb.addItem(self.xCenter)
        self.vb.addItem(self.yCenter)
        self.addFitPlots()

    def disableAnalysis(self):

        self.vb.removeItem(self.xCenter)
        self.vb.removeItem(self.yCenter)
        self.removeFitPlots()

        self.ImageBox.removeItem(self.PlotY)
        self.ImageBox.removeItem(self.PlotX)

    def Main(self):

        self.Plot_CamImage()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

    def initCam(self):

        self.cam.openCommunications(self.getCurrentCam())
        self.initIntegrationTime()
        self.initGain()
        if not self.bAutoExp:
            self.cam.setCameraParameters(self.IntTime, pixelFormat)
        self.size = self.cam.getPixelSize(self.getCurrentCam())
        self.bcamera = 1

    def update(self):

        if self.bcamera != None:
            self.updateCamera()
        QtGui.QApplication.processEvents()

    def updateCamera(self):
        for self.currentAvg in range(self.setAverage()):
            try:
                image, nbCam = self.cam.getImage()
                if len(image[0][:, 0]) == 0 or len(image[0][0, :]) == 0:
                    print('Image Delivery Error, repeat attempts')
                    return
                if len(image) == 1:
                    image = image[0]
                image = np.rot90(image, k=self.turnCount%4)
                image = image[::self.imgInvert[0], ::self.imgInvert[1]]
                if self.currentAvg == 0:
                    imageCalculations["Image"] = image
                else:
                    imageCalculations["Image"] = \
                        ((self.currentAvg-1)*imageCalculations["Image"] +
                         image)/self.currentAvg
                self.update_CamImage()
                self.calculateCenter()
                if self.check_analysis.isChecked():
                    self.analyse()
            except genicam._genicam.LogicalErrorException:
                self.cam.close()
                self.initCam()

    def analyse(self):
        self.updateImgCalc()
        self.cacluateFluence()
        self.calculateBeamProportion()

    def cacluateFluence(self):
        try:
            power = float(self.line_power.text())
            anglePump = float(self.line_angle.text())
            repitionRate = float(self.line_reprate.text())
            try:
                FWHMx = imageCalculations["FWHM_X"] * float(self.size[0])
                FWHMy = imageCalculations["FWHM_Y"] * float(self.size[1])
                F =calculateFluenceFromPower(power, anglePump, repitionRate, FWHMx, FWHMy, self.check_chopper.isChecked())
                self.FList.append(F)
                if self.check_average.isChecked():
                    if len(self.FList) < int(self.line_nbAverages.text()):
                        F = sum(self.FList[:]) / len(self.FList)
                    else:
                        F = sum(self.FList[-int(self.line_nbAverages.text()):])/int(self.line_nbAverages.text())
                self.label_fluence.setText(str(round(F, 2)))
            except KeyError:
                self.label_fluence.setText("No Fit")
        except ValueError:
            pass

    def calculateBeamProportion(self):

        if self.bFit:
            try:
                ratio1 = imageCalculations["FWHM_X"] / imageCalculations["FWHM_Y"]
            except ZeroDivisionError:
                ratio1 = 0

            self.label_0ratio.setText(str(round(ratio1, 2)))

            fwhmX = round(imageCalculations["FWHM_X"] * float(self.size[1]), 1)
            self.fwhmXList.append(fwhmX)
            fwhmY = round(imageCalculations["FWHM_Y"] * float(self.size[1]), 1)
            self.fwhmYList.append(fwhmY)
            if self.check_average.isChecked():
                if len(self.fwhmXList) < int(self.line_nbAverages.text()):
                    fwhmX = sum(self.fwhmXList[:]) / len(self.fwhmXList)
                    fwhmY = sum(self.fwhmYList[:]) / len(self.fwhmYList)
                else:
                    fwhmX = sum(self.fwhmXList[-int(self.line_nbAverages.text()):]) / int(self.line_nbAverages.text())
                    fwhmY = sum(self.fwhmYList[-int(self.line_nbAverages.text()):]) / int(self.line_nbAverages.text())

            try:
                self.label_0fwhmx.setText(str(fwhmX) + ' µm')
                self.label_0fwhmy.setText(str(fwhmY) + ' µm')
            except KeyError:
                pass

            try:
                self.label_0fwhmx_slice.setText(str(round((imageCalculations["SliceFWHM_X"] * float(self.size[0])), 1)) + ' µm')
                self.label_0fwhmy_slice.setText(str(round((imageCalculations["SliceFWHM_Y"] * float(self.size[1])), 1)) + ' µm')
            except KeyError:
                pass

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.timer.stop()
            print(str(self.getCurrentCam()[0]))
            #self.Setting.saveLines('./GUI/savedLines.txt', self.markerListX, self.markerListY)
            self.Setting.guisave(self.ui,
                                 QtCore.QSettings('./GUI/saved.ini', QtCore.QSettings.IniFormat),
                                 str(self.getCurrentCam()[0]), self.turnCount, self.imgInvert)
            sys.exit()
            event.accept()
        else:
            event.ignore()

    def updateImgCalc(self):

        xgauss, ygauss = self.gaussFit()

        if type(xgauss) != int:
            imageCalculations["FWHM_X"] = abs(xgauss[2] * 2.354)
            imageCalculations["Center_GaussFitX"] = xgauss[1]
            self.bFitX = True
        else: 
            self.bFitX = False

        if type(ygauss) != int:
            imageCalculations["Center_GaussFitY"] = ygauss[1]
            imageCalculations["FWHM_Y"] = abs(ygauss[2] * 2.354)
            self.bFitY = True
        else:
            self.bFitY = False

        try:
            imageCalculations["SliceY"] = imageCalculations["Image"][int(imageCalculations["Center_GaussFitY"]), :]
            imageCalculations["SliceX"] = imageCalculations["Image"][:, int(imageCalculations["Center_GaussFitX"])]

            xgaussSlice, ygaussSlice = self.gaussFitSlice()
            imageCalculations["SliceFWHM_X"] = abs(xgaussSlice[2] * 2.354)
            imageCalculations["SliceFWHM_Y"] = abs(ygaussSlice[2] * 2.354)
        except (IndexError, KeyError, TypeError):
            pass

    def calculateCenter(self):
        imageCalculations["SumY"] = np.sum(imageCalculations["Image"], axis=0)
        imageCalculations["SumX"] = np.sum(imageCalculations["Image"], axis=1)
        imageCalculations["Center_X"] = len(imageCalculations["SumX"]) / 2
        imageCalculations["Center_Y"] = len(imageCalculations["SumY"]) / 2

    def setPlotBoundaries(self):
        self.PlotY.setYRange(0, len(imageCalculations["SumY"]))
        self.PlotX.setXRange(0, len(imageCalculations["SumX"]))

    def gaussFit(self):

        self.bFit =True

        ygauss = self.fitGauss(imageCalculations["SumY"], [np.max(imageCalculations["SumY"]), np.argmax(imageCalculations["SumY"]), 0.5, imageCalculations["SumY"][0]])
        xgauss = self.fitGauss(imageCalculations["SumX"], [np.max(imageCalculations["SumX"]), np.argmax(imageCalculations["SumX"]), 0.5,imageCalculations["SumY"][0]])
        QtGui.QApplication.processEvents()
        if type(ygauss) is int or type(xgauss) is int:
            self.bFit = False

        if self.bFit:
            imageCalculations["GaussX"] = self.gaus(np.linspace(
                0, len(imageCalculations["SumX"]), len(imageCalculations["SumX"])), xgauss[0], xgauss[1], xgauss[2],xgauss[3])
            imageCalculations["GaussY"] = self.gaus(np.linspace(
                0, len(imageCalculations["SumY"]),
                len(imageCalculations["SumY"])), ygauss[0], ygauss[1], ygauss[2], ygauss[3])
        return xgauss, ygauss

    def gaus(self, x, a, x0, sigma, c):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))+c

    def fitGauss(self, data, init):
        result = False
        i = 0
        while not result and i < 5:
            try:
                i += 1
                popt, pcov = curve_fit(self.gaus, np.linspace(0, len(data),
                                                      len(data)), data, p0=init)
                result = True
            except RuntimeError:
                pass
        if result:
            return popt
        else:
            return 0

    def getColormap(self):
        return self.combo_colourmap.currentText()

    def Plot_CamImage(self):

        try:
            colormap = cm.get_cmap(self.getColormap())  # cm.get_cmap("CMRmap")
        except:
            print('Colormap not found/illegal: ' + str(self.combo_colourmap.currentText()))
            colormap = cm.get_cmap('inferno')
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        #lut = color
        # Apply the colormap
        self.Image.setLookupTable(lut, update = True)

        if self.check_analysis.isChecked():
            self.addFitPlots()

    def removeFitPlots(self):
        self.PlotY.clear()
        self.PlotX.clear()

    def addFitPlots(self):
        self.curve = self.PlotY.plot(pen=(215, 128, 26), name = 'Integral')
        self.curve2 = self.PlotX.plot(pen=(215, 128, 26), name = 'Integral')
        self.curve6 = self.PlotX.plot(pen=(255, 0, 0), name = 'Fit')
        self.curve7 = self.PlotY.plot(pen=(255, 0, 0), name = 'Fit')
        self.sliceY = self.PlotY.plot(pen=(215, 125, 50), name = 'Slice')
        self.sliceX = self.PlotX.plot(pen=(215, 125, 50), name = 'Slice')
        self.XCenter0 = self.PlotX.addLine(x=0, movable=True, pen=(215, 0, 26))
        self.YCenter0 = self.PlotY.addLine(y=0, movable=True, pen=(215, 0, 26))

    def update_CamImage(self):

        self.Image.setImage(imageCalculations["Image"], autoLevels=self.check_colorMap.isChecked(), levels=(0, 255))
        self.bar_pixSaturation.setValue(np.max(imageCalculations["Image"]))
        if self.check_analysis.isChecked():
            try:

                if self.tab_analysis.currentIndex() == 0:
                    self.curve.setData(x=imageCalculations["SumY"], y=np.arange(len(imageCalculations["SumY"])))
                    self.curve2.setData(imageCalculations["SumX"])
                    self.curve7.setData(x=imageCalculations["GaussY"], y=np.arange(len(imageCalculations["GaussY"])))
                    self.curve6.setData(imageCalculations["GaussX"])


                else:
                    self.sliceX.setData(imageCalculations["SliceX"])
                    self.curve7.setData(x=imageCalculations["Slice_GaussY"], y=np.arange(len(imageCalculations["Slice_GaussY"])))
                    self.sliceY.setData(x = imageCalculations["SliceY"],  y=np.arange(len(imageCalculations["GaussY"])))
                    self.curve6.setData(imageCalculations["Slice_GaussX"])

                if self.bZoom:
                    self.PlotY.setYRange(self.yMin, self.yMax)
                    self.PlotX.setXRange(self.xMin, self.xMax)
                    self.init = 0
                elif self.init == 0:
                    self.setPlotBoundaries()
                    self.init = 1

                self.XCenter0.setValue(imageCalculations["Center_GaussFitX"])
                self.YCenter0.setValue(imageCalculations["Center_GaussFitY"])
                self.xCenter.setValue(imageCalculations["Center_GaussFitX"])
                self.yCenter.setValue(imageCalculations["Center_GaussFitY"])

            except (KeyError, AttributeError):
                pass

    def gaussFitSlice(self):

        bFit =True
        ygauss = self.fitGauss(imageCalculations["SliceY"], [20000, 750, 100, 50])
        xgauss = self.fitGauss(imageCalculations["SliceX"], [20000, 750, 100, 50])

        QtGui.QApplication.processEvents()
        if type(ygauss) is int or type(xgauss) is int:
            bFit = False

        if bFit:
            imageCalculations["Slice_GaussX"] = self.gaus(np.linspace(
                0, len(imageCalculations["SliceX"]), len(imageCalculations["SliceX"])), xgauss[0], xgauss[1], xgauss[2],xgauss[3])
            imageCalculations["Slice_GaussY"] = self.gaus(np.linspace(
                0, len(imageCalculations["SliceY"]),
                len(imageCalculations["SliceY"])), ygauss[0], ygauss[1], ygauss[2], ygauss[3])
        return xgauss, ygauss



def main():

    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())


# catch excpetion, print it and wait for user input
# designed so that the python console does not close immediatly
# when it is run from a shortcut

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(traceback.format_exc())
        print(e)
        input("Press enter to exit...\n")
        raise
