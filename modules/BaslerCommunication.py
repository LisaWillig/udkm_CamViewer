from pypylon import pylon
from pypylon import genicam
import pyqtgraph as pg
import numpy as np

# test cameras
namesCamsToUse = {0: 'MOKEBS_10'}


class BaslerCam:

    def __init__(self):

        self.tlFactory = pylon.TlFactory.GetInstance()

        # Get all attached devices and exit application if no device is found.
        self.devices = self.tlFactory.EnumerateDevices()
        if len(self.devices) == 0:
            raise pylon.RUNTIME_EXCEPTION("No camera present.")
        # Create an array of instant cameras for the found devices and avoid
        # exceeding a maximum number of devices.
        self.cameras = pylon.InstantCameraArray(min(len(self.devices), len(self.devices)))
        self.camList = []

    def openCommunications(self, camsToUse):
        self.camNames = camsToUse
        camContext = 0

    # Create and attach all Pylon Devices.
        for i, cam in enumerate(self.cameras):
            try:
                cam.Attach(self.tlFactory.CreateDevice(self.devices[i]))
                cam.Open()

                if cam.DeviceUserID() not in self.camNames.values():
                    cam.Close()
                else:
                    camContext = self.setCamContextByKey(cam.DeviceUserID())
                    cam.SetCameraContext(camContext)
                    self.camList.append(i)

            except genicam._genicam.RuntimeException:
                self.cameras.DetachDevice()
                self.cameras.DestroyDevice()

    def setCamContextByKey(self, deviceID):

        for key, value in self.camNames.items():
            if value == deviceID:
                return key

    def setCameraParameters(self, exposureTime=10, pixelFormat='Mono8'):
        j = 0
        self.stopAcquisition()
        for i in self.camList:
            cam = self.cameras[i]

            if type(exposureTime) == dict:
                exposure = exposureTime[j]
            else:
                exposure = exposureTime
            if cam.PixelFormat != pixelFormat:
                cam.PixelFormat = pixelFormat

            cam.MaxNumBuffer = 2
            try:
                cam.Gain = cam.Gain.Min
            except genicam.LogicalErrorException:
                cam.GainRaw = cam.GainRaw.Min
            cam.Width = cam.Width.Max
            cam.Height = cam.Height.Max
            cam.ExposureTimeRaw = int(exposure*1000)
            j += 1

    def getGainRange(self):
        gain = []
        for i in self.camList:
            cam = self.cameras[i]
            gain.append(cam.GainRaw.Min)
            gain.append(cam.GainRaw.Max)
        return gain

    def autoExposure(self, bstate):
        self.stopAcquisition()
        for i in self.camList:
            cam = self.cameras[i]
            if bstate:
                cam.ExposureAuto = 'Continuous'
            else:
                cam.ExposureAuto = 'Off'
        self.startAcquisition()

    def autoGain(self, bstate):
        self.stopAcquisition()
        for i in self.camList:
            cam = self.cameras[i]
            if bstate:
                cam.GainAuto = 'Continuous'
            else:
                cam.GainAuto = 'Off'
        self.startAcquisition()

    def getIntTime(self):
        for i in self.camList:
            cam=self.cameras[i]
            return cam.ExposureTimeAbs.Value

    def getGain(self):
        for i in self.camList:
            cam = self.cameras[i]
            return cam.GainRaw.Value

    def startAcquisition(self):
        for i in self.camList:
            cam = self.cameras[i]
            cam.StartGrabbing(1)

    def stopAcquisition(self):
        for i in self.camList:
            cam = self.cameras[i]
            cam.StopGrabbing()

    def getImage(self):
        return self.startGrab(1)

    def startGrab(self, countOfImagesToGrab):
        # Starts grabbing for all cameras starting with index 0. The grabbing
        # is started for one camera after the other. That's why the images of all
        # cameras are not taken at the same time.
        # However, a hardware trigger setup can be used to cause all cameras to grab images synchronously.
        # According to their default configuration, the cameras are
        # set up for free-running continuous acquisition.
        imageArray = []
        contextArray = []
        for j in self.camList:
            cam = self.cameras[j]
            # Grab c_countOfImagesToGrab from the cameras.
            for i in range(countOfImagesToGrab):
                if not cam.IsGrabbing():
                    cam.StartGrabbing(1)
                img, cameraContextValue = self.retrieveImage(cam)
                imageArray.append(img)
                contextArray.append(cameraContextValue)
        return imageArray, contextArray

    def grabResultImage(self, cam):
        return cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    def retrieveImage(self, cam):

        grabResult = self.grabResultImage(cam)

        # When the cameras in the array are created the camera context value
        # is set to the index of the camera in the array.
        # The camera context is a us♠er settable value.
        # This value is attached to each grab result and can be used
        # to determine the camera that produced the grab result.
        cameraContextValue = grabResult.GetCameraContext()

        # Print the index and the model name of the camera.
        # print("Camera ", cameraContextValue, ": ",
        #       cam.GetCameraContext())
        # Now, the image data can be processed.
        #print("GrabSucceeded: ", grabResult.GrabSucceeded())
        retrieve = 0
        while retrieve == 0:
            try:
                img = grabResult.GetArray()
                retrieve = 1
            except TypeError:
                pass

        return img, cameraContextValue

    def getIntLimits(self):
        for i in self.camList:
            cam = self.cameras[i]
            min = cam.ExposureTimeRaw.Min/1000
            max = cam.ExposureTimeRaw.Max/1000
            return min, max

    def setIntegrationTime(self, exposureTime):
        self.stopAcquisition()
        for i in self.camList:
            cam = self.cameras[i]
            cam.ExposureTimeRaw = int(exposureTime*1000)
        self.startAcquisition()

    def setGain(self, Gain):
        self.stopAcquisition()
        for i in self.camList:
            cam=self.cameras[i]
            cam.GainRaw = int(Gain)
        self.startAcquisition()

    def getPixelSize(self, camera):

        conversion = np.genfromtxt("modules\\camPixelConversion.txt", dtype='str')
        for i in self.camList:
            cam = self.cameras[i]
            if cam.DeviceUserID() == camera[0]:
                dev = cam.DeviceModelName()
                for i, entry in enumerate(conversion[:, 0]):
                    if entry == dev:
                        return conversion[i, 1], conversion[i, 2]

        return 0, 0

    def close(self):
        for i in self.camList:
            cam = self.cameras[i]
            cam.StopGrabbing()
            cam.Close()
            self.camList.remove(i)

    def returnCamList(self):
        camList = []
        for i, cam in enumerate(self.cameras):
            try:
                cam.Attach(self.tlFactory.CreateDevice(self.devices[i]))
                cam.Open()
                camList.append(cam.DeviceUserID())
            except genicam._genicam.RuntimeException:
                self.cameras.DetachDevice()
                self.cameras.DestroyDevice()
        return camList

def main():

    cams = BaslerCam()
    cams.openCommunications(namesCamsToUse)
    cams.getGainRange()
    cams.setCameraParameters(6)
    cams.startAcquisition()
    for i in range(2):
        img, cont = cams.getImage()
        print(img)

if __name__ == '__main__':
        main()