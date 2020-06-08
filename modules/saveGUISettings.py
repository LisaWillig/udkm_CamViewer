#===================================================================
# save "ui" controls and values to registry "setting"
# currently only handles comboboxes editlines & checkboxes
# ui = qmainwindow object
# settings = qsettings object

# Original from:
# https://stackoverflow.com/questions/23279125/python-pyqt4-functions-to-save-and-restore-ui-widget-values
#===================================================================

import inspect
from distutils.util import strtobool
import PyQt5
import pyqtgraph as pg

class GUISettings:

    def guisave(self, ui, settings, camera, imageTurn = None, ImageInvert = None):

        # Save geometry
        #settings.setValue('size', ui.size())
        #settings.setValue('pos', ui.pos())

        settings.beginGroup(str(camera))
        for name, obj in inspect.getmembers(ui):
          # if type(obj) is QComboBox:  # this works similar to isinstance, but missed some field... not sure why?
          if isinstance(obj, PyQt5.QtWidgets.QComboBox):
              name = obj.objectName()  # get combobox name
              if name == "combo_Cam":
                  continue
              index = obj.currentIndex()  # get current index from combobox
              text = obj.itemText(index)  # get the text for current index
              settings.setValue(name, text)  # save combobox selection to registry

          if isinstance(obj, PyQt5.QtWidgets.QLineEdit):
              name = obj.objectName()
              value = obj.text()
              settings.setValue(name, value)  # save ui values, so they can be restored next time

          if isinstance(obj, PyQt5.QtWidgets.QCheckBox):
              name = obj.objectName()
              state = obj.isChecked()
              settings.setValue(name, state)

          if isinstance(obj, PyQt5.QtWidgets.QGroupBox):
              name = obj.objectName()
              state = obj.isChecked()
              settings.setValue(name, state)

          if isinstance(obj, PyQt5.QtWidgets.QRadioButton):
              name = obj.objectName()
              value = obj.isChecked()  # get stored value from registry
              settings.setValue(name, value)

          if isinstance(obj, PyQt5.QtWidgets.QToolButton):
              name = obj.objectName()
              value = obj.isChecked()  # get stored value from registry
              settings.setValue(name, value)

          if isinstance(obj, PyQt5.QtWidgets.QSlider):
              name = obj.objectName()
              value = obj.value()  # get stored value from registry
              settings.setValue(name, value)

          if isinstance(obj, PyQt5.QtWidgets.QSpinBox):
              name = obj.objectName()
              value = obj.value()  # get stored value from registry
              settings.setValue(name, value)

          if imageTurn:
            settings.setValue("imageTurn", imageTurn)
          if ImageInvert:
            settings.setValue("ImageInvert", ImageInvert)

        settings.endGroup()


    def guirestore(self, ui, settings, camera = None):

        if camera != None:
            settings.beginGroup(str(camera))
        else:
            settings.beginGroup("General")
        for name, obj in inspect.getmembers(ui):
            if isinstance(obj, PyQt5.QtWidgets.QComboBox):
                index = obj.currentIndex()  # get current region from combobox
                # text   = obj.itemText(index)   # get the text for new selected index
                name = obj.objectName()
                if name == "combo_Cam":
                    continue
                value = (settings.value(name))

                if value == "":
                    continue

                index = obj.findText(value)  # get the corresponding index for specified string in combobox

                if index == -1:  # add to list if not found
                    obj.insertItems(0, [value])
                    index = obj.findText(value)
                    obj.setCurrentIndex(index)
                else:
                    obj.setCurrentIndex(index)  # preselect a combobox value by index

            if isinstance(obj, PyQt5.QtWidgets.QLineEdit):
                name = obj.objectName()
                value = settings.value(name)  # get stored value from registry
                obj.setText(value)  # restore lineEditFile

            if isinstance(obj, PyQt5.QtWidgets.QCheckBox):
                name = obj.objectName()
                value = settings.value(name)  # get stored value from registry
                if value != None:
                    try:
                        obj.setChecked(strtobool(value))
                    except AttributeError:
                        obj.setChecked(value)

            if isinstance(obj, PyQt5.QtWidgets.QGroupBox):
                name = obj.objectName()
                value = settings.value(name)  # get stored value from registry
                if value != None:
                    try:
                        obj.setChecked(strtobool(value))
                    except AttributeError:
                        obj.setChecked(value)

            if isinstance(obj, PyQt5.QtWidgets.QRadioButton):
                name = obj.objectName()
                value = settings.value(name)  # get stored value from registry
                if value != None:
                    try:
                        obj.setChecked(strtobool(value))
                    except AttributeError:
                        obj.setChecked(value)

            if isinstance(obj, PyQt5.QtWidgets.QToolButton):
                name = obj.objectName()
                value = settings.value(name)  # get stored value from registry
                if value != None:
                    try:
                        obj.setChecked(strtobool(value))
                    except AttributeError:
                        obj.setChecked(value)

            if isinstance(obj, PyQt5.QtWidgets.QSlider):
                name = obj.objectName()
                value = settings.value(name)    # get stored value from registry
                if value != None:
                    obj. setValue(int(value))   # restore value from registry

            if isinstance(obj, PyQt5.QtWidgets.QSpinBox):
                name = obj.objectName()
                value = settings.value(name)    # get stored value from registry
                if value != None:
                    obj. setValue(int(value))   # restore value from registry

        settings.endGroup()

    def imageSettings(self, ui, settings, camera = None):
        if camera != None:
            settings.beginGroup(str(camera))

        try:
            imageTurn = int(settings.value('imageTurn'))
            ImageInvert = settings.value('ImageInvert')
            ImageInvert = [int(x) for x in ImageInvert]
            return imageTurn, ImageInvert
        except TypeError:
            pass

    def saveLines(self, path, x, y):

        if x or y:
            with open(path, 'a+') as file:
                for idx, entry in enumerate(x):
                    pos = entry.value()
                    pen = entry.pen
                    angle = entry.angle
                file.write(str(pos) + '\t' + str(pen) + '\t' + str(angle) + '\n')
                for idx, entry in enumerate(y):
                    pos = entry.value()
                    pen = entry.pen
                    angle = entry.angle
                file.write(str(pos) + '\t' + str(pen) + '\t' + str(angle) + '\n')

    def readLines(self, path):
        y = []
        x = []
        with open(path, 'r') as file:
            lines = file.readlines()
            for entry in lines:
                line = entry.split('\t')
                line[-1] = line[-1].strip('\n')
                if line[-1] == '0':
                    y.append(pg.InfiniteLine(pen='y', angle=int(line[2]), movable=True, pos = int(float(line[0])))) # pg.mkPen(line[1])
                if line[-1] == '90':
                    x.append(pg.InfiniteLine(pen='y', angle=int(line[2]), movable=True, pos = int(float(line[0]))))
        return x, y



