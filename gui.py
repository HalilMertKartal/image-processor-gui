import os

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QLineEdit, QInputDialog, QDialog, QDialogButtonBox, QFormLayout

import processImg as pi
import cv2
from PIL import Image
import matplotlib.pyplot as plt

saved_img = "TEMP/LOADED.png"
manipulated_img = "TEMP/RESULTING.png"

isGrayscaled = False
isNoised = False

cropRejected = False


# mode = 0 for "rgb color balance" or mode = 1 for "crop"
class InputDialog(QDialog):
    def __init__(self, mode, parent=None):
        super().__init__(parent)

        if mode == 1:
            self.x1 = QLineEdit(self)
            self.x2 = QLineEdit(self)
            self.y1 = QLineEdit(self)
            self.y2 = QLineEdit(self)
            buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

            layout = QFormLayout(self)
            layout.addRow("Enter left x coordinate of cropping area",  self.x1)
            layout.addRow("Enter right x coordinate of cropping area", self.x2)
            layout.addRow("Enter upper y coordinate of cropping area", self.y1)
            layout.addRow("Enter lower y coordinate of cropping area", self.y2)

            layout.addWidget(buttonBox)

            buttonBox.accepted.connect(self.accept)
            buttonBox.rejected.connect(self.reject)

        elif mode == 0:
            self.r = QLineEdit(self)
            self.g = QLineEdit(self)
            self.b = QLineEdit(self)
            buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

            layout = QFormLayout(self)
            layout.addRow("Enter a number for the r value", self.r)
            layout.addRow("Enter a number for the g value", self.g)
            layout.addRow("Enter a number for the b value", self.b)

            layout.addWidget(buttonBox)

            buttonBox.accepted.connect(self.accept)
            buttonBox.rejected.connect(self.reject)


    def accept(self) -> None:
        global cropRejected
        QDialog.accept(self)
        cropRejected = False

    def reject(self) -> None:
        global cropRejected
        QDialog.reject(self)
        cropRejected = True


    def getInputs(self, mode):
        if mode == 1:
            return self.x1.text(), self.x2.text(), self.y1.text(), self.y2.text()
        elif mode == 0:
            return self.r.text(), self.g.text(), self.b.text()


def isInitialized():
    global saved_img
    if saved_img == 'TEMP/LOADED.png':
        return False
    else:
        return True


class Error(Exception):
    """Base class for other exceptions"""
    pass


class WrongInputError(Error):
    pass


# Pop up an error message if a user try to do operations before loading an image
def notLoadedError(functionName):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText("NO IMAGES HAVE BEEN LOADED")
    msg.setInformativeText(functionName + " could not be executed.")
    msg.setWindowTitle("FATAL ERROR")
    msg.setDetailedText(
        "Your command could not be executed because you haven't been uploaded any image yet. Please try again after loading an image.")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Image Processing GUI")
        MainWindow.resize(1727, 900)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.savedimg = QtWidgets.QLabel(self.centralwidget)
        self.savedimg.setGeometry(QtCore.QRect(25, 250, 820, 611))
        self.savedimg.setText("")
        self.savedimg.setPixmap(QtGui.QPixmap(saved_img))
        self.savedimg.setScaledContents(True)
        self.savedimg.setObjectName("savedimg")
        self.manipulatedimg = QtWidgets.QLabel(self.centralwidget)
        self.manipulatedimg.setGeometry(QtCore.QRect(880, 250, 820, 611))
        self.manipulatedimg.setText("")
        self.manipulatedimg.setPixmap(QtGui.QPixmap(manipulated_img))
        self.manipulatedimg.setScaledContents(True)
        self.manipulatedimg.setObjectName("manipulatedimg")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1720, 21))
        self.menubar.setObjectName("menubar")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.file = QtWidgets.QMenu(self.menubar)
        self.file.setObjectName("file")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")

        MainWindow.setStatusBar(self.statusbar)

        self.actionSave_Image_2 = QtWidgets.QAction(MainWindow)
        self.actionSave_Image_2.setObjectName("actionSave_Image_2")

        self.actionSave_Image = QtWidgets.QAction(MainWindow)
        self.actionSave_Image.setObjectName("actionSave_Image")

        self.actionBlur = QtWidgets.QAction(MainWindow)
        self.actionBlur.setObjectName("actionBlur")

        self.actionDeblur = QtWidgets.QAction(MainWindow)
        self.actionDeblur.setObjectName("actionDeblur")

        self.actionGreyScale = QtWidgets.QAction(MainWindow)
        self.actionGreyScale.setObjectName("actionGreyScale")

        self.actionFlipImage = QtWidgets.QAction(MainWindow)
        self.actionFlipImage.setObjectName("actionFlipImage")

        self.actionMirrorImage = QtWidgets.QAction(MainWindow)
        self.actionMirrorImage.setObjectName("actionMirrorImage")

        self.actionRotateImage = QtWidgets.QAction(MainWindow)
        self.actionRotateImage.setObjectName("actionRotateImage")

        self.actionReverseColorOfImage = QtWidgets.QAction(MainWindow)
        self.actionReverseColorOfImage.setObjectName("actionReverseColorOfImage")

        self.actionAdjustBrightness = QtWidgets.QAction(MainWindow)
        self.actionAdjustBrightness.setObjectName("actionAdjustBrightness")

        self.actionAdjustSaturation = QtWidgets.QAction(MainWindow)
        self.actionAdjustSaturation.setObjectName("actionAdjustSaturation")

        self.actionDetectEdges = QtWidgets.QAction(MainWindow)
        self.actionDetectEdges.setObjectName("actionDetectEdges")

        self.actionAddNoise = QtWidgets.QAction(MainWindow)
        self.actionAddNoise.setObjectName("actionAddNoise")

        self.actionAdjustContrast = QtWidgets.QAction(MainWindow)
        self.actionAdjustContrast.setObjectName("actionAdjustContrast")

        self.actionChangeColorBalanceOfImage = QtWidgets.QAction(MainWindow)
        self.actionChangeColorBalanceOfImage.setObjectName("actionChangeColorBalance")

        self.actionCropImage = QtWidgets.QAction(MainWindow)
        self.actionCropImage.setObjectName("actionCropImage")

        # Order of buttons can be changed by changing the order of the menuEdit lines below
        self.menuEdit.addAction(self.actionBlur)
        self.menuEdit.addAction(self.actionDeblur)
        self.menuEdit.addAction(self.actionGreyScale)
        self.menuEdit.addAction(self.actionCropImage)
        self.menuEdit.addAction(self.actionFlipImage)
        self.menuEdit.addAction(self.actionMirrorImage)
        self.menuEdit.addAction(self.actionRotateImage)
        self.menuEdit.addAction(self.actionReverseColorOfImage)
        self.menuEdit.addAction(self.actionChangeColorBalanceOfImage)
        self.menuEdit.addAction(self.actionAdjustBrightness)
        self.menuEdit.addAction(self.actionAdjustContrast)
        self.menuEdit.addAction(self.actionAdjustSaturation)
        self.menuEdit.addAction(self.actionAddNoise)
        self.menuEdit.addAction(self.actionDetectEdges)

        self.file.addAction(self.actionSave_Image_2)
        self.file.addAction(self.actionSave_Image)
        self.menubar.addAction(self.file.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Call functions when a corresponding button is triggered
        self.actionSave_Image.triggered.connect(lambda: self.saveimg_act())
        self.actionBlur.triggered.connect(lambda: self.blur_act())
        self.actionDeblur.triggered.connect(lambda: self.deblur_act())
        self.actionGreyScale.triggered.connect(lambda: self.greyScale_act())
        self.actionSave_Image_2.triggered.connect(lambda: self.load_img_act())
        self.actionFlipImage.triggered.connect(lambda: self.flipImage_act())
        self.actionMirrorImage.triggered.connect(lambda: self.mirrorImage_act())
        self.actionRotateImage.triggered.connect(lambda: self.rotateImage_act())
        self.actionReverseColorOfImage.triggered.connect(lambda: self.reverseColorOfImage_act())
        self.actionAdjustBrightness.triggered.connect(lambda: self.adjustBrightness_act())
        self.actionAdjustSaturation.triggered.connect(lambda: self.adjustSaturation_act())
        self.actionDetectEdges.triggered.connect(lambda: self.detectEdges_act())
        self.actionAddNoise.triggered.connect(lambda: self.addNoise_act())
        self.actionAdjustContrast.triggered.connect(lambda: self.adjustContrast_act())
        self.actionChangeColorBalanceOfImage.triggered.connect(lambda: self.changeColorBalance_act())
        self.actionCropImage.triggered.connect(lambda: self.cropImage_act())

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Image Processing GUI", "Image Processing GUI"))

        # Add names to buttons
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.file.setTitle(_translate("MainWindow", "File"))
        self.actionSave_Image_2.setText(_translate("MainWindow", "Load Image"))
        self.actionSave_Image.setText(_translate("MainWindow", "Save Image"))
        self.actionBlur.setText(_translate("MainWindow", "Blur Image"))
        self.actionDeblur.setText(_translate("MainWindow", "Deblur Image"))
        self.actionGreyScale.setText(_translate("MainWindow", "Greyscale Image"))
        self.actionFlipImage.setText(_translate("MainWindow", "Flip Image"))
        self.actionMirrorImage.setText(_translate("MainWindow", "Mirror Image"))
        self.actionRotateImage.setText(_translate("MainWindow", "Rotate Image"))
        self.actionReverseColorOfImage.setText(_translate("MainWindow", "Reverse Color Of Image"))
        self.actionAdjustBrightness.setText(_translate("MainWindow", "Adjust Brightness Of Image"))
        self.actionAdjustSaturation.setText(_translate("MainWindow", "Adjust Saturation Of Image"))
        self.actionDetectEdges.setText(_translate("MainWindow", "Detect The Edges In The Image"))
        self.actionAddNoise.setText(_translate("MainWindow", "Add Noise To The Image"))
        self.actionAdjustContrast.setText(_translate("MainWindow", "Adjust Contrast Of The Image"))
        self.actionChangeColorBalanceOfImage.setText(_translate("MainWindow", "Change Color Balance Of The Image"))
        self.actionCropImage.setText(_translate("MainWindow", "Crop Image"))

    def saveimg_act(self):
        # Do not change the default image
        if not isInitialized():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("FATAL ERROR")
            msg.setInformativeText("Save image cannot be executed.")
            msg.setWindowTitle("Fatal Error")
            msg.setDetailedText(
                "Your command could not be executed because no loaded image found to save. Please try again after loading an image.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
            return

        global manipulated_img
        text, ok = QtWidgets.QFileDialog.getSaveFileName()

        if text != "" and ok:
            if (".jpg" in text) or (".png" in text) or (".gif" in text):
                manImg = pi.Image.open(manipulated_img)
                manImg.save(text)
            else:
                manImg = pi.Image.open(manipulated_img).convert("RGB")
                manImg.save(text + ".jpg")

    def blur_act(self):
        # Do not change the default image
        if not isInitialized():
            notLoadedError("Blur Image")
            return

        global manipulated_img
        pi.blurImage(manipulated_img).save("TEMP/temp.jpg")
        manipulated_img = "TEMP/temp.jpg"
        self.manipulatedimg.setPixmap(QtGui.QPixmap("TEMP/temp.jpg"))

    def deblur_act(self):
        # Do not change the default image
        if not isInitialized():
            notLoadedError("Deblur Image")
            return

        global manipulated_img
        pi.deblurImage(manipulated_img).save("TEMP/temp.jpg")
        manipulated_img = "TEMP/temp.jpg"
        self.manipulatedimg.setPixmap(QtGui.QPixmap("TEMP/temp.jpg"))

    def greyScale_act(self):
        global manipulated_img, isGrayscaled

        # Do not change the default image
        if not isInitialized():
            notLoadedError("Grayscale Image")
            return
        # Do not greyscale the image if it's already have been greyscaled
        if isGrayscaled:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("FATAL ERROR")
            msg.setInformativeText("Grayscale Image could not be executed.")
            msg.setWindowTitle("Fatal Error")
            msg.setDetailedText(
                "Your command could not be executed because you can not greyscale an image which is have already been greyscaled. Please try again.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
            return


        pi.grayScaleImage(manipulated_img).save("TEMP/temp.jpg")
        manipulated_img = "TEMP/temp.jpg"
        self.manipulatedimg.setPixmap(QtGui.QPixmap("TEMP/temp.jpg"))

        isGrayscaled = True

    def load_img_act(self):
        global saved_img, manipulated_img, isGrayscaled, isNoised
        text, ok = QtWidgets.QFileDialog.getOpenFileName(filter="*.jpg;*.jpeg;*.png;")

        if ok:
            saved_img = text
            manipulated_img = text
            self.savedimg.setPixmap(QtGui.QPixmap(saved_img))
            self.manipulatedimg.setPixmap(QtGui.QPixmap(saved_img))
            isGrayscaled = False
            isNoised = False

    def flipImage_act(self):
        # Do not change the default image
        if not isInitialized():
            notLoadedError("Flip Image")
            return

        global manipulated_img
        pi.flipImage(manipulated_img).save("TEMP/temp.jpg")
        manipulated_img = "TEMP/temp.jpg"
        self.manipulatedimg.setPixmap(QtGui.QPixmap("TEMP/temp.jpg"))

    def mirrorImage_act(self):
        # Do not change the default image
        if not isInitialized():
            notLoadedError("Mirror Image")
            return

        global manipulated_img
        pi.mirrorImage(manipulated_img).save("TEMP/temp.jpg")
        manipulated_img = "TEMP/temp.jpg"
        self.manipulatedimg.setPixmap(QtGui.QPixmap("TEMP/temp.jpg"))

    def rotateImage_act(self):
        # Do not change the default image
        if not isInitialized():
            notLoadedError("Rotate Image")
            return

        global manipulated_img
        msg = QInputDialog()
        rotateAngle, ok = QInputDialog.getDouble(msg, "QInputDialog().getText()",
                                                 "Enter rotate angle (in degrees)", QLineEdit.Normal)
        if ok:
            pi.rotateImage(manipulated_img, rotateAngle).save("TEMP/temp.jpg")
            manipulated_img = "TEMP/temp.jpg"
            self.manipulatedimg.setPixmap(QtGui.QPixmap("TEMP/temp.jpg"))

    def reverseColorOfImage_act(self):
        # Do not change the default image
        if not isInitialized():
            notLoadedError("Reverse Color Of The Image")
            return

        global manipulated_img
        pi.reverseColorOfImage(manipulated_img).save("TEMP/temp.jpg")
        manipulated_img = "TEMP/temp.jpg"
        self.manipulatedimg.setPixmap(QtGui.QPixmap("TEMP/temp.jpg"))

    def adjustBrightness_act(self):
        # Do not change the default image
        if not isInitialized():
            notLoadedError("Adjust Brightness Of The Image")
            return

        global manipulated_img
        msg = QInputDialog()
        scale, ok = QInputDialog.getDouble(msg, "QInputDialog().getText()",
                                           "Enter the brightness level", QLineEdit.Normal)
        if ok:
            pi.adjustBrightness(manipulated_img, scale).save("TEMP/temp.jpg")
            manipulated_img = "TEMP/temp.jpg"
            self.manipulatedimg.setPixmap(QtGui.QPixmap("TEMP/temp.jpg"))

    def adjustSaturation_act(self):
        # Do not change the default image
        if not isInitialized():
            notLoadedError("Adjust Saturation Of The Image")
            return

        if isGrayscaled:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("FATAL ERROR")
            msg.setInformativeText("Adjust Saturation could not be executed.")
            msg.setWindowTitle("Fatal Error")
            msg.setDetailedText(
                "Your command could not be executed on grayscale images. Please try again.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
            return

        global manipulated_img
        msg = QInputDialog()
        scale, ok = QInputDialog.getDouble(msg, "QInputDialog().getText()",
                                           "Enter the saturation level", QLineEdit.Normal)
        if ok:
            pi.adjustSaturation(manipulated_img, scale).save("TEMP/temp.jpg")
            manipulated_img = "TEMP/temp.jpg"
            self.manipulatedimg.setPixmap(QtGui.QPixmap("TEMP/temp.jpg"))

    def detectEdges_act(self):
        # Do not change the default image
        if not isInitialized():
            notLoadedError("Detect Edges In The Image")
            return

        global manipulated_img
        if isNoised:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("FATAL ERROR")
            msg.setInformativeText("Detect Edges could not be executed.")
            msg.setWindowTitle("Fatal Error")
            msg.setDetailedText(
                "Your command could not be executed because you can't detect edges in a noisy image. Please try again.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
        else:
            pi.detectEdges(manipulated_img).save("TEMP/temp.jpg")
            manipulated_img = "TEMP/temp.jpg"
            self.manipulatedimg.setPixmap(QtGui.QPixmap("TEMP/temp.jpg"))

    def addNoise_act(self):
        global isNoised
        # Do not change the default image
        if not isInitialized():
            notLoadedError("Add Noise To The Image")
            return

        global manipulated_img
        cv2.imwrite("TEMP/temp.jpg", pi.addNoise(manipulated_img))
        manipulated_img = "TEMP/temp.jpg"
        self.manipulatedimg.setPixmap(QtGui.QPixmap("TEMP/temp.jpg"))
        isNoised = True

    def adjustContrast_act(self):
        # Do not change the default image
        if not isInitialized():
            notLoadedError("Adjust Contrast Of The Image")
            return

        global manipulated_img
        msg = QInputDialog()
        scale, ok = QInputDialog.getDouble(msg, "QInputDialog().getText()",
                                           "Enter the contrast level", QLineEdit.Normal)
        if ok:
            pi.adjustContrast(manipulated_img, scale).save("TEMP/temp.jpg")
            manipulated_img = "TEMP/temp.jpg"
            self.manipulatedimg.setPixmap(QtGui.QPixmap("TEMP/temp.jpg"))

    def changeColorBalance_act(self):
        # Do not change the default image
        if not isInitialized():
            notLoadedError("Change Color Balance Of The Image")
            return

        global manipulated_img


        if isGrayscaled:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("FATAL ERROR")
            msg.setInformativeText("Change Color Balance could not be executed.")
            msg.setWindowTitle("Fatal Error")
            msg.setDetailedText(
                "That operation is not valid for grayscale images. Please try again.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
            return



        dialog = InputDialog(0)
        dialog.exec()

        # If cancelled, do nothing
        if cropRejected:
            self.manipulatedimg.setPixmap(QtGui.QPixmap(manipulated_img))

            try:
                os.remove("TEMP/plot_form.jpg")
            except Exception:
                pass
            return

        r, g, b = dialog.getInputs(0)

        succeeded = False

        try:
            r = float(r)
            g = float(g)
            b = float(b)
            succeeded = True
        except ValueError:
            pass

        if succeeded:
            pi.changeColorBalance(manipulated_img, r, g, b).save("TEMP/temp.jpg")
            manipulated_img = "TEMP/temp.jpg"
            self.manipulatedimg.setPixmap(QtGui.QPixmap("TEMP/temp.jpg"))
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("FATAL ERROR")
            msg.setInformativeText("Change Color Balance could not be executed.")
            msg.setWindowTitle("Fatal Error")
            msg.setDetailedText(
                "Your command could not be executed because given input(s) was not a number. Please try again.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
            self.manipulatedimg.setPixmap(QtGui.QPixmap(manipulated_img))

    def cropImage_act(self):
        if not isInitialized():
            notLoadedError("Crop Image")
            return

        global manipulated_img

        im = Image.open(manipulated_img)
        plt.imshow(im)
        plt.savefig("TEMP/plot_form.jpg")
        self.manipulatedimg.setPixmap(QtGui.QPixmap("TEMP/plot_form.jpg"))

        dialog = InputDialog(1)
        dialog.exec()

        # If cancelled, do nothing
        if cropRejected:
            self.manipulatedimg.setPixmap(QtGui.QPixmap(manipulated_img))

            try:
                os.remove("TEMP/plot_form.jpg")
            except Exception:
                pass
            return

        leftX, rightX, upY, lowY = dialog.getInputs(1)

        succeeded = False

        try:
            leftX = int(leftX)
            rightX = int(rightX)
            upY = int(upY)
            lowY = int(lowY)
            succeeded = True
        except ValueError:
            pass

        w, h = im.size

        try:
            if succeeded:
                if (not 0 <= leftX < w) or (not leftX < rightX <= w) or (not 0 <= upY < h) or (not upY < lowY <= h):
                    raise WrongInputError
        except WrongInputError or ValueError or TypeError:
            succeeded = False

        if succeeded:
            pi.cropImage(manipulated_img, leftX, upY, rightX, lowY).save("TEMP/temp.jpg")
            manipulated_img = "TEMP/temp.jpg"
            self.manipulatedimg.setPixmap(QtGui.QPixmap("TEMP/temp.jpg"))
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("FATAL ERROR")
            msg.setInformativeText("Crop Image could not be executed.")
            msg.setWindowTitle("Fatal Error")
            msg.setDetailedText(
                "Your command could not be executed because given input(s) was not a number or makes no sense. Please try again.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()

            self.manipulatedimg.setPixmap(QtGui.QPixmap(manipulated_img))

        try:
            os.remove("TEMP/plot_form.jpg")
        except Exception:
            pass


if __name__ == "__main__":
    import sys

    try:
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())

    finally:
        try:
            os.remove("TEMP/temp.jpg")
        except FileNotFoundError or FileExistsError:
            exit(0)
        finally:
            exit(0)
