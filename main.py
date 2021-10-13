import cv2
import Tracker
import Calibration as Calib
import os

if __name__ == '__main__':
    dirName = "images"
    filePrefix = "image"
    fileType = "jpg"
    boxLength = 0.0025
    cameraParamFile = "camera.yaml"

    # Camera Calibration
    if True: #not os.path.exists("camera.yaml"):
        ret, mtx, dist, rvecs, tvecs = Calib.calibrate(dirName, filePrefix, fileType, boxLength)
        Calib.saveCameraParameters(mtx, dist, cameraParamFile)
        print("Calibration is finished. RMS: ", ret)

    mtx, dist = Calib.readCameraParameters(cameraParamFile)

    # ARuco Marker Tracking
    Tracker.trackArUcoMarker(mtx, dist)

    print("Done")