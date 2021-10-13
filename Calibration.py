import numpy as np
import cv2
import glob

def readCameraParameters(path):
    cvFile = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    cameraMatrix = cvFile.getNode("K").mat() # have to specify the type to retrieve, or get FileNode objeect
    distCoeffs = cvFile.getNode("D").mat()
    cvFile.release()
    return [cameraMatrix, distCoeffs]

def saveCameraParameters(cameraMatrix, distCoeffs, path):
    cvFile = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cvFile.write("K", cameraMatrix)
    cvFile.write("D", distCoeffs)
    cvFile.release()

''' 
Apply camera calibration for images in dir
param : image dir, file prefix, image format(jpg or png), real square size(m), width(num of cross points), height
'''
def calibrate(dirpath, prefix, imageFormat, squareSize, width=9, height=6):
    terminationCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]

    images = glob.glob(dirpath+'/' + prefix + '*.' + imageFormat)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(width-1, height-1,0) * squareSize
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * squareSize
    print(objp)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fName in images:
        img = cv2.imread(fName)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), terminationCriteria) # refining
            imgpoints.append(corners2)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            # cv2.imshow(fName, img)
            # cv2.waitKey(0)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]
