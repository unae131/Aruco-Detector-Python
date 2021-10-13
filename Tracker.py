import numpy as np
import cv2
import cv2.aruco as aruco

def trackArUcoMarker(cameraMatrix, distCoeffs):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
            parameters = aruco.DetectorParameters_create()
            
            # lists of detected ids and the corners
            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=cameraMatrix, distCoeff=distCoeffs)

            if np.all(ids is not None): # markers found

                rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(corners, 0.0011, cameraMatrix, distCoeffs)

                for i in range(0, len(ids)):
                    # print(_objPoints, rvec, tvec)
                    aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec[i], tvec[i], 0.00055)

                aruco.drawDetectedMarkers(frame, corners)

            cv2.imshow('frame', frame)

            key = cv2.waitKey(3) & 0xFF # ignore state of NumLock
            if key == 27: # esc
                break
        
        else:
            print('error')

    cap.release()
    cv2.destroyAllWindows()