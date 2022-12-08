import numpy as np
import argparse
import cv2
import cv2.aruco as aruco
from calibrate import load_coefficients

cap = cv2.VideoCapture(0)  # Get the camera source



def track(matrix_coefficients, distortion_coefficients, capture=None):

    while True:
        ret, frame = capture.read()

        if ret == False:
            break

        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)  # Use 5x5 dictionary to find markers
        parameters = aruco.DetectorParameters_create()  # Marker detection parameters
        # lists of ids and the corners beloning to each id
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                                parameters=parameters)
        ids[0].sort()
        controlAruco = []
        compareAruco = []
        if np.all(ids is not None):  # If there are markers found by detector
            for i in range(0, len(ids)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.03, matrix_coefficients,
                                                                           distortion_coefficients)
                (rvec - tvec).any()  # get rid of numpy value array error
                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.1)  # Draw Axis

                R, _ = cv2.Rodrigues(rvec)
                # convert (np.matrix(R).T) matrix to array using np.squeeze(np.asarray()) to get rid off the ValueError: shapes (1,3) and (1,3) not aligned
                R = np.squeeze(np.asarray(np.matrix(R).T))
                if(ids[i] == 0):
                    controlAruco.append(R)
                else:
                    compareAruco.append(R)

        if len(controlAruco) == 1 and len(compareAruco) > 0:
            print('##############')
            x_radians = np.arccos(np.dot(controlAruco[0][0], compareAruco[0][0]))
            x_degrees = x_radians*180/np.pi
            y_radians = np.arccos(np.dot(controlAruco[0][1], compareAruco[0][1]))
            y_degrees = y_radians*180/np.pi
            z_radians = np.arccos(np.dot(controlAruco[0][2], compareAruco[0][2]))
            z_degrees = z_radians*180/np.pi
            print("x: " + str(x_degrees))
            print("y: " + str(y_degrees))
            print("z: " + str(z_degrees))

        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey() & 0xFF
        if key == ord('q'):  # Quit
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--vid_path', type=str, required=True, help='video directory path')

    args = parser.parse_args()
    if(args.vid_path == None):
        cap = cv2.VideoCapture(0)  # Get the camera source
        print("No Video File Selected! Using Default Camera Input!")
    else:
        cap = cv2.VideoCapture(args.vid_path)

    camera_matrix, dist_matrix = load_coefficients("phone.yml")
    track(matrix_coefficients=camera_matrix,distortion_coefficients=dist_matrix, capture=cap)
    