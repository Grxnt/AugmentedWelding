import numpy as np
import argparse
import cv2
import cv2.aruco as aruco
from calibrate import load_coefficients

cap = cv2.VideoCapture(0)  # Get the camera source

def relativePosition(rvec1, tvec1, rvec2, tvec2):
    #Get relative position for rvec2 & tvec2, using rvec1 and tvec1 as a reference point
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

    # Inverse the second marker, the right one in the image
    R, _ = cv2.Rodrigues(rvec2)
    R = np.matrix(R).T
    invTvec = np.dot(R, np.matrix(-tvec2))
    invRvec, _ = cv2.Rodrigues(R)

    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]

    #reshape vectors
    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec

def track(matrix_coefficients, distortion_coefficients, capture=None):

    while True:
        # Read a picture frame from video feed
        ret, frame = capture.read()

        if ret == False:
            break

        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)         # Change grayscale
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)  # Use 5x5 dictionary to find markers
        parameters = aruco.DetectorParameters_create()         # Marker detection parameters
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
                aruco.drawDetectedMarkers(frame, corners)  # Draw a square around the markers
                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.1)  # Draw Axis

                # Append detected aruco markers for later processing
                if(ids[i] == 0):
                    controlAruco.append([rvec,tvec])
                else:
                    compareAruco.append([rvec,tvec])

        if len(controlAruco) == 1 and len(compareAruco) > 0:
            print('##############')

            # Find the rotation of the compareAruco marker with reference to the controlAruco
            relative_rvec, relative_tvec = relativePosition(controlAruco[0][0], controlAruco[0][1], compareAruco[0][0], compareAruco[0][1])

            # Convert the Rodriguez Vector into a rotational Matrix
            R, _ = cv2.Rodrigues(relative_rvec)
            # convert (np.matrix(R).T) matrix to array using np.squeeze(np.asarray()) to get rid off the ValueError: shapes (1,3) and (1,3) not aligned
            R = np.squeeze(np.asarray(np.matrix(R).T))

            # Use Rotational matrix to rotate a unit vector
            # Specifically, for this application, we need to compare the rotated z-axis, to the control z-axis, and the rotated y-axis, with the control y-axis
            x = np.array([[1],[0],[0]])
            y = np.array([[0],[1],[0]])
            z = np.array([[0],[0],[1]])

            work_angle = np.arccos(np.dot(R[2], z))
            work_angle = work_angle*180/np.pi
            travel_angle = np.arccos(np.dot(R[0], x))
            travel_angle = travel_angle*180/np.pi

            # Print work and travel angles
            print("Work Angle is: ", work_angle)
            print("Travel Angle is: ", travel_angle)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Wait until a key press is made
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
    