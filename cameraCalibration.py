import cv2 as cv
import os
import numpy as np

# Checker board size
CHESS_BOARD_DIM = (8, 6)

# The size of Square in the checker board.
SQUARE_SIZE = 27  # millimeters

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


calib_data_path = "calib_data"
CHECK_DIR = os.path.isdir(calib_data_path)


if not CHECK_DIR:
    os.makedirs(calib_data_path)
    print(f'"{calib_data_path}" Directory is created')

else:
    print(f'"{calib_data_path}" Directory already Exists.')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
obj3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)

obj3D[:, :2] = np.mgrid[0 : CHESS_BOARD_DIM[0], 0 : CHESS_BOARD_DIM[1]].T.reshape(
    -1, 2
)
obj3D *= SQUARE_SIZE
#print(obj3D)


# Arrays to store object points and image points from all the images.
objPoints3D = []  # 3d point in real world space
imgPoints2D = []  # 2d points in image plane.

# The images directory path
image_dir_path = "images"

files = os.listdir(image_dir_path)
for file in files:
    print("Showing photo: " + file)
    imagePath = os.path.join(image_dir_path, file)
    # print(imagePath)

    image = cv.imread(imagePath)
    grayScale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(image, CHESS_BOARD_DIM, None)
    if ret == True:
        objPoints3D.append(obj3D)
        corners2 = cv.cornerSubPix(grayScale, corners, (3, 3), (-1, -1), criteria)
        imgPoints2D.append(corners2)

        img = cv.drawChessboardCorners(image, CHESS_BOARD_DIM, corners2, ret)
        cv.imshow('img', image)
        cv.waitKey(500)

cv.destroyAllWindows()
print(" ---------------------------------------------------------------")
print("| Calibration in progress, a little patience is appreciated.... |")
print(" ---------------------------------------------------------------")
h, w = image.shape[:2]
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objPoints3D, imgPoints2D, grayScale.shape[::-1], None, None
)
print("--------------------------------------------------------------------------------------\n")
print("CALIBRATED")
print("--------------------------------------------------------------------------------------\n")
print("Camera Calibrated:", ret)
print("--------------------------------------------------------------------------------------\n")
print("CameraMatrix:\n", mtx)
print("--------------------------------------------------------------------------------------\n")
print("Distortion Parameters:\n", dist)
print("--------------------------------------------------------------------------------------\n")
#print("Rotation Vectors:\n", rvecs)
#print("--------------------------------------------------------------------------------------\n")
#print("Translation Vectors:\n", tvecs)
#print("--------------------------------------------------------------------------------------\n")
print("Undistortion in progress!")

undistortedCounter = 0;

for file in files:    
    imagePath = os.path.join(image_dir_path, file)

    #Undistort
    img = cv.imread(imagePath)
    h, w = img.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    dst = cv.undistort(img, mtx, dist, None, newCameraMatrix)

    #Crop the image 
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(f"{calib_data_path}/undistorted{undistortedCounter}.png", dst)
    undistortedCounter += 1

print("Undistortion complete")
#Reprojection Error 
mean_error = 0

for i in range(len(objPoints3D)):
    imgPoints, _ = cv.projectPoints(objPoints3D[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgPoints2D[i], imgPoints, cv.NORM_L2)/len(imgPoints)
    mean_error += error

print("\nTotal error: {}".format(mean_error/len(objPoints3D)))
print("\n\n\n")
