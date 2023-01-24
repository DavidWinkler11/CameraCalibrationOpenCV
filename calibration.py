import numpy as np 
import cv2 as cv 
import glob

##################### FIND CHESSBOARD CORNERS - objPoints AND imgPoints #################################

chessboardSize = (8,6)

#Termination criteria 
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ......, (6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)

#Arrays to store object points and image points from all the images
objPoints = []
imgPoints = []

images = glob.glob('images/*.png')

for image in images:
    print(image)
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #Find the corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    #If found, add object points, image points (after refining them)
    if ret == True:
        objPoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgPoints.append(corners2)

        #Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        
        cv.imshow('img', img)
        cv.waitKey(1000)

cv.destroyAllWindows()

##################### CALIBRATION #################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)

print("\nCamera Calibrated:", ret)
print("\nCameraMatrix:\n", cameraMatrix)
print("\nDistortion Parameters:\n", dist)
print("\nRotation Vectors:\n", rvecs)
print("\nTranslation Vectors:\n", tvecs)

#################### UNDISTORTION #################################
img = cv.imread('images/image11.png')
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

#Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

#Crop the image 
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.png', dst)


#Reprojection Error 
mean_error = 0

for i in range(len(objPoints)):
    imgPoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgPoints[i], imgPoints2, cv.NORM_L2)/len(imgPoints2)
    mean_error += error

print("\nTotal error: {}".format(mean_error/len(objPoints)))
print("\n\n\n")









