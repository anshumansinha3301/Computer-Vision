# computer_vision_concepts.py
# A collection of important Computer Vision concepts implemented in Python

import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations

# ------------------- 1. Reading and displaying an image -------------------
image = cv2.imread('image.jpg')  # Read an image from file
cv2.imshow('Original Image', image)  # Display the original image
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close all windows

# ------------------- 2. Converting to grayscale -------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert BGR image to grayscale
cv2.imshow('Grayscale Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------- 3. Image Smoothing / Noise Reduction -------------------
gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
median_blur = cv2.medianBlur(gray, 5)  # Apply median filter

# ------------------- 4. Edge Detection -------------------
edges_canny = cv2.Canny(gray, 100, 200)  # Canny edge detection
cv2.imshow('Edges', edges_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------- 5. Thresholding -------------------
ret, binary_global = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # Global thresholding
binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)  # Adaptive thresholding

# ------------------- 6. Morphological Operations -------------------
kernel = np.ones((5,5), np.uint8)  # Define a 5x5 kernel

eroded = cv2.erode(binary_global, kernel, iterations=1)  # Erosion

dilated = cv2.dilate(binary_global, kernel, iterations=1)  # Dilation

opened = cv2.morphologyEx(binary_global, cv2.MORPH_OPEN, kernel)  # Opening
closed = cv2.morphologyEx(binary_global, cv2.MORPH_CLOSE, kernel)  # Closing

# ------------------- 7. Contour Detection -------------------
contours, hierarchy = cv2.findContours(binary_global, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
image_contours = image.copy()
cv2.drawContours(image_contours, contours, -1, (0,255,0), 2)  # Draw contours
cv2.imshow('Contours', image_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------- 8. Feature Detection (SIFT, ORB) -------------------
orb = cv2.ORB_create()  # ORB feature detector
keypoints, descriptors = orb.detectAndCompute(gray, None)  # Detect keypoints and compute descriptors
image_orb = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0))  # Draw keypoints
cv2.imshow('ORB Features', image_orb)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------- 9. Template Matching -------------------
template = cv2.imread('template.jpg', 0)  # Read template
res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)  # Match template
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # Get best match location
h, w = template.shape
top_left = max_loc
bottom_right = (top_left[0]+w, top_left[1]+h)
cv2.rectangle(image, top_left, bottom_right, (255,0,0), 2)  # Draw rectangle on match
cv2.imshow('Template Match', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------- 10. Video Capture and Frame Processing -------------------
cap = cv2.VideoCapture(0)  # Capture video from webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    edges_frame = cv2.Canny(gray_frame, 100, 200)  # Edge detection
    cv2.imshow('Video Edges', edges_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# ------------------- 11. Simple Object Tracking using Color -------------------
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert frame to HSV color space
lower_blue = np.array([100,150,0])
upper_blue = np.array([140,255,255])
mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)  # Threshold for blue color
result = cv2.bitwise_and(frame, frame, mask=mask)  # Apply mask to original frame
cv2.imshow('Color Tracking', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------- 12. Perspective Transformation -------------------
pts1 = np.float32([[50,50],[200,50],[50,200],[200,200]])  # Source points
pts2 = np.float32([[10,100],[200,50],[100,250],[220,220]])  # Destination points
matrix = cv2.getPerspectiveTransform(pts1, pts2)  # Compute transformation matrix
warped = cv2.warpPerspective(image, matrix, (300,300))  # Apply perspective warp
cv2.imshow('Warped Perspective', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
