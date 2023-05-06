import numpy as np
import cv2
import random
import math

for j in range(200):
	# Create a black image with a white background
	img = np.zeros((500,500,3), dtype=np.uint8)
	gt = np.zeros((500,500,1), dtype=np.uint8)
	# img[:] = (255, 255, 255)

	# Define the number of triangles to draw
	num_triangles = 5

	# Loop over the number of triangles
	for i in range(num_triangles):
	    # Define the vertices of the triangle
	    while True:
	        v1 = (random.randint(0, 500), random.randint(0, 500))
	        v2 = (random.randint(0, 500), random.randint(0, 500))
	        v3 = (random.randint(0, 500), random.randint(0, 500))
	        # Calculate the area of the triangle
	        area = 0.5 * abs(v1[0] * (v2[1] - v3[1]) + v2[0] * (v3[1] - v1[1]) + v3[0] * (v1[1] - v2[1]))
	        # Calculate the length of each side of the triangle
	        side1 = math.sqrt((v2[0]-v1[0])**2 + (v2[1]-v1[1])**2)
	        side2 = math.sqrt((v3[0]-v2[0])**2 + (v3[1]-v2[1])**2)
	        side3 = math.sqrt((v1[0]-v3[0])**2 + (v1[1]-v3[1])**2)
	        # Check if the area and side lengths of the triangle are within the specified range
	        if 20 <= area <= 400 and max(side1, side2, side3) <= 30:
	            break
	    vertices = np.array([v1, v2, v3], np.int32)
	    
	    # Reshape the vertices to be a 1 x 3 x 2 array
	    vertices = vertices.reshape((-1,1,2))
	    
	    # Define a random color for the triangle
	    color = (100, 100, 100)
	    
	    # Draw the triangle on the image
	    cv2.fillPoly(img, [vertices], color)
	    cv2.fillPoly(gt, [vertices], 255)

	    v1 = (random.randint(0, 500), random.randint(0, 500))
	    v2 = (random.randint(10, 30), random.randint(10, 30))
	    cv2.rectangle(img, v1, (v1[0]+v2[0], v1[1]+v2[1]), (100, 100, 100), -1)

	# Display the image
	img = cv2.resize(img, (128, 128))
	gt = cv2.resize(gt, (128, 128))

	cv2.imshow('Triangles', img)
	cv2.imshow('gt', gt)
	cv2.imwrite("toy/input/"+str(j) + ".jpg", img)
	cv2.imwrite("toy/target/"+str(j) + "_mask.png", gt)
	if cv2.waitKey(1) == 27:
		break
	