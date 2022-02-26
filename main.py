import cv2
import numpy as np
from sklearn.cluster import KMeans


#TODO:cut image at first
#TODO:Enter number of clusters


def find_cell_colour(cell):
	(B, G, R) = cv2.split(cell)


	meanB = np.sum(B)/(B.shape[0]*B.shape[1])

	meanG = np.sum(G) / (G.shape[0] * G.shape[1])

	meanR = np.sum(R) / (R.shape[0] * R.shape[1])
	print(meanB , meanG ,meanR)
	if meanR > meanG or meanR > meanB:
		return (0,0,255)
	else:
		return (0,255,0)

img = cv2.imread("C:/andrey/Mushtra/images/mushtra_sept.jpg")

img = cv2.resize(img, (0,0), fx = 0.2, fy =0.15)
img = img[20:, 50:]
(B, G, R) = cv2.split(img)
# show each channel individually
# cv2.imshow("Red", R)
# cv2.imshow("Green", G)
# cv2.imshow("Blue", B)



gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = G+B-R
edges = cv2.Canny(gray,10,600,apertureSize = 3)

minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,150,minLineLength=minLineLength,maxLineGap=maxLineGap)

x_centroids = []
y_centroids = []
for x in range(0, len(lines)):
	for x1,y1,x2,y2 in lines[x]:
		#cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)

		#находим х.у горизонтальных и вертикальных линий
		if abs(y1-y2)<5:
			#cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
			y_centroids.append([y1, y2])

		if abs(x1-x2) < 5:
			#cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
			x_centroids.append([x1, x2])


kmeans = KMeans(n_clusters=13, random_state=0).fit(x_centroids)

x_centroids = []
for i in kmeans.cluster_centers_:
	x = int ((i[0]+i[1])/2)

	x_centroids.append(x)
	cv2.line(img, (x, 0), (x, 1000), (0, 0, 0), 1)

x_centroids = sorted(x_centroids)

kmeans = KMeans(n_clusters=31, random_state=0).fit(y_centroids)

y_centroids = []
for i in kmeans.cluster_centers_:
	y = int ((i[0]+i[1])/2)

	y_centroids.append(y)
	cv2.line(img, (0, y), (1000, y), (0, 0, 0), 1)

y_centroids = sorted(y_centroids)


for i, line_x in enumerate(x_centroids[:-2]):
	for j, line_y in enumerate(y_centroids[:-2]):
		next_line_x = x_centroids[i + 1]
		next_line_y = y_centroids[j + 1]

		#centers of cells
		x = int((line_x + next_line_x) / 2)
		y = int((line_y + next_line_y) / 2)

		cell = img[line_y : next_line_y, line_x : next_line_x]
		find_cell_colour(cell)

		# cv2.imshow("filename", cell)
		#
		# cv2.waitKey(0)


		image = cv2.circle(img,
		                   (x, y),
		                 radius=5,
		                   color=find_cell_colour(cell),
		                 #color=(0,0,0),#colour(img, x_centroids[i], x2, y_centroids[j], y2),
		                 thickness=-1)


cv2.imshow("filename",img)

cv2.waitKey(0)