import cv2
import numpy as np

# 0-16, 17-21, 22-26, (0,17), (21,22), (26,16) (inclusive)
Image = cv2.imread("./Example.jpg")
print(Image.shape)

# Draw lines between Face Keypoints
FacePoints = np.load("FaceKeyPoints.npy")
print(FacePoints)
'''
# Draw Line connecting Face Keypoints
Start = (FacePoints[0][0][0],FacePoints[0][0][1])

#cv2.line(Image, Start, (FacePoints[0][1][0],FacePoints[0][1][1]), (255,0,0), 5)

for i in range(1,17): #FacePoints.shape[1]-1):
	cv2.line(Image, Start, (FacePoints[0][i][0],FacePoints[0][i][1]), (255,0,0), 5)
	Start = (FacePoints[0][i][0],FacePoints[0][i][1])

Start = (FacePoints[0][17][0],FacePoints[0][17][1])
for i in range(18,27): #FacePoints.shape[1]-1):
	cv2.line(Image, Start, (FacePoints[0][i][0],FacePoints[0][i][1]), (255,0,0), 5)
	Start = (FacePoints[0][i][0],FacePoints[0][i][1])

# (0,17)
Start = (FacePoints[0][0][0],FacePoints[0][0][1])
cv2.line(Image, Start, (FacePoints[0][17][0],FacePoints[0][17][1]), (255,0,0), 5)

# (16,26)
Start = (FacePoints[0][16][0],FacePoints[0][16][1])
cv2.line(Image, Start, (FacePoints[0][26][0],FacePoints[0][26][1]), (255,0,0), 5)
'''

#Create Polygon Coordinates from Face Keypoints to Segment out Facial Areas - # 0-16, 26-17, (17,0)
#And Obtain Boxes for Facial Blurring
Img = np.copy(Image)
Segment = np.zeros_like(Image)
for i in np.arange(FacePoints.shape[0]):
	Points = FacePoints[i,:,:2]
	Polys = np.concatenate((Points[:17,:],Points[17:27,:][::-1]), axis=0).astype(np.int32)
	
	#Create Segmentation Masks from Polygons
	cv2.fillPoly(Segment, pts=[Polys], color=(255,255,255))

	#Obtain Box Coordinates for Cropping and Blurring
	Minx, Miny = np.amin(Polys, axis=0)
	Maxx, Maxy = np.amax(Polys, axis=0)
	Img[Miny:Maxy+1,Minx:Maxx+1,:] = cv2.GaussianBlur(Image[Miny:Maxy+1,Minx:Maxx+1,:], (23,23), 30)

#Only apply the Blurred Pixels belonging to the Segmented Faces Themselves to the Original Image
Image[Segment>0] = Img[Segment>0]

#Apply Facial Masks to Image
#Image[Shell > 0] = 0
#Image[:,:,2][Shell[:,:,2] > 0] = 255

#Display Test Image
cv2.imshow("Test",Image)
cv2.waitKey(0)

if __name__=='__main__':
    pass
