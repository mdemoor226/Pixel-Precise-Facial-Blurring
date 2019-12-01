import cv2
import numpy as np
from PoseEstimator import Estimator

class FacialBlurring(object):
    def __init__(self):
        self.HEstimator = Estimator(hands=False)
    
    def BlurFaces(self, Image):
        self.HEstimator.Analyze(Image)
        Facepoints = self.HEstimator.GetFace()

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
        
        return Image

if __name__ == '__main__':
    pass
