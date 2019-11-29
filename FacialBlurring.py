import cv2
import numpy as np
from PoseEstimator import Estimator

class FacialBlurring(object):
    def __init__(self):
        self.HEstimator = Estimator(hands=False)
    
    def BlurFaces(self, Image):
        self.HEstimator.Analyze(Image)
        Facepoints = self.HEstimator.GetFace()
        
        #Apply Morphological Transformations to connect the dots between Facial Keypoints
        
        #Segment out Faces
        
        #Blur Faces
        
        #Reapply Masks to Image
        
        return BlurredFaces


HEstimator = Estimator(hands=False)





if __name__ == '__main__':
    pass
