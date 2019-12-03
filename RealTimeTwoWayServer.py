import numpy as np
import cv2
import imagezmq
from Estimation import FaceRemover
from threading import Thread
import socket
from time import sleep

#Much of the Object Detection and Message Passing Code here is adatped (not a direct copy paste) from Adrian Rosebrocks PyImageSearch Tuturials and ImUtils Library:
#https://github.com/jrosebr1/imutils

# List of Class Labels that MobileNet SSD was trained to detect
CLASSES = ["background", "airplane", "bicycle", "bird", "boat",
	   "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
	   "dog", "horse", "motorcycle", "person", "potted plant", "sheep",
	   "sofa", "train", "tv"]

#Most Recently Processed Image Frame
STREAM = False
FRAME = None

def Server(CThreshold=0.7):
    global FRAME, STREAM

    #Initialize ImageHub
    print("Awaiting Incoming Connection...")
    ImageHub = imagezmq.ImageHub()

    #Initialize Face Remover
    Privacy = FaceRemover()

    #Stream Loop
    STREAM = True
    while STREAM:
        #Recieve camera name and acknowledge with a receipt reply
        (CamName, Frame) = ImageHub.recv_image()
        ImageHub.send_reply(b'OK')
        if Frame is None:
            continue
        
        #Detect and Blur any Faces in the Image
        Frame, FLAG = Privacy.DetectAndBlur(Frame)
        
        #Write the Device name to be displayed on the recieved Image Frame
        cv2.putText(Frame, CamName, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        #Publish postprocessed Frame to be sent back to (or another) Client
        FRAME = Frame
                    
        #Temporary Testing
        #cv2.imshow("Testing 123...",Frame)
        #cv2.waitKey(1)    

def MessagePassing():
    global STREAM
    
    # Send Data Back to Client
    SERVER_IP = "172.24.118.97"

    #Initialize Sender Object for the Server
    print("Connecting to Client...")
    Sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(SERVER_IP))

    #Obtain Hostname, initialize Video Stream, and Warm Up the Camera
    ServerName = socket.gethostname()
    
    #Send the Processed Image Frames Back to the Client
    sleep(10)
    while True:
        try:
            if FRAME is None:
                continue
            
            Frame = FRAME
            Sender.send_image(ServerName, Frame)

        except KeyboardInterrupt:
            print("Shutting down...")
            STREAM = False
            break

if __name__=='__main__':
    WebServer = Thread(target=Server)
    WebServer.start()
    MessagePassing()
    WebServer.join()
