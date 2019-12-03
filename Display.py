import numpy as np
import cv2
import imagezmq

#Initialize ImageHub
print("Awaiting Incoming Connection...")
ImageHub = imagezmq.ImageHub()

#Stream Loop
while True:
        try:
                #Recieve camera name and acknowledge with a receipt reply
                (CamName, Frame) = ImageHub.recv_image()
                ImageHub.send_reply(b'OK')

                cv2.imshow("Facial Privacy", Frame)
                cv2.waitKey(1)

        except Exception:
                print("Either something went wrong or you killed the program. Either way shutting down...")
                break

