# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import subprocess


class Detection:

    def __init__(self,prototxt,model,confidence,sleep,condition = True):


        self.prototxt = prototxt
        self.model = model
        self.confidence = confidence
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                            "sofa", "train", "tvmonitor"]
        #self.COLORS =  np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

        self.sleep = sleep
        self.condition = condition


    def loadModel(self):
        print("[INFO] loading model...")
        subprocess.run(["/usr/bin/unset","DISPLAY","XAUTHORITY"],shell=True)
        #self.net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)



    def startVideo(self):
        #vs = VideoStream(src=0).start()
        self.vs = VideoStream(usePiCamera=True).start()
        time.sleep(2.0)
        self.fps = FPS().start()


    def overFrame(self):
        while self.condition:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 200xels
            frame = self.vs.read()
            frame = imutils.resize(frame, width=200)

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

            # pass the blob through the network and obtain the detections and prediction
            #show the time it takes for this step
            start = time.time()
            self.net.setInput(blob)
            self.detections = self.net.forward()
            end = time.time()
            print("time",end-start)

            for i in np.arange(0, self.detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with
                    # the prediction
                confidence = self.detections[0,0,i,2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > self.confidence:
                    print("Hereeeeee")
                    # extract the index of the class label
                    idx = int(self.detections[0, 0, i, 1])
                    label = "{}: {:.2f}%".format(self.CLASSES[idx],
                           confidence * 100)
                    print(label)

        #cv2.imshow("Frame", frame)
      # key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        # update the FPS counter
            self.fps.update()



    def stopAndDisplay(self):
        self.fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))


    def cleanup(self):
        cv2.destroyAllWindows()
        self.vs.stop()


    def execute(self):

        self.loadModel()
        self.startVideo()

        self.overFrame()
        self.stopAndDisplay()
        self.cleanup()
