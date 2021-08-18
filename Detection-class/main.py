from detection.DetectionTools import Detection

detector = Detection("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel", 0.7, 0, True)
detector.execute()
