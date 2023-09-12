# USAGE
# python server.py --conf config/config.json

# import the necessary packages
#from pyimagesearch.motion_detection import SingleMotionDetector
#from pyimagesearch.parseyolooutput import ParseYOLOOutput
#from pyimagesearch.keyclipwriter import KeyClipWriter
#from pyimagesearch.utils.conf import Conf

# external library stuff
from onnx_inference.image import Image
from onnx_inference.images_input import Images
from onnx_inference.onnx_object_detection import OnnxObjectDetection
from typing import Dict, Tuple, List
import random

import logging as logger
from datetime import datetime
import numpy as np
#import imagezmq
#import argparse
import imutils
import cv2
# import os
# import time
# import sys
#import zmq

yolo_classnames = ['person', 'bicycle', 'car']
yolo_colors: Dict[str, Tuple[int, int, int]] = {cls_name: [random.randint(0, 255) for _ in range(3)] for k, cls_name in
                                                enumerate(yolo_classnames)}

# def overlap(rectA, rectB):
# 	# check if x1 of rectangle A is greater x2 of rectangle B or if
# 	# x2 of rectangle A is less than x1 of rectangle B, if so, then
# 	# both of them do not overlap and return False
# 	if rectA[0] > rectB[2] or rectA[2] < rectB[0]:
# 		return False

# 	# check if y1 of rectangle A is greater y2 of rectangle B or if
# 	# y2 of rectangle A is less than y1 of rectangle B, if so, then
# 	# both of them do not overlap and return False
# 	if rectA[1] > rectB[3] or rectA[3] < rectB[1]:
# 		return False

# 	# otherwise the two rectangles overlap and hence return True
# 	return True

# calculate the speed using k = 410
# def distance(width):
# 	return (2782/width) - 2.74

# def calculateSpeed(width1, width2, time):
# 	d1 = distance(width1)
# 	d2 = distance(width2)
# 	speed = (d1 -d2)/time # inches per second
# 	speedMPH = speed / (5280) * 3600 # miles per hour
# 	return round(speedMPH, 2)

# def calculateTime(speed, width2):
# 	if speed ==0:
# 		return "literally never"
# 	d2 = distance(width2)
# 	return round(d2 / speed, 2)

# SPEED_LIMIT = 30

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--conf", required=True, 
# 	help="Path to the input configuration file")
# ap.add_argument("-y", "--yolo", required=True,
# 	help="If we should utilize YOLO or just background subtraction")
#

# load the configuration file and initialize the ImageHub object
#conf = Conf(args["conf"])

# useYOLO = None
# if args["yolo"] == "True":
# 	useYOLO = True
# elif args["yolo"] == "False":
# 	useYOLO = False
	
# imageHub = imagezmq.ImageHub()

# initialize the motion detector, the total number of frames read
# thus far, and the spatial dimensions of the frame
#md = SingleMotionDetector(accumWeight=0.1)
total = 0
(W, H) = (None, None)

# # load the COCO class labels our YOLO model was trained on
# labelsPath = os.path.sep.join([conf["yolo_path"], "coco.names"])
# LABELS = open(labelsPath).read().strip().split("\n")

# # initialize a list of colors to represent each possible class label
# np.random.seed(42)
# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
# 	dtype="uint8")

# # derive the paths to the YOLO weights and model configuration
# weightsPath = os.path.sep.join([conf["yolo_path"], "yolov3.weights"])
# configPath = os.path.sep.join([conf["yolo_path"], "yolov3.cfg"])

# # load our YOLO object detector trained on COCO dataset (80 classes)
# # and determine only the *output* layer names that we need from YOLO
# print("[INFO] loading YOLO from disk...")
# net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# ln = net.getLayerNames()
# ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# # initialize the YOLO output parsing object
# pyo = ParseYOLOOutput(conf)

# # initialize key clip writer and the consecutive number of
# # frames that have *not* contained any action
# #kcw = KeyClipWriter(bufSize=conf["buffer_size"])
# #consecFrames = 0
# print("[INFO] starting advanced security surveillance...")

# initialize all variables for speed calc
# firstWidth = None
# firstTime = None

# firstCheck = False
# secondCheck = False

cam = cv2.VideoCapture(1)
yolo_path = "models/yolov7-tiny.onnx"
# yolo_path = "models/yolov5n.onnx"
yolo = OnnxObjectDetection(weight_path=yolo_path, classnames=yolo_classnames)
IMAGE_SIZE = (640, 640)

# start looping over all the frames
while True:
	# receive RPi name and frame from the RPi and acknowledge
	# the receipt
	# (rpiName, frame) = imageHub.recv_image()
	# imageHub.send_reply(b'OK')
	# speed = 0
    
	# resize the frame, convert it to grayscale, and blur it
	result, frame = cam.read()
	if not result:
		print("Logging: Error capturing frame")
		continue

	#frame = imutils.resize(frame, width=IMAGE_SIZE, height=IMAGE_SIZE)
	frame = cv2.resize(frame, IMAGE_SIZE, interpolation = cv2.INTER_AREA)
	print(frame.shape)

	# grab the current timestamp and draw it on the frame
	timestamp = datetime.now()
	cv2.putText(frame, timestamp.strftime(
		"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# if we do not already have the dimensions of the frame,
	# initialize it
	if H is None and W is None:
		(H, W) = frame.shape[:2]

	image = Image(id="none", np_data=frame)
	images = Images(images=[image])

	for i, batch in enumerate(images.create_batch(batch_size=1)):
		logger.info(f"Processing batch: {i} containing {len(batch)} image(s)...")

		raw_out = yolo.predict_object_detection(input_data=batch.to_onnx_input(image_size=IMAGE_SIZE))
		batch.init_detected_objects(raw_out)

		annotations = batch.annotate_objects(input_size=IMAGE_SIZE, letterboxed_image=True, class_colors=yolo_colors)

	frame = cv2.resize(annotations[0], [960, 540], interpolation = cv2.INTER_AREA)
	cv2.imshow("Camera", frame)

	# input = np.array([frame], dtype=np.float16)
	# input = np.transpose(input, (0, 3, 1, 2))
	# raw_out = yolo.predict_object_detection(input_data=input)
	# images.init_detected_objects(raw_out)

	# annotations = images.annotate_objects(input_size=yolo.input_size, letterboxed_image=True, class_colors=yolo_colors)
	# frame = images.images[0].np_data

	# if the total number of frames has reached a sufficient
	# number to construct a reasonable background model, then
	# continue to process the frame
	# if total > conf["frame_count"]:
    # # construct a blob from the input frame and then perform
    #     # a forward pass of the YOLO object detector, giving us
    #     # our bounding boxes and associated probabilities
	# 	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0,
    #         (416, 416), swapRB=True, crop=False)
	# 	net.setInput(blob)
	# 	layerOutputs = net.forward(ln)

    #     # parse YOLOv3 output
	# 	(boxes, confidences, classIDs) = pyo.parse(layerOutputs,
    #         LABELS, H, W)

    #     # apply non-maxima suppression to suppress weak,
    #     # overlapping bounding boxes
	# 	idxs = cv2.dnn.NMSBoxes(boxes, confidences,
    #         conf["confidence"], conf["threshold"])

    #     # ensure at least one detection exists
	# 	if len(idxs) > 0:
    #         # loop over the indexes we are keeping
	# 		for i in idxs.flatten():
    #             # extract the bounding box coordinates
	# 			(x, y) = (boxes[i][0], boxes[i][1])
	# 			(w, h) = (boxes[i][2], boxes[i][3])

	# 			color = [int(c) for c in COLORS[classIDs[i]]]
	# 			cv2.rectangle(frame, (x, y), (x + w, y + h),
	# 				color, 2)
	# 			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
	# 				confidences[i])
	# 			y = (y - 15) if (y - 15) > 0 else h - 15
	# 			cv2.putText(frame, text, (x, y),
	# 				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# # update the background model and increment the total number
	# # of frames read thus far
	# # md.update(gray)
	total += 1

	# show the frame
	#cv2.imshow("Camera", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		print("[INFO] Quitting Video Stream ...")
		break

# do a bit of cleanup
cv2.destroyAllWindows()