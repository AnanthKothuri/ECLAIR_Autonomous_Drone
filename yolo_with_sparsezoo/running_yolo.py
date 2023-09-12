# external library stuff
# from onnx_inference.image import Image
# from onnx_inference.images_input import Images
# from onnx_inference.onnx_object_detection import OnnxObjectDetection
# from typing import Dict, Tuple, List
# import random

import logging as logger
from datetime import datetime
import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--modelPath", required=True, 
	help="Path to the inputted onnx model file")
ap.add_argument("-c", "--cuda", required=True,
	help="True if CUDA is available, false otherwise")
args = vars(ap.parse_args())

yolo_classnames = ['person', 'bicycle', 'car']
yolo_colors: Dict[str, Tuple[int, int, int]] = {cls_name: [random.randint(0, 255) for _ in range(3)] for k, cls_name in
                                                enumerate(yolo_classnames)}
total = 0
(W, H) = (None, None)

cam = cv2.VideoCapture(1)
yolo_path = args["modelPath"]

useCUDA = None
if args["cuda"] == "True":
	useCUDA = True
elif args["cuda"] == "False":
	useCUDA = False
yolo = OnnxObjectDetection(weight_path=yolo_path, classnames=yolo_classnames, cuda=useCUDA)
IMAGE_SIZE = (640, 640)

# start looping over all the frames
while True:
    
	# resize the frame, convert it to grayscale, and blur it
	result, frame = cam.read()
	if not result:
		print("Logging: Error capturing frame")
		continue

	#frame = imutils.resize(frame, width=IMAGE_SIZE, height=IMAGE_SIZE)
	frame = cv2.resize(frame, IMAGE_SIZE, interpolation = cv2.INTER_AREA)

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
	total += 1

	# show the frame
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		print("[INFO] Quitting Video Stream ...")
		break

# do a bit of cleanup
cv2.destroyAllWindows()