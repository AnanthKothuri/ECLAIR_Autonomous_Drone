import cv2
import random
import numpy as np
import tensorflow as tf
from datetime import datetime

#Name of the classes according to class indices.
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
         'hair drier', 'toothbrush']
#Creating random colors for bounding box visualization.
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def detect(img):
    #Load and preprocess the image.
    SIZE = 320
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], im)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    ori_images = [img.copy()]

    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(output_data):
        image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score),3)
        name = names[cls_id]
        color = colors[name]
        name += ' '+str(score)
        cv2.rectangle(image,box[:2],box[2:],color,2)
        cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  
    # plt.imshow(ori_images[0])
    # print("created output image")
    return ori_images[0]


cam = cv2.VideoCapture(1)
IMAGE_SIZE = (640, 640)
DISPLAY_SIZE = (1280, 720)
total = 0

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./saved_model.tflite")
interpreter.allocate_tensors()

print("[INFO] Loaded model")
print("[INFO] Starting video stream")

# start looping over all the frames
while True:

    # resize the frame, convert it to grayscale, and blur it
    result, frame = cam.read()
    if not result:
        print("Logging: Error capturing frame")
        continue

    #frame = imutils.resize(frame, width=IMAGE_SIZE, height=IMAGE_SIZE)
    frame = cv2.resize(frame, IMAGE_SIZE, interpolation = cv2.INTER_AREA)
    frame = detect(frame)

    # resize frame
    frame = cv2.resize(frame, DISPLAY_SIZE, interpolation = cv2.INTER_AREA)

    # grab the current timestamp and draw it on the frame
    timestamp = datetime.now()
    cv2.putText(frame, timestamp.strftime(
    	"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
    	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
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