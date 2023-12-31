# ECLAIR_Autonomous_Drone
This is a personal repo that contains testing code for YOLO object detection in realtime.

This uses YOLOv8 and potentially YOLOv5, and runs on the NVIDIA Jetson mini GPU. The goal is to create realtime person detection so that we can convert 2d person coordinates to 3d world coordinates.

Much of the code was modified from https://github.com/AnweshCR7/onnx-inference-yolo 

## Yolo with Onnx
To start, navigate to the yolo_with_onnx directory. Run the install_dependencies.sh shell script to install dependencies (hopefully I didn't miss anything). 

Run the command "python3 running_yolov7.py -p models/yolov7-tiny.onnx -c False"

The -p argument is the path to the model file
The -c argument is true if there is a GPU present

Run deep_sparse_benchmark.sh to get a benchmark performance of deep-sparse vs yolov8 on the machine.
DeepSparse code taken from https://neuralmagic.com/blog/yolov8-detection-10x-faster-with-deepsparse-500-fps-on-a-cpu/
