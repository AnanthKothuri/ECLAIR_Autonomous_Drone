# Install packages for DeepSparse and YOLOv8
pip install deepsparse[yolov8] ultralytics

# Export YOLOv8n and YOLOv8s ONNX models
yolo task=detect mode=export model=yolov8n.pt format=onnx opset=13
yolo task=detect mode=export model=yolov8s.pt format=onnx opset=13

# Benchmark with DeepSparse!
deepsparse.benchmark yolov8n.onnx --scenario=sync --input_shapes="[1,3,640,640]"

deepsparse.benchmark yolov8s.onnx --scenario=sync --input_shapes="[1,3,640,640]"
