from ultralytics import YOLO
import torch

# Set device to CPU to avoid GPU memory issues
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Load the YOLO model
model = YOLO('best.pt')

# Export to ONNX with minimal settings
model.export(format='onnx', imgsz=640, simplify=False, dynamic=False)