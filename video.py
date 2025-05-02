from ultralytics import YOLO

# Load a pretrained YOLO11n model
model_name = "yolo11n.pt"
model = YOLO(model_name)

# Define path to video file
source = "path/to/video.mp4"

# Run inference on the source
results = model(source, stream=True, save=True)  # generator of Results objects
