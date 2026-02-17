from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.yaml")  # build a new model from YAML

# Train the model
results = model.train(data="rocket8.yaml", epochs=100, imgsz=640)