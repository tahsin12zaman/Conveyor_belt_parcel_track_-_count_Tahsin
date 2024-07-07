
from ultralytics import YOLO

# Load a model and specify device ('cpu' or 'cuda')
model = YOLO('yolov8n.pt', device='cuda')  # Specify 'cuda' for GPU usage

# Train the model with custom checkpoint saving period
model.train(
    data='C:/Users/HP/PycharmProjects/conv_belt_parcel/dataset/data.yaml',
    epochs=50,
    imgsz=640,
    project='C:/Users/HP/PycharmProjects/conv_belt_parcel/runs',
    name='experiment1',
    save_period=5  # Save a checkpoint every 5 epochs
)


