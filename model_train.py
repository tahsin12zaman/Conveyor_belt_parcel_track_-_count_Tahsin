from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')

# Set the device to 'cuda' for GPU usage
model.to('cuda')

# Train the model with custom checkpoint saving period
model.train(
    data='/home/kow-ai-3/Documents/Office_Work/Dataset/dfew/Conveyor_belt_parcel_track_-_count_Tahsin/dataset/data.yaml',
    epochs=50,
    imgsz=640,
    project='runs',
    name='experiment1',
    save_period=5  # Save a checkpoint every 5 epochs
)
