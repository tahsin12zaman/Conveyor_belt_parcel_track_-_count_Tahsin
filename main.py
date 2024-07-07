import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# Loading my tarined yolo model
model_path = 'my_model.pt'
model = YOLO(model_path)

# Function to do inferencing on any frame
def detect_parcels(frame, model):
    # Perform inference
    results = model(frame)
    detections = results[0].boxes.xyxy  # Extract bounding boxes
    scores = results[0].boxes.conf  # Extract confidence scores

    # Converting detections to list of tuples (x1, y1, x2, y2, score)
    boxes = []
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection
        score = scores[i]
        boxes.append((x1.item(), y1.item(), x2.item(), y2.item(), score.item()))
    return boxes

# Function for calculating the Euclidean distance between two points - - so that no parcel is counted twice
def euclidean_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# Function to process video frames and count parcels
def process_frames(video_path, model, max_distance=50):
    cap = cv2.VideoCapture(video_path)
    parcel_count = 0
    seen_parcels = set()
    tracks = deque(maxlen=100)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detecting parcels in current frame
        detections = detect_parcels(frame, model)

        current_centroids = []
        for detection in detections:
            x1, y1, x2, y2, score = detection
            if score > 0.5:  # Confidence threshold
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                current_centroids.append((centroid, (x1, y1, x2, y2)))

        for centroid, bbox in current_centroids:
            matched = False
            for track in tracks:
                if euclidean_distance(centroid, track['centroid']) < max_distance:
                    track['centroid'] = centroid
                    track['bbox'] = bbox
                    matched = True
                    break
            if not matched:
                tracks.append({'centroid': centroid, 'bbox': bbox, 'id': len(tracks)})

        for track in tracks:
            track_id = track['id']
            bbox = track['bbox']

            if track_id not in seen_parcels:
                seen_parcels.add(track_id)
                parcel_count += 1

            # Draw bounding box and ID
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return parcel_count

# Example video for detection, tracking & counting
video_path = 'sample_vid.mp4'
total_parcel_count = process_frames(video_path, model)
print(f'Total parcels counted: {total_parcel_count}')
