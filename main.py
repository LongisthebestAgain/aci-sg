import cv2
from sympy import ring
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# Define the door regions (x, y, w, h) for each room
door_regions = {
    "Room1": (100, 200, 50, 200),  # Example region for Room 1 (x, y, w, h)
    "Room2": (300, 200, 50, 200),  # Example region for Room 2 (x, y, w, h)
}

room_people_count = {
    "Room1": 0,
    "Room2": 0,
}


# Draw door regions on the frame
def draw_door_regions(frame, door_regions):
    for room, region in door_regions.items():
        x, y, w, h = region
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, room, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )
    return frame


# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
model.classes = [0]  # Class 0 is 'person'


def detect_people(frame):
    results = model(frame)
    people_detections = []
    for det in results.xyxy[0].numpy():
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:
            people_detections.append((int(x1), int(y1), int(x2), int(y2)))
    return people_detections


# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30)


def track_people(frame, detections):
    bbox_xywh = []
    confs = []

    for det in detections:
        x1, y1, x2, y2 = det
        w, h = x2 - x1, y2 - y1
        bbox_xywh.append([x1, y1, w, h])
        confs.append(1.0)

    outputs = tracker.update_tracks(bbox_xywh, confs, frame=frame)

    tracked_people = []
    for track in outputs:
        track_id = track.track_id
        x1, y1, x2, y2 = track.to_ltrb()
        tracked_people.append((track_id, x1, y1, x2, y2))

    return tracked_people


# Update room count based on tracked people
def update_room_count(tracked_people, door_regions, room_people_count):
    for track_id, x1, y1, x2, y2 in tracked_people:
        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2

        for room, region in door_regions.items():
            door_x, door_y, door_w, door_h = region
            if (
                door_x < person_center_x < door_x + door_w
                and door_y < person_center_y < door_y + door_h
            ):
                if person_center_y > door_y + door_h // 2:
                    room_people_count[room] += 1
                else:
                    room_people_count[room] -= 1

    return room_people_count


# Main video processing loop
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        people_detections = detect_people(frame)
        tracked_people = track_people(frame, people_detections)
        updated_count = update_room_count(
            tracked_people, door_regions, room_people_count
        )

        frame = draw_door_regions(frame, door_regions)
        for room, count in updated_count.items():
            cv2.putText(
                frame,
                f"{room}: {count}",
                (10, 50 + 30 * list(door_regions.keys()).index(room)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

        cv2.imshow("Room Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run the video processing on a video file or webcam
if __name__ == "__main__":
    process_video(0)  # Replace with your video file or 0 for webcam
