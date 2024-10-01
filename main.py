from cv2 import imshow, waitKey
from ultralytics import YOLO

import door
from tracker import person_tracker
import video_utils

# Use the local video file path
video_path = "855564-hd_1920_1080_24fps.mp4"
tracker = person_tracker("yolov5su")
frames = video_utils.read_video(video_path)
door_coord = door.start(frames[0])
for frame in frames:
    person_dict = tracker.detect_frame(frame=frame)
    person_dict["door"] = door_coord[0]
    print(person_dict)
    frame_out = tracker.draw_bbox(frame, person_dict)
    imshow("p", frame_out)
    waitKey(2)
