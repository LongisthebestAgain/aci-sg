from cv2 import imshow, waitKey
import cv2

import door
from tracker import person_tracker
import validator
import video_utils



intercepted_ppl = set()
prev_dict = {}
inside = 0
# Use the local video file path
video_path = "jessa.mp4"
tracker = person_tracker("yolov5su")
frames = video_utils.read_video(video_path, target_width=1280, target_height=720)
door_coord = door.start(frames[0])
for frame in frames:
    people_dict = tracker.detect_frame(frame=frame)

    for person in list(intercepted_ppl):
        if person not in people_dict.keys():
            inside += 1
            intercepted_ppl.remove(person)

        elif not validator.do_bboxes_intercept(people_dict[person], door_coord[0]):
            intercepted_ppl.remove(person)

    for person in people_dict.items():
        tracking_id, bbox = person
        draw_bbox(
            frame,
            tracking_id,
            bbox,
            color=(0, 0, 255)
            if validator.do_bboxes_intercept(bbox, door_coord[0])
            else (255, 0, 0),
        )
        if validator.do_bboxes_intercept(bbox, door_coord[0]):
            if tracking_id not in prev_dict.keys() and prev_dict:
                inside -= 1

            intercepted_ppl.add(tracking_id)

    prev_dict = people_dict.copy()

    draw_bbox(frame, "DOOR", door_coord[0], (0, 255, 0))
    cv2.putText(
        frame,
        f"Inside: {inside}",
        (20, 20),
        cv2.FONT_HERSHEY_TRIPLEX,
        1,
        (255, 255, 255),
        2,
    )

    imshow("p", frame)
    waitKey(2)
