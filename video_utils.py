from cv2 import imshow, waitKey
import cv2
import serial
import controller
import door
from tracker import person_tracker
import validator


def draw_bbox(video_frame, track_id, bbox, color=(0, 255, 0)):
    x1, y1, x2, y2 = bbox
    cv2.putText(
        video_frame,
        f"ID:{track_id}",
        (int(x1), int(y1) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )
    cv2.rectangle(video_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    return video_frame


def read_video(file, arduino, target_fps=15, target_width=640, target_height=480):
    cap = cv2.VideoCapture(file)

    # Get original FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / target_fps)

    # Initialize variables
    light_on = False
    door_coord = []
    intercepted_ppl = set()
    prev_dict = {}
    inside = [0]  # Use a list to make 'inside' mutable
    tracker = person_tracker("yolov5su")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Reduce resolution
        frame = cv2.resize(frame, (target_width, target_height))

        # Reduce FPS
        if frame_count % frame_interval == 0:
            # if frame_count == 0:
            #     door_coord = door.start(frame)  # do this only once
            #     print(door_coord)
            door_coord = [[623, 284, 770, 514]]

            process_frame(
                frame, inside, intercepted_ppl, prev_dict, tracker, door_coord
            )
            if inside[0] == 0 and light_on:
                controller.turn_relay_off(arduino)
                light_on = False
                print("light_off")

            elif inside[0] > 0 and not light_on:
                controller.turn_relay_on(arduino)
                light_on = True
                print("light_on")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def process_frame(frame, inside, intercepted_ppl, prev_dict, tracker, door_coord):
    people_dict = tracker.detect_frame(frame=frame)
    for person in list(intercepted_ppl):
        if person not in people_dict.keys():
            inside[0] += 1
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
            if tracking_id not in prev_dict.keys():
                inside[0] = max(inside[0] - 1, 0)
            intercepted_ppl.add(tracking_id)

    draw_bbox(frame, "DOOR", door_coord[0], (0, 255, 0))
    cv2.putText(
        frame,
        f"Inside: {inside[0]}",
        (50, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (255, 255, 255),
        2,
    )
    prev_dict.clear()
    prev_dict.update(people_dict)

    # Display the frame
    imshow("Frame", frame)
    if waitKey(1) & 0xFF == ord("q"):
        return


ARDUINO = serial.Serial("COM6", 9600, timeout=1)

read_video(
    "jes.mp4", target_fps=15, target_width=1280, target_height=720, arduino=ARDUINO
)
