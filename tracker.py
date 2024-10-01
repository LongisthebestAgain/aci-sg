import cv2
import torch
from ultralytics import YOLO
import video_utils


class person_tracker:
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )[0]
        cls_ids_dict = results.names
        person_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_class_id = box.cls.tolist()[0]
            obj_class_name = cls_ids_dict[object_class_id]

            if obj_class_name == "person":
                person_dict[track_id] = result

        return person_dict

    def draw_bbox(self, video_frame, player_detection):
        for track_id, bbox in player_detection.items():
            x1, y1, x2, y2 = bbox
            cv2.putText(
                video_frame,
                f"ID:{track_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_TRIPLEX,
                2,
                (0, 255, 0),
                2,
            )
            cv2.rectangle(
                video_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
            )

        return video_frame

    def live(self, frames):
        for frame in frames:
            detection = self.detect_frame(frame)
            output_frame = self.draw_bbox(frame, detection)
            cv2.imshow("r", output_frame)
            cv2.waitKey(1)
