import cv2


def read_video(file):
    frames = []
    cap = cv2.VideoCapture(file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
    cap.release()

    return frames