import cv2

# Initialize the list to store box coordinates
boxes = []
drawing = False
x1, y1 = None, None


def draw_rectangle(frame, event, x, y, flags, param):
    global drawing, x1, y1, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = frame.copy()
            cv2.rectangle(img_copy, (x1, y1), (x, y), (0, 255, 0), 2)
            cv2.imshow("Frame", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        boxes.append([x1, y1, x, y])
        cv2.rectangle(frame, (x1, y1), (x, y), (0, 255, 0), 2)
        cv2.imshow("Frame", frame)


def start(frame):
    # Load the video
    cv2.imshow("Frame", frame)
    cv2.setMouseCallback(
        "Frame",
        lambda event, x, y, flags, param: draw_rectangle(
            frame, event, x, y, flags, param
        ),
    )

    print("Draw boxes on the frame. Press 'q' to quit and see the coordinates.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    return boxes
