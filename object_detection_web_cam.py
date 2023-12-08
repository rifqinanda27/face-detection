import cv2
import numpy as np

def detect_objects():
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getUnconnectedOutLayersNames()

    cap = cv2.VideoCapture(0)

    while True:
        ret, image = cap.read()
        if not ret:
            break

        height, width, __ = image.shape

        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        detections = net.forward(layer_names)

        boxes = []
        confidences = []
        class_ids = []

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[0] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append(([x, y, w, h]))
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indices:
            x, y, w, h = boxes[i]
            class_id = class_ids[i]

            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x +w, y + h), color, 2)

            label = f"{classes[class_id]}: {confidences[i]:.2f}"
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow('Object DETECTION', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    detect_objects()