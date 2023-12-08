import cv2
import numpy as np

def detect_objects(image_path):
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getUnconnectedOutLayersNames()

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    detections = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids =[]

    for detections in detections:
        for obj in detections:
            scores = obj[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        label = f"{classes[class_id]}: {confidences[i]:.2f}"
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    image_path = "dog_bike_car.jpg"
    detect_objects(image_path)