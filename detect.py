import numpy as np
import time
import cv2
import os
from tensorflow.keras.models import load_model

#model = load_model(os.path.sep.join(["E:", "Models", "DeepLearning", "ObjectDetection", "LicensePlate", "recognition_100x100x3.h5"]))
model = load_model("model.h5")
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def resize(image, width=None, height=None):
    if (width is None) & (height is None):
        raise Exception("Height and Width npth are None")
    elif (width is not None) & (height is not None):
        raise Exception("You haved passed npth Height and Width both value")
    elif (width is not None) & (height is None):
        h, w, c = image.shape
        height = int((h / w) * width)
        return cv2.resize(image, (width, height))
    elif (width is None) & (height is not None):
        width = int((w / h) * height)
        return cv2.resize(image, (width, height))


def detect(image, net, ln, Labels, colors, drawBox=True, return_cords=False, minConfi=0.1, thresh=0.3, wide=4, show_text=True):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence >= minConfi:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, minConfi, thresh)

    coords_boxes = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            color = [int(c) for c in colors[classIDs[i]]]

            if drawBox:
                cv2.rectangle(image, (x, y), (x + w, y + h), color, wide)
                if show_text:
                    text = "{}: {:.4f}".format(Labels[classIDs[i]], confidences[i])
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if return_cords:
                coords_boxes.append([Labels[classIDs[i]], confidences[i], x, y, x+w, y+h])

    if drawBox and return_cords:
        return image, coords_boxes
    elif drawBox and not return_cords:
        return image
    elif not drawBox and return_cords:
        return coords_boxes

configPath = "yolov3_tiny_custom.cfg"

weightsPath1 = "licenseplate_detection.weights"
weightsPath2 =  "character_detection.weights"

net1 = cv2.dnn.readNetFromDarknet(configPath, weightsPath1)
net2 = cv2.dnn.readNetFromDarknet(configPath, weightsPath2)

image = cv2.imread("testing/3.jpg")
image = resize(image, width=800)

ln1 = net1.getLayerNames()
ln1 = [ln1[i[0] - 1] for i in net1.getUnconnectedOutLayers()]

ln2 = net2.getLayerNames()
ln2 = [ln2[i[0] - 1] for i in net2.getUnconnectedOutLayers()]

drawed, coords_boxes = detect(image.copy(), net1, ln1, ["Plate"], return_cords=True, colors=[(255, 0, 0)])
for (l, c, x1, y1, x2, y2) in coords_boxes:
    drawed2, bboxes = detect(drawed[y1: y2, x1: x2].copy(), net2, ln2, ["*"], 
        return_cords=True, colors=[(0, 255, 255)], wide=1, show_text=False)
    drawed[y1: y2, x1: x2] = drawed2.copy()
    for (l, c, x21, y21, x22, y22) in bboxes:
        c = cv2.resize(cv2.cvtColor(drawed2[y21: y22, x21: x22], cv2.COLOR_BGR2RGB), (100, 100)).reshape(1, 100, 100, 3) / 255.0
        c = labels[model.predict(c).argmax()]
        #text = "{}: {:.4f}".format(Labels[classIDs[i]], confidences[i])
        cv2.putText(drawed, c, (x1 + x21, y1 + y21 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #print(label, x1)
    #
cv2.imshow("frame", drawed)
cv2.imwrite("img.png", drawed)
cv2.waitKey(0)

