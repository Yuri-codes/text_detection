from flask import Flask, jsonify, request
import cv2

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    image = cv2.imread('images/photo1.jpg')
    orig = image.copy()
    (H, W) = image.shape[:2]

    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    net = cv2.dnn.readNet('frozen_east_text_detection.pb')

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue
            confidences.append(scoresData[x])

    if confidences == []:
        return jsonify({'result':'good'}), 201
    else:
        return jsonify({'result':'bad'}), 201

if __name__ == '__main__':
    app.run(debug=True)
