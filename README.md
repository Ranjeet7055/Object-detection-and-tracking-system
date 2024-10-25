# Object-detection-and-tracking-system
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Object Detection and Tracking</title>
    <style>
        #video {
            width: 640px;
            height: 480px;
        }
        #canvas {
            position: absolute;
            top: 0;
            left: 0;
        }
    </style>
</head>
<body>
    <video id="video" autoplay></video>
    <canvas id="canvas" width="640" height="480"></canvas>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const context = canvas.getContext('2d');

        // Get access to the camera
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
                video.srcObject = stream;
                video.play();
            });
        }

        video.addEventListener('play', () => {
            setInterval(async () => {
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const frame = canvas.toDataURL('image/jpeg');
                
                // Send the frame to the backend for processing
                const response = await fetch('http://localhost:5000/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ image: frame })
                });
                const result = await response.json();

                // Draw the results on the canvas
                context.clearRect(0, 0, canvas.width, canvas.height);
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                result.objects.forEach(obj => {
                    context.strokeStyle = 'red';
                    context.lineWidth = 2;
                    context.strokeRect(obj.x, obj.y, obj.width, obj.height);
                    context.fillText(obj.label, obj.x, obj.y - 10);
                });
            }, 100); // Process frame every 100ms
        });
    </script>
</body>
</html>


# App.py
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = open("coco.names").read().strip().split("\n")

def detect_objects(image):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    result = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        label = str(classes[class_ids[i]])
        result.append({"x": x, "y": y, "width": w, "height": h, "label": label})
    return result

@app.route('/process', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = base64.b64decode(data['image'].split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    objects = detect_objects(image)
    return jsonify({"objects": objects})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
