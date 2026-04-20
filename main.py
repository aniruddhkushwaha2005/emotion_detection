import base64
import os
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Load ONNX model once at startup.
ort_session = ort.InferenceSession("fer2013_mini_XCEPTION.onnx")
input_name = ort_session.get_inputs()[0].name
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Keep short rolling history for temporal smoothing.
recent_emotions = deque(maxlen=6)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("process_image")
def handle_image(data):
    try:
        _, encoded = data.split(",", 1)
        decoded = base64.b64decode(encoded)
        img_np = np.frombuffer(decoded, dtype=np.uint8)
        frame = cv2.imdecode(img_np, flags=cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        response_data = {
            "faces": [],
            "stats": {
                "emotion": "Scanning...",
                "confidence": 0.0,
                "faces_count": len(faces),
            },
        }

        max_confidence = 0.0
        dominant_emotion = "No Face Detected" if len(faces) == 0 else "Scanning..."

        for (x, y, w, h) in faces:
            face_roi = gray[y : y + h, x : x + w]
            emotion_str = "Unknown"
            conf_val = 0.0

            if face_roi.shape[0] >= 48 and face_roi.shape[1] >= 48:
                face_roi = cv2.equalizeHist(face_roi)
                face = cv2.resize(face_roi, (48, 48))
                face = face / 255.0

                face_flipped = cv2.flip(face, 1)
                batch = np.vstack(
                    [
                        np.reshape(face, (1, 48, 48, 1)),
                        np.reshape(face_flipped, (1, 48, 48, 1)),
                    ]
                ).astype(np.float32)

                predictions = ort_session.run(None, {input_name: batch})[0]
                final_pred = np.mean(predictions, axis=0)

                recent_emotions.append(final_pred)
                avg_pred = np.mean(recent_emotions, axis=0)

                max_idx = int(np.argmax(avg_pred))
                emotion_str = emotion_labels[max_idx]
                conf_val = float(avg_pred[max_idx])

                if conf_val > max_confidence:
                    max_confidence = conf_val
                    dominant_emotion = emotion_str

            response_data["faces"].append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "emotion": emotion_str,
                }
            )

        response_data["stats"]["emotion"] = dominant_emotion
        response_data["stats"]["confidence"] = max_confidence

        emit("result", response_data)

    except Exception as e:
        print("Image processing error:", e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    socketio.run(app, host="0.0.0.0", port=port, debug=False)