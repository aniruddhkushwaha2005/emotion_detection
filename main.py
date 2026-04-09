import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
from collections import deque

app = Flask(__name__)


socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


model = load_model('fer2013_mini_XCEPTION.hdf5', compile=False)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


recent_emotions = deque(maxlen=6)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('process_image')
def handle_image(data):
    
    try:
       
        header, encoded = data.split(",", 1)
        decoded = base64.b64decode(encoded)
        img_np = np.frombuffer(decoded, dtype=np.uint8)
        frame = cv2.imdecode(img_np, flags=cv2.IMREAD_COLOR)

      
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        response_data = {
            "faces": [],
            "stats": {
                "emotion": "Scanning...",
                "confidence": 0.0,
                "faces_count": len(faces)
            }
        }
        
        max_confidence = 0.0
        dominant_emotion = "No Face Detected" if len(faces) == 0 else "Scanning..."
        
        for (x, y, w, h) in faces:
            
            face_roi = gray[y:y+h, x:x+w]
            
            emotion_str = "Unknown"
            conf_val = 0.0
            
            
            if face_roi.shape[0] >= 48 and face_roi.shape[1] >= 48:
                
                
                face_roi = cv2.equalizeHist(face_roi)
                
            
                face = cv2.resize(face_roi, (48, 48))
                face = face / 255.0
                
                
                face_flipped = cv2.flip(face, 1)
                
               
                batch = np.vstack([
                    np.reshape(face, (1, 48, 48, 1)),
                    np.reshape(face_flipped, (1, 48, 48, 1))
                ])

              
                predictions = model.predict(batch, verbose=0)
                
                
                final_pred = np.mean(predictions, axis=0)
                
                
                global recent_emotions
                recent_emotions.append(final_pred)
                avg_pred = np.mean(recent_emotions, axis=0)
                
                
                max_idx = np.argmax(avg_pred)
                emotion_str = emotion_labels[max_idx]
                conf_val = float(avg_pred[max_idx])
                
                if conf_val > max_confidence:
                    max_confidence = conf_val
                    dominant_emotion = emotion_str
                    
            
            response_data["faces"].append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "emotion": emotion_str
            })
            
        response_data["stats"]["emotion"] = dominant_emotion
        response_data["stats"]["confidence"] = max_confidence
        
       
        emit('result', response_data)
        
    except Exception as e:
        print("Image processing error:", e)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)