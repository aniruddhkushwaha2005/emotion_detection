# Emotion Detector Pro - Sci-Fi HUD 🤖👁️

A real-time facial expression and emotion recognition web application built with **Flask**, **WebSockets**, and **TensorFlow Xception**. This project features a stunning, fully responsive Sci-Fi "HUD" (Heads-Up Display) interface inspired by cyberpunk aesthetics.

## ✨ Features
*   **Real-Time Native Feed**: Securely captures the user's native webcam directly from their browser (Phone or PC) via `navigator.mediaDevices`.
*   **Ultra-Low Latency**: Uses `Flask-SocketIO` to establish a two-way tunnel. The browser sends hyper-compressed frames to the backend, which processes them instantly and returns target coordinates to draw HUD graphics natively on an HTML5 Canvas!
*   **Advanced AI Enhancements**: 
    *   **Histogram Equalization**: Auto-balances lighting on the face to recognize micro-expressions in the dark!
    *   **Test-Time Augmentation (TTA)**: Evaluates normal and horizontally-flipped frames simultaneously to dramatically boost stability against facial rotation.
    *   **Temporal Smoothing**: Employs an Exponential Moving Average across the last 6 frames to stop flickering and guarantee a liquid-smooth emotion read.
*   **Responsive UI**: A fully dynamic dark-mode HUD with glowing neon alerts, scanline animations, and glassmorphism. Completely usable on Mobile and Tablets.

## 🛠️ Technology Stack
*   **Backend**: Python, Flask, Flask-SocketIO
*   **Machine Learning**: TensorFlow (Keras `fer2013_mini_XCEPTION.hdf5`), OpenCV (Haar Cascades for blistering fast Face Tracking)
*   **Frontend**: HTML5 Video, HTML5 Canvas, Vanilla Javascript, CSS3

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/emotion-detector.git
   cd emotion-detector
   ```

2. **Install the dependencies:**
   Make sure you are using Python 3.10+ and install requirements (NumPy < 2 is strictly required for OpenCV and TensorFlow compatibility).
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server:**
   ```bash
   python main.py
   ```

4. **View the Application:**
   Open `http://localhost:5000` in your browser.

> [!TIP]
> **Mobile Testing**: To view this natively on your phone over the local Wifi, Chrome restricts Camera access on standard `http://`. We recommend using a secure tunnel like `localtunnel` (Run `npx localtunnel --port 5000`) and opening the public HTTPS link on your phone!

## 🧠 The Emotion Model
The project currently detects 7 expressions: Happy, Sad, Surprise, Fear, Disgust, Angry, and Neutral.

---
*Created with ❤️ for exploring Neural Networks and next-gen UI design.*
