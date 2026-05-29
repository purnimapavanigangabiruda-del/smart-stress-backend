from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import time
from collections import deque
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

latest_frame = None

model = load_model("best_emotion_model.keras")

emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

os.makedirs("captured_images", exist_ok=True)

green_signal = deque(maxlen=180)
motion_signal = deque(maxlen=180)
time_signal = deque(maxlen=180)
blink_times = deque(maxlen=100)

frame_count = 0
last_emotion = "Detecting"
previous_eye_detected = True

current_data = {
    "emotion": "Detecting",
    "heart_rate": 0,
    "respiratory_rate": 0,
    "blink_rate": 0,
    "blood_pressure": "Estimating"
}


def estimate_rate(signal_values, time_values, min_bpm, max_bpm):
    if len(signal_values) < 80:
        return 0

    signal = np.array(signal_values)
    times = np.array(time_values)

    duration = times[-1] - times[0]

    if duration <= 5:
        return 0

    fps = len(signal) / duration
    signal = signal - np.mean(signal)

    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)

    bpm = freqs * 60
    valid = (bpm >= min_bpm) & (bpm <= max_bpm)

    if not np.any(valid):
        return 0

    return int(bpm[valid][np.argmax(fft[valid])])


def estimate_blood_pressure(heart_rate):
    if heart_rate == 0:
        return "Estimating"

    if heart_rate < 60:
        return "105/70 mmHg"
    elif heart_rate <= 80:
        return "115/75 mmHg"
    elif heart_rate <= 100:
        return "120/80 mmHg"
    else:
        return "130/85 mmHg"


def generate_frames():
    global latest_frame, frame_count, last_emotion, previous_eye_detected, current_data

    while True:
        success, frame = camera.read()

        if not success:
            break

        latest_frame = frame.copy()
        current_time = time.time()
        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(120, 120)
        )

        emotion = last_emotion

        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]

            face_color = frame[y:y+h, x:x+w]

            if frame_count % 5 == 0:
                emotion_img = cv2.resize(face_color, (96, 96))
                emotion_img = emotion_img / 255.0
                emotion_img = np.reshape(emotion_img, (1, 96, 96, 3))

                prediction = model.predict(emotion_img, verbose=0)
                emotion_index = np.argmax(prediction)
                last_emotion = emotions[emotion_index]

            emotion = last_emotion

            forehead = frame[
                y + int(0.10 * h): y + int(0.30 * h),
                x + int(0.30 * w): x + int(0.70 * w)
            ]

            if forehead.size > 0:
                green_signal.append(np.mean(forehead[:, :, 1]))
                motion_signal.append(y + h / 2)
                time_signal.append(current_time)

            face_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(25, 25)
            )

            if len(eyes) == 0:
                if previous_eye_detected:
                    blink_times.append(current_time)
                    previous_eye_detected = False
            else:
                previous_eye_detected = True

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)

        heart_rate = estimate_rate(green_signal, time_signal, 50, 120)
        respiratory_rate = estimate_rate(motion_signal, time_signal, 8, 30)
        blink_rate = len([t for t in blink_times if current_time - t <= 60])
        blood_pressure = estimate_blood_pressure(heart_rate)

        current_data["emotion"] = emotion
        current_data["heart_rate"] = heart_rate
        current_data["respiratory_rate"] = respiratory_rate
        current_data["blink_rate"] = blink_rate
        current_data["blood_pressure"] = blood_pressure

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame +
            b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/data")
def data():
    return jsonify(current_data)


@app.route("/capture")
def capture():
    global latest_frame

    if latest_frame is not None:
        cv2.imwrite("captured_images/captured.jpg", latest_frame)
        return "Image Captured Successfully"

    return "No Frame Available"


@app.route("/questionnaire")
def questionnaire():
    return """
    <h1 style='font-family:sans-serif;text-align:center;margin-top:100px;'>
    Questionnaire Module Page
    </h1>
    """


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)