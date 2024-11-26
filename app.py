from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
from utils.audio_to_text import audio_to_text
from utils.text_analysis import analyze_text
from utils.face_detection import FaceEmotionDetector
from googletrans import Translator

app = Flask(__name__)

# Try to initialize the camera
try:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise ValueError("Camera not accessible")
    webcam_available = True
except Exception as e:
    print(f"Webcam not available: {e}")
    camera = None
    webcam_available = False

face_emotion_detector = FaceEmotionDetector() if webcam_available else None
translator = Translator()
recording = False
text_result = ""
emotions_result = {}

def gen_frames():
    while True:
        if not webcam_available:
            break
        success, frame = camera.read()
        if not success:
            break
        else:
            results = face_emotion_detector.detect_faces(frame)
            frame = face_emotion_detector.draw_faces(frame, results)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', webcam_available=webcam_available)

@app.route('/video_feed')
def video_feed():
    if not webcam_available:
        return "Webcam not available", 404
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, text_result, emotions_result
    recording = True
    text_result = ""
    emotions_result = {}
    thread = threading.Thread(target=record_audio)
    thread.start()
    return jsonify({"status": "Recording started"})

def record_audio():
    global recording, text_result, emotions_result
    text_result = audio_to_text()
    if text_result:
        translated_text = translator.translate(text_result, src='pt', dest='en').text
        emotions = analyze_text(translated_text)
        emotions_result = {translate_emotion(emotion): f"{score}%" for emotion, score in emotions.items()}
    else:
        emotions_result = {}
    recording = False

@app.route('/get_result', methods=['GET'])
def get_result():
    if not recording:
        if text_result:
            return jsonify({"text": text_result, "emotions": emotions_result})
        else:
            return jsonify({"error": "Nenhum áudio foi capturado."}), 400
    else:
        return jsonify({"status": "Recording in progress"}), 202

def translate_emotion(emotion):
    translations = {
        "anger": "Raiva",
        "disgust": "Nojo",
        "fear": "Medo",
        "joy": "Feliz",
        "sadness": "Triste",
        "surprise": "Surpreso",
        "neutral": "Neutro",
        "happy": "Feliz",
        "sad": "Triste",
        "angry": "Raiva",
    }
    return translations.get(emotion.lower(), emotion)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4444)
