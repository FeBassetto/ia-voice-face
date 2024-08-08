import cv2
from deepface import DeepFace
import time

class FaceEmotionDetector:
    def __init__(self):
        self.detector = DeepFace.build_model("Emotion")
        self.last_detection_time = 0
        self.detection_interval = 2  # Intervalo de 2 segundos

    def detect_faces(self, frame):
        current_time = time.time()
        if current_time - self.last_detection_time >= self.detection_interval:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            detections = []
            for result in results:
                bbox = result["region"]
                emotion = result["dominant_emotion"]
                detections.append((bbox, emotion))
            self.last_detection_time = current_time
            self.last_detections = detections
        else:
            detections = self.last_detections if hasattr(self, 'last_detections') else []
        return detections

    def draw_faces(self, frame, results):
        for (box, emotion) in results:
            (x, y, w, h) = box['x'], box['y'], box['w'], box['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            translated_emotion = self.translate_emotion(emotion)
            cv2.putText(frame, translated_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        return frame

    def translate_emotion(self, emotion):
        translations = {
            "angry": "Raiva",
            "disgust": "Nojo",
            "fear": "Medo",
            "happy": "Feliz",
            "sad": "Triste",
            "surprise": "Surpreso",
            "neutral": "Neutro"
        }
        return translations.get(emotion, emotion)
