import cv2
import mediapipe as mp

class FaceEmotionDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.2)

    def detect_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        return results

    def draw_faces(self, frame, results):
        if results.detections:
            for detection in results.detections:
                self.mp_drawing.draw_detection(frame, detection)
        return frame
