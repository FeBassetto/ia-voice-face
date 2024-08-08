import ttkbootstrap as ttk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import threading
import cv2
from utils.audio_to_text import audio_to_text
from utils.text_analysis import analyze_text
from utils.face_detection import FaceEmotionDetector
from googletrans import Translator

class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection App")
        self.root.geometry("800x600")
        self.style = ttk.Style("cosmo")

        self.translator = Translator()
        self.recording = False
        self.text = ""
        self.face_emotion_detector = FaceEmotionDetector()

        # Estilização dos Botões
        self.action_button = ttk.Button(root, text="Começar a Gravar", command=self.toggle_action, bootstyle="primary", cursor="hand2")
        self.action_button.pack(pady=20)

        self.progress_label = ttk.Label(root, text="", font=("Helvetica", 12))
        self.progress_label.pack(pady=10)

        self.result_label = ttk.Label(root, text="", wraplength=350, font=("Helvetica", 12))
        self.result_label.pack(pady=20)

        self.camera_label = ttk.Label(root)
        self.camera_label.pack()

        self.cap = cv2.VideoCapture(0)
        self.update_camera()

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            results = self.face_emotion_detector.detect_faces(frame)
            frame = self.face_emotion_detector.draw_faces(frame, results)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
        else:
            img = Image.new('RGB', (640, 480), color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            d.text((10, 10), "Erro ao acessar a câmera", fill=(255, 255, 0))
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)

        self.root.after(10, self.update_camera)

    def toggle_action(self):
        if self.action_button["text"] == "Começar a Gravar":
            self.start_recording()
        elif self.action_button["text"] == "Mostrar Resultado":
            self.show_result()

    def start_recording(self):
        self.action_button.config(text="Gravando...", state="disabled", bootstyle="danger", cursor="watch")
        self.progress_label.config(text="Gravando...")
        self.recording = True
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.start()

    def record_audio(self):
        self.text = audio_to_text()
        self.recording = False
        self.action_button.config(text="Mostrar Resultado", state="normal", bootstyle="success", cursor="hand2")
        self.progress_label.config(text="")

    def show_result(self):
        self.progress_label.config(text="Analisando...")
        self.action_button.config(text="Analisando...", state="disabled", bootstyle="secondary", cursor="watch")
        self.root.update_idletasks()

        if self.text:
            translated_text = self.translator.translate(self.text, src='pt', dest='en').text
            emotions = analyze_text(translated_text)
            emotion_str = '\n'.join([f"{self.translate_emotion(emotion)}: {score}%" for emotion, score in emotions.items()])
            self.result_label.config(text=f"Texto: {self.text}\n\nResultado da Análise:\n{emotion_str}")
        else:
            messagebox.showerror("Erro", "Nenhum áudio foi capturado.")

        self.progress_label.config(text="")
        self.action_button.config(text="Começar a Gravar", state="normal", bootstyle="primary", cursor="hand2")

    def translate_emotion(self, emotion):
        translations = {
            "joy": "Alegria",
            "sadness": "Tristeza",
            "anger": "Raiva",
            "fear": "Medo",
            "surprise": "Surpresa",
            "disgust": "Nojo",
            "neutral": "Neutro"
        }
        return translations.get(emotion, emotion)

    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    root = ttk.Window(themename="cosmo")
    app = EmotionApp(root)
    root.mainloop()
