import speech_recognition as sr

def audio_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio, language='pt-BR')
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            return ""
