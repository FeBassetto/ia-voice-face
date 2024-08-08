import speech_recognition as sr

def audio_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Diga algo:")
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio, language='pt-BR')
            print("Você disse: " + text)
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition não conseguiu entender o áudio")
            return ""
        except sr.RequestError as e:
            print(f"Erro ao requisitar resultados do serviço Google Speech Recognition; {e}")
            return ""
