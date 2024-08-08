from transformers import pipeline

def analyze_text(text):
    emotion_analysis = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    result = emotion_analysis(text)

    emotions = {item['label']: round(item['score'] * 100, 2) for item in result[0]}
    return emotions
