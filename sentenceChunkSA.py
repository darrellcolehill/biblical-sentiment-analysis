from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from nltk.tokenize import sent_tokenize
import pandas as pd 
from narrative_story_1 import texts

tokenizer = AutoTokenizer.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")

emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise"
]


def analyze_text(text):
    chunk_data = []

    chunks = sent_tokenize(text)
    
    with ThreadPoolExecutor() as executor:
        chunk_results = list(executor.map(analyze_chunk, chunks))
    
    for i, chunk in enumerate(chunks):
        chunk_result = chunk_results[i]
        chunk_data.append({
            "sentence": chunk,
            **{emotion: chunk_result[j] for j, emotion in enumerate(emotions)}
        })

    return pd.DataFrame(chunk_data)


def analyze_chunk(chunk_text):
    inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs.logits).detach().numpy()[0]
    # Slice to match the 27 emotions
    probabilities = probabilities[:27]
    return probabilities


def save_to_excel(df, filename="emotion_analysis_sentence_chunk.xlsx"):
    df.to_excel(filename, index=False)

    statistics_df = calculate_summary_statistics(df)

    with pd.ExcelWriter(filename, mode="a", engine="openpyxl") as writer:
        statistics_df.to_excel(writer, sheet_name="Statistics", index=False)
    print(f"Results saved to '{filename}'")


def calculate_summary_statistics(df):
    summary_data = {
        "Emotion": emotions,
        "Mean Probability": [],
        "Standard Deviation": []
    }

    for emotion in emotions:
        mean_prob = df[emotion].mean()
        std_prob = df[emotion].std()

        summary_data["Mean Probability"].append(mean_prob)
        summary_data["Standard Deviation"].append(std_prob)

    df_summary = pd.DataFrame(summary_data)
    return df_summary


data = analyze_text(texts[0])
save_to_excel(data)