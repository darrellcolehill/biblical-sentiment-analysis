import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from nltk.tokenize import sent_tokenize

class SlidingWindowSA:

    def __init__(self, model_name = "joeddav/distilbert-base-uncased-go-emotions-student"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.emotions = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
            "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise"
        ]


    def change_model(self, model_name):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)


    def analyze_text(self, text):
        chunks = sent_tokenize(text)
        chunk_data = {
            "Chunk": [],
            **{emotion: [] for emotion in self.emotions},
        }

        for i in range(len(chunks)):
            if i < len(chunks) - 1:
                sentence = chunks[i] + " " + chunks[i + 1]
            else:
                break

            probabilities = self.analyze_chunk(sentence)
            chunk_data["Chunk"].append(sentence)
            for j in range(len(probabilities)):
                chunk_data[self.emotions[j]].append(probabilities[j])

        return pd.DataFrame(chunk_data)


    def analyze_chunk(self, chunk_text):
        inputs = self.tokenizer(chunk_text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Apply sigmoid to get probabilities and slice to match the 27 emotions
        probabilities = torch.sigmoid(logits).detach().numpy()[0][:27]
        return probabilities


    def save_to_excel(self, df, filename="emotion_analysis_sliding_window.xlsx"):

        if not os.path.exists(f"./results/{self.model_name}"):
            os.makedirs(f"./results/{self.model_name}")

        df.to_excel(f"./results/{self.model_name}/{filename}", index=False)

        statistics_df = self.calculate_summary_statistics(df)

        with pd.ExcelWriter(f"./results/{self.model_name}/{filename}", mode="a", engine="openpyxl") as writer:
            statistics_df.to_excel(writer, sheet_name="Statistics", index=False)
        print(f"Results saved to './results/{self.model_name}/{filename}'")

        
    def calculate_summary_statistics(self, df_chunks):
        summary_data = {
            "Emotion": self.emotions,
            "Mean Probability": [],
            "Standard Deviation": []
        }

        for emotion in self.emotions:
            mean_prob = df_chunks[emotion].mean()
            std_prob = df_chunks[emotion].std()

            summary_data["Mean Probability"].append(mean_prob)
            summary_data["Standard Deviation"].append(std_prob)

        df_summary = pd.DataFrame(summary_data)
        return df_summary


# Example Usage
# analyzer = SlidingWindowSA()
# df_chunks = analyzer.analyze_text(fathers_pov)
# analyzer.save_to_excel(df_chunks)