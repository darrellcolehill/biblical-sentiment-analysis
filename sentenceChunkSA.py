import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from nltk.tokenize import sent_tokenize
import pandas as pd 
import os


class SentenceChunkSA:

    def __init__(self, model_name = "joeddav/distilbert-base-uncased-go-emotions-student", selected_emotion_model = "goEmotions"):
        self.model_name = model_name
        self.selected_emotion_model = selected_emotion_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        with open('emotionModels.json', 'r') as file:
            self.all_emotion_models = json.load(file)


    def change_model(self, model_name, emotion_model):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        
        self.selected_emotion_model = emotion_model
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def analyze_text(self, text):
        chunk_data = []

        chunks = sent_tokenize(text)
        
        with ThreadPoolExecutor() as executor:
            chunk_results = list(executor.map(self.analyze_chunk, chunks))
        
        for i, chunk in enumerate(chunks):
            chunk_result = chunk_results[i]
            chunk_data.append({
                "sentence": chunk,
                **{emotion: chunk_result[j] for j, emotion in enumerate(self.all_emotion_models[self.selected_emotion_model])}
            })

        return pd.DataFrame(chunk_data)


    def analyze_chunk(self, chunk_text):

        if self.selected_emotion_model == "goEmotions":
            inputs = self.tokenizer(chunk_text, return_tensors="pt", truncation=True, padding=True)
            outputs = self.model(**inputs)
            # TODO: Find out if I should be using softmax or sigmoid here. 
            probabilities = torch.sigmoid(outputs.logits).detach().numpy()[0]
            # Slice to match the 27 emotions
            probabilities = probabilities[:27]
            return probabilities
        else:
            inputs = self.tokenizer(chunk_text, return_tensors="pt", truncation=True, padding=True)
            outputs = self.model(**inputs)
            
            # Apply softmax to convert logits to probabilities (for 7 emotions)
            probabilities = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
            return probabilities


    def save_to_excel(self, df, filename="emotion_analysis_sentence_chunk.xlsx"):

        if not os.path.exists(f"./results/{self.selected_emotion_model}/{self.model_name}"):
            os.makedirs(f"./results/{self.selected_emotion_model}/{self.model_name}")

        df.to_excel(f"./results/{self.selected_emotion_model}/{self.model_name}/{filename}", index=False)

        statistics_df = self.calculate_summary_statistics(df)

        with pd.ExcelWriter(f"./results/{self.selected_emotion_model}/{self.model_name}/{filename}", mode="a", engine="openpyxl") as writer:
            statistics_df.to_excel(writer, sheet_name="Statistics", index=False)

        print(f"Results saved to './results/{self.selected_emotion_model}/{self.model_name}/{filename}'")


    def calculate_summary_statistics(self, df):
        summary_data = {
            "Emotion": self.all_emotion_models[self.selected_emotion_model],
            "Mean Probability": [],
            "Standard Deviation": []
        }

        for emotion in self.all_emotion_models[self.selected_emotion_model]:
            mean_prob = df[emotion].mean()
            std_prob = df[emotion].std()

            summary_data["Mean Probability"].append(mean_prob)
            summary_data["Standard Deviation"].append(std_prob)

        df_summary = pd.DataFrame(summary_data)
        return df_summary
    


# Example Usage
# analyzer = SentenceChunkSA()
# df_chunks = analyzer.analyze_text(texts[0])
# analyzer.save_to_excel(df_chunks)