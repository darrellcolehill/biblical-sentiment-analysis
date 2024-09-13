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


def analyze_chunk(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    outputs = model(**inputs)
    
    probabilities = torch.sigmoid(outputs.logits).detach().numpy()[0]
    
    # Slice to match the 27 emotions
    probabilities = probabilities[:27]
    
    return probabilities


all_chunk_probabilities = []
chunk_results_store = []

def analyze_text_chunks(text):
    chunks = sent_tokenize(text)
    
    with ThreadPoolExecutor() as executor:
        chunk_results = list(executor.map(analyze_chunk, chunks))
    
    for i, chunk in enumerate(chunks):
        chunk_result = chunk_results[i]
        chunk_results_store.append({
            "sentence": chunk,
            **{emotion: chunk_result[j] for j, emotion in enumerate(emotions)}
        })



# Analyze all texts
for text in texts:
    analyze_text_chunks(text)

# Convert results into a DataFrame
df = pd.DataFrame(chunk_results_store)

# Save the DataFrame to an Excel file
excel_file = "sentiment_analysis_results.xlsx"
df.to_excel(excel_file, index=False)

print(f"Results saved to {excel_file}")

all_chunk_probabilities = np.array([list(result.values())[1:] for result in chunk_results_store])

mean_probs = np.mean(all_chunk_probabilities, axis=0)
std_probs = np.std(all_chunk_probabilities, axis=0)

statistics_df = pd.DataFrame({
    "Emotion": emotions,
    "Mean": mean_probs,
    "Standard Deviation": std_probs
})

with pd.ExcelWriter(excel_file, mode="a", engine="openpyxl") as writer:
    statistics_df.to_excel(writer, sheet_name="Statistics", index=False)

print(f"Mean and Standard Deviation saved to the 'Statistics' sheet in {excel_file}")
