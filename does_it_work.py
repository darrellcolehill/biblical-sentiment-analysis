from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from narrative_story_3 import texts

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")

# Define the emotions
emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise"
]

def analyze_text(text):
    # Tokenize the combined text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Perform emotion detection
    outputs = model(**inputs)
    
    # Get the logits
    logits = outputs.logits
    
    # Apply sigmoid to get probabilities
    probabilities = torch.sigmoid(logits).detach().numpy()[0]
    
    # Ignore the extra logit
    probabilities = probabilities[:27]  # Slice to match the 27 emotions
    
    # Return probabilities for analysis
    return probabilities

# Analyze texts in parallel
with ThreadPoolExecutor() as executor:
    results = list(executor.map(analyze_text, texts))

# Calculate mean and standard deviation for each emotion
mean_probs = np.mean(results, axis=0)
std_probs = np.std(results, axis=0)

# Print results
for i, emotion in enumerate(emotions):
    print(f"{emotion}: Mean = {mean_probs[i]:.4f}, Std Dev = {std_probs[i]:.4f}")
