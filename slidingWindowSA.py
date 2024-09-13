import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from nltk.tokenize import sent_tokenize

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")

# Define the emotions (GoEmotions has 27 labels, ignoring the extra class)
emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise"
]

sons_pov = """I asked my father for my share of the inheritance and left to live freely, but I wasted everything. When a famine hit, I was left hungry and desperate. I decided to return home, ashamed, and beg my father to take me back as a servant. To my surprise, my father welcomed me with love, embraced me, and celebrated my return, even though I didn't deserve it."""
sons_pov_longer = """I was tired of living under my father's roof, so I demanded my share of the inheritance. I wanted freedom and control over my own life. My father granted my request and gave me my share of the estate. I took everything and went to a distant land, where I lived extravagantly, spending recklessly on pleasures. It wasn't long before I had nothing left. Then, a severe famine struck, and I found myself penniless and hungry. Out of desperation, I found a job feeding pigs, but I was so hungry that I longed to eat the pig’s food. That’s when it hit me—my father's servants had more than enough to eat, yet here I was starving. I decided to go back to my father, prepared to beg him to take me in, not as his son, but as a servant. I rehearsed my apology the entire way. But as I approached, I saw my father running towards me. Before I could even finish my confession, he embraced me and kissed me. I was overwhelmed when he called for a celebration, dressing me in fine clothes and preparing a feast. I had expected judgment but received forgiveness, love, and joy instead. I felt undeserving, but my father welcomed me back with open arms."""

fathers_pov = """My younger son asked for his inheritance, left, and lost everything. Yet, when he came back, I saw him from afar and felt overwhelming compassion. I ran to him, embraced him, and threw a celebration because he was lost and now has returned, alive. My older son was upset, but I had to explain that the return of his brother, who was once lost, is a moment of joy for all of us."""
fathers_pov_longer = """One day, my younger son came to me and demanded his share of the inheritance. It broke my heart, but I gave him what he asked for, knowing I couldn’t stop him from making his own choices. He left, and I waited, hoping that one day he would return. I heard stories about his reckless living and the hardships he faced. My heart ached for him, but I couldn't force him back. Then one day, while I was looking out toward the horizon, I saw him coming home. My heart leapt with compassion. I ran to him without hesitation. He began to apologize, but I didn't care about his past mistakes. I was just overjoyed to have him back. I told my servants to prepare a feast, clothe him in the best robe, and put a ring on his finger. My son, who I thought was lost forever, had returned. He was dead to the world, but now he was alive again. We had to celebrate. Yet, my other son felt hurt and angry. I reminded him that everything I have is already his, but the return of his brother was a special moment—he was lost and now found. My love for both my sons remains strong, but this celebration was necessary for the joy of reunion and forgiveness."""

brothers_pov = """I’ve always obeyed my father and worked hard, yet he never threw a celebration for me. When my younger brother wasted everything and came back, my father immediately welcomed him and threw a feast. I was angry and felt it was unfair. My father reminded me that I’ve always had his love, but we should celebrate because my brother, who was lost, has come back home."""


legacy_standard_bible = """
Jesus continued: “There was a man who had two sons. The younger one said to his father, ‘Father, give me my share of the estate.’ So he divided his property between them.
“Not long after that, the younger son got together all he had, set off for a distant country and there squandered his wealth in wild living. After he had spent everything, there was a severe famine in that whole country, and he began to be in need. So he went and hired himself out to a citizen of that country, who sent him to his fields to feed pigs. He longed to fill his stomach with the pods that the pigs were eating, but no one gave him anything. “When he came to his senses, he said, ‘How many of my father’s hired servants have food to spare, and here I am starving to death!  I will set out and go back to my father and say to him: Father, I have sinned against heaven and against you. I am no longer worthy to be called your son; make me like one of your hired servants.’ So he got up and went to his father.
“But while he was still a long way off, his father saw him and was filled with compassion for him; he ran to his son, threw his arms around him and kissed him.
 “The son said to him, ‘Father, I have sinned against heaven and against you. I am no longer worthy to be called your son.’
 “But the father said to his servants, ‘Quick! Bring the best robe and put it on him. Put a ring on his finger and sandals on his feet.  Bring the fattened calf and kill it. Let’s have a feast and celebrate.  For this son of mine was dead and is alive again; he was lost and is found.’ So they began to celebrate.
 “Meanwhile, the older son was in the field. When he came near the house, he heard music and dancing.  So he called one of the servants and asked him what was going on.  ‘Your brother has come,’ he replied, ‘and your father has killed the fattened calf because he has him back safe and sound.’
 “The older brother became angry and refused to go in. So his father went out and pleaded with him.  But he answered his father, ‘Look! All these years I’ve been slaving for you and never disobeyed your orders. Yet you never gave me even a young goat so I could celebrate with my friends. But when this son of yours who has squandered your property with prostitutes comes home, you kill the fattened calf for him!’
 “‘My son,’ the father said, ‘you are always with me, and everything I have is yours. 32 But we had to celebrate and be glad, because this brother of yours was dead and is alive again; he was lost and is found.’”
"""



# Function to analyze a sentence chunk
def analyze_chunk(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.sigmoid(logits).detach().numpy()[0][:27]  # Slice to match the 27 emotions
    return probabilities

# Function to process chunks of text
def process_chunks(text):
    chunks = sent_tokenize(text)

    # Create a DataFrame to store the results for each chunk
    chunk_data = {
        "Chunk": [],
        **{emotion: [] for emotion in emotions},
    }

    for i in range(len(chunks)):
        if i < len(chunks) - 1:
            sentence = chunks[i] + " " + chunks[i + 1]
        else:
            break

        probabilities = analyze_chunk(sentence)

        # Store the chunk text and corresponding probabilities in the chunk_data dictionary
        chunk_data["Chunk"].append(sentence)
        for j in range(len(probabilities)):
            chunk_data[emotions[j]].append(probabilities[j])

    df_chunks = pd.DataFrame(chunk_data)
    return df_chunks


# Function to calculate mean and standard deviation from the DataFrame
def calculate_summary_statistics(df_chunks):
    summary_data = {
        "Emotion": emotions,
        "Mean Probability": [],
        "Standard Deviation": []
    }

    for emotion in emotions:
        mean_prob = df_chunks[emotion].mean()
        std_prob = df_chunks[emotion].std()

        summary_data["Mean Probability"].append(mean_prob)
        summary_data["Standard Deviation"].append(std_prob)

    df_summary = pd.DataFrame(summary_data)
    return df_summary


# Function to save the results to an Excel file
def save_to_excel(df_chunks, filename="emotion_analysis_updated.xlsx"):
    df_summary = calculate_summary_statistics(df_chunks)

    with pd.ExcelWriter(filename) as writer:
        df_chunks.to_excel(writer, sheet_name="Chunks", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
    print(f"Results saved to '{filename}'")


# Main function to run the analysis
def run_emotion_analysis(text):
    df_chunks = process_chunks(text)
    save_to_excel(df_chunks)


# Run analysis on the selected text
text = fathers_pov
run_emotion_analysis(text)