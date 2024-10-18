
# Biblical Sentiment Analysis

This repository performs sentiment analysis on biblical narratives using multiple Large Language Models (LLMs). The analysis focuses on emotions based on the **Ekman emotion model** and the **GoEmotions emotion model**, offering insights into biblical texts from various points of view.

Specifically, the experiment explores **sentiment and emotional trajectories** within the story of the Prodigal Son, analyzed from the perspectives of the **Father**, **Younger Son**, and **Older Son**. The analysis is conducted using four different LLMs and two distinct text chunking techniques: **sliding window chunking** and **sentence-based chunking**.

## Features
- **Multiple LLMs**: Uses four different large language models for comparative analysis.
- **Emotion Models**: Supports both the Ekman and GoEmotions emotion models.
- **Text Chunking**: Includes both sliding window and sentence-based chunking techniques.
- **Perspective-based Analysis**: Sentiment analysis is performed from the viewpoints of three key characters in the Prodigal Son narrative: the Father, Younger Son, and Older Son.

## Getting Started

### How to Duplicate the Experiment

1. Clone this repository:
    ```bash
    git clone https://github.com/darrellcolehill/biblical-sentiment-analysis.git
    cd biblical-sentiment-analysis
    ```

2. Open `experiment.ipynb` in Jupyter Notebook.

3. Run all cells:
    - From the Jupyter Notebook toolbar, select **Cell > Run All** to execute the experiment.

The notebook contains preconfigured cells that will guide you through the entire process, from loading the text to visualizing the sentiment and emotional outcomes.

## Usage

The notebook processes the **Prodigal Son** text, analyzes the emotions expressed by each character, and tracks the sentiment progression over time.

### LLMs Used
1. [joeddav/distilbert-base-uncased-go-emotions-student](https://huggingface.co/joeddav/distilbert-base-uncased-go-emotions-student)
2. [monologg/bert-base-cased-goemotions-original](https://huggingface.co/monologg/bert-base-cased-goemotions-original)
3. [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)
4. [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

Each LLM provides unique insights based on its architecture and training, allowing for a diverse sentiment and emotion comparison.

### Emotion Models Supported
- **Ekman Model**: Includes six basic emotions—anger, disgust, fear, happiness, sadness, and surprise.
- **GoEmotions Model**: Includes a broader range of 27 emotions, including neutral, caring, and enthusiasm.

## Results
The results can be found in the "results" folder. 

## Experiment Replication Video
[![IMAGE ALT TEXT](http://img.youtube.com/vi/cx9y_vY9CEA/0.jpg)](http://www.youtube.com/watch?v=cx9y_vY9CEA "Biblical Sentiment Analysis Video")
