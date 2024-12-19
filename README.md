# Token-Visualizer

**Token-Visualizer** is a visual tool designed to analyze and display the tokenization process of text inputs using various NLP models. The tool allows users to input text and visualize how different pre-trained models (like BERT, RoBERTa, and DistilBERT) tokenize the text. The tokens are then displayed with color coding based on their embeddings, providing an insightful view into the tokenization and attention mechanism of the selected model.

## Features
- **Text Tokenization**: Enter any text, and the tool will tokenize it using the selected NLP model.
- **Multiple Model Support**: Choose from popular NLP models such as:
  - BERT Base Uncased
  - RoBERTa Base
  - DistilBERT Base
- **Token Visualization**: Tokens are displayed with color-coding based on their embedding magnitudes, providing a clear visual representation of their significance.
- **Real-time Feedback**: The tool updates the tokenized view and embeddings as you interact with the interface.
- **Interactive Interface**: The tool features an easy-to-use UI with options to input text, select models, and see the tokenization results instantly.

## Tech Stack
- **Frontend**: HTML, CSS, JavaScript (Widgets for UI components)
- **Backend**: Python, Flask (Server-side logic and tokenization)
- **Machine Learning Models**: Hugging Face's `transformers` library (BERT, RoBERTa, DistilBERT)

## Prerequisites
Before you can run the Token-Visualizer tool locally, make sure you have the following installed:
- Python 3.6+
- pip (Python package installer)

Additionally, you will need to install the required Python libraries. These can be installed using `pip` by running the following command:

```bash
pip install -r requirements.txt
