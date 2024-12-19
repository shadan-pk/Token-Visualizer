from flask import Flask, request, render_template, jsonify
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

app = Flask(__name__)

# Function to get the model and tokenizer based on selection
def get_model_and_tokenizer(model_name):
    model_mapping = {
        'bert-base-uncased': 'bert-base-uncased',
        'roberta-base': 'roberta-base',
        'distilbert-base-uncased': 'distilbert-base-uncased'
    }
    selected_model = model_mapping.get(model_name, 'bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(selected_model)
    model = AutoModel.from_pretrained(selected_model)
    return tokenizer, model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tokenize', methods=['POST'])
def tokenize():
    data = request.json
    text = data.get('text', '')
    model_name = data.get('model', 'bert-base-uncased')

    # Load model and tokenizer
    tokenizer, model = get_model_and_tokenizer(model_name)
    inputs = tokenizer(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Get token embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze()

    # Compute token weights
    token_weights = torch.norm(embeddings, dim=1).numpy()

    # Remove special tokens
    special_tokens = [tokenizer.cls_token, tokenizer.sep_token]
    filtered_tokens = [token for token in tokens if token not in special_tokens]
    filtered_indices = [i for i, token in enumerate(tokens) if token not in special_tokens]
    filtered_weights = token_weights[filtered_indices]

    # Normalize weights
    filtered_weights = (filtered_weights - filtered_weights.min()) / (filtered_weights.max() - filtered_weights.min())

    # Generate color map
    color_map = plt.cm.viridis(filtered_weights)
    hex_colors = [mcolors.rgb2hex(color[:3]) for color in color_map]

    # Create response data
    colored_tokens = [{'token': token, 'color': color} for token, color in zip(filtered_tokens, hex_colors)]

    return jsonify(colored_tokens)

if __name__ == '__main__':
    app.run(debug=True)
