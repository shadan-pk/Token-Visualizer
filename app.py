import random
from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)

model_cache = {}

def get_model_and_tokenizer(model_name):
    if model_name in model_cache:
        return model_cache[model_name]
    
    if model_name == 'chatgpt':
        # Placeholder for ChatGPT model loading
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        model = AutoModel.from_pretrained('gpt2')
    elif model_name == 'llama-3.2':
        # Placeholder for LLaMA 3.2 model loading
        tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')  # Replace with actual LLaMA model identifier
        model = AutoModel.from_pretrained('facebook/opt-125m')  # Replace with actual LLaMA model identifier
    elif model_name == 'mistral':
        # Placeholder for Mistral model loading
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')  # Replace with actual Mistral model identifier
        model = AutoModel.from_pretrained('roberta-base')  # Replace with actual Mistral model identifier
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    
    model_cache[model_name] = (tokenizer, model)
    return tokenizer, model

def generate_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tokenize', methods=['POST'])
def tokenize():
    try:
        data = request.json
        text = data.get('text', '')
        model_name = data.get('model', 'bert-base-uncased')

        # Validate input
        if not text:
            return jsonify({'error': 'No text provided'}), 400

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

        # Clean tokens
        cleaned_tokens = [token.replace('Ġ', '') for token in filtered_tokens]

        # Generate color map
        token_colors = [generate_color() for _ in cleaned_tokens]

        # Create response
        response = {
            'tokens': cleaned_tokens,
            'weights': filtered_weights.tolist(),
            'colors': token_colors
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)