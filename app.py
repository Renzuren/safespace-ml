# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

# Configuration
class Config:
    BERT_MODEL = 'bert-base-uncased'
    MAX_LENGTH = 256
    HIDDEN_SIZE = 768
    NUM_CLASSES = 5
    RNN_HIDDEN_SIZE = 256
    RNN_NUM_LAYERS = 2
    RNN_DROPOUT = 0.3
    BIDIRECTIONAL = True
    DROPOUT_RATE = 0.4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# Model Architecture
class SkipRNNLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        output_size = hidden_size * (2 if bidirectional else 1)
        self.layer_norm = torch.nn.LayerNorm(output_size)
        self.proj = torch.nn.Linear(input_size, output_size) if input_size != output_size else torch.nn.Identity()

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        skip_out = self.layer_norm(self.dropout(rnn_out + self.proj(x)))
        return skip_out

class HarassmentModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.BERT_MODEL)
        
        self.skip_rnn1 = SkipRNNLayer(
            config.HIDDEN_SIZE, config.RNN_HIDDEN_SIZE,
            config.RNN_NUM_LAYERS, config.RNN_DROPOUT, config.BIDIRECTIONAL
        )
        
        self.skip_rnn2 = SkipRNNLayer(
            config.RNN_HIDDEN_SIZE * 2, config.RNN_HIDDEN_SIZE // 2,
            max(1, config.RNN_NUM_LAYERS // 2), config.RNN_DROPOUT, config.BIDIRECTIONAL
        )
        
        rnn_out_size = (config.RNN_HIDDEN_SIZE // 2) * 2
        self.attention = torch.nn.MultiheadAttention(rnn_out_size, 4, config.DROPOUT_RATE, batch_first=True)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(rnn_out_size * 2, config.RNN_HIDDEN_SIZE // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.DROPOUT_RATE),
            torch.nn.Linear(config.RNN_HIDDEN_SIZE // 4, config.NUM_CLASSES)
        )
        self.dropout = torch.nn.Dropout(config.DROPOUT_RATE)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids, attention_mask).last_hidden_state
        rnn1 = self.skip_rnn1(bert_out)
        rnn2 = self.skip_rnn2(rnn1)
        attn_out, _ = self.attention(rnn2, rnn2, rnn2)
        
        combined = torch.cat([torch.max(attn_out, dim=1)[0], torch.mean(attn_out, dim=1)], dim=1)
        return self.classifier(self.dropout(combined))

# Load Model
print("Loading tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL)
model = HarassmentModel(config)
model.load_state_dict(torch.load('best_model.pt', map_location=config.DEVICE))
model.to(config.DEVICE)
model.eval()
print(f"Model loaded successfully on {config.DEVICE}")

LABELS = ['Physical Harassment', 'Verbal Harassment', 'Non-Verbal Harassment', 'Not Harassment', 'Cyber Sexual Harassment']

# Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({'error': 'Missing description field'}), 400
        
        description = data['description']
        
        # Tokenize input
        inputs = tokenizer(
            description,
            max_length=config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = model(inputs['input_ids'].to(config.DEVICE), inputs['attention_mask'].to(config.DEVICE))
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
        
        # Return response
        return jsonify({
            'label': LABELS[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probabilities': {label: float(probs[i]) for i, label in enumerate(LABELS)}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'device': str(config.DEVICE)
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Harassment Detection Microservice',
        'endpoints': {
            '/predict': 'POST - Detect harassment from text',
            '/health': 'GET - Check service health'
        }
    })

if __name__ == '__main__':
    # Run Flask app
    app.run(host='0.0.0.0', port=8000, debug=False)
