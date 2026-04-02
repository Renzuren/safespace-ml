# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import traceback
import boto3
from botocore.client import Config
import os
import tempfile

# Configuration
class Config:
    BERT_MODEL = 'bert-base-uncased'
    MAX_LENGTH = 256
    HIDDEN_SIZE = 768
    NUM_OFFENSE_CLASSES = 5
    NUM_SEVERITY_CLASSES = 4
    RNN_HIDDEN_SIZE = 256
    RNN_NUM_LAYERS = 2
    RNN_DROPOUT = 0.3
    BIDIRECTIONAL = True
    DROPOUT_RATE = 0.4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Digital Ocean Spaces Configuration
    SPACES_KEY = os.environ.get('SPACES_KEY', 'DO00ZM9R4HFWCBVZ8PXA')
    SPACES_SECRET = os.environ.get('SPACES_SECRET', 'N/HW3jv0QQZrDTg5oCihhyfXo8VX1R21rmHoUzanC7k')
    SPACES_REGION = os.environ.get('SPACES_REGION', 'sgp1')
    SPACES_BUCKET = os.environ.get('SPACES_BUCKET', 'uplb-oash-bucket')
    SPACES_ENDPOINT = f'https://{SPACES_REGION}.digitaloceanspaces.com'
    MODEL_FILENAME = 'best_model.pt'

config = Config()

# Digital Ocean Spaces Downloader
class DigitalOceanSpacesDownloader:
    def __init__(self):
        self.session = boto3.session.Session()
        self.client = self.session.client(
            's3',
            region_name=config.SPACES_REGION,
            endpoint_url=config.SPACES_ENDPOINT,
            aws_access_key_id=config.SPACES_KEY,
            aws_secret_access_key=config.SPACES_SECRET
        )
    
    def download_model(self, local_path):
        """Download model from Digital Ocean Spaces"""
        try:
            print(f"Downloading model from Digital Ocean Spaces...")
            print(f"Bucket: {config.SPACES_BUCKET}")
            print(f"File: {config.MODEL_FILENAME}")
            
            # Download file
            self.client.download_file(
                config.SPACES_BUCKET, 
                config.MODEL_FILENAME, 
                local_path
            )
            
            # Verify file size
            file_size = os.path.getsize(local_path)
            print(f"Download complete! File size: {file_size / (1024*1024):.2f} MB")
            return True
            
        except Exception as e:
            print(f"Error downloading from Spaces: {e}")
            return False
    
    def check_file_exists(self):
        """Check if model file exists in Spaces"""
        try:
            self.client.head_object(Bucket=config.SPACES_BUCKET, Key=config.MODEL_FILENAME)
            print("✓ Model file found in Digital Ocean Spaces")
            return True
        except Exception as e:
            print(f"✗ Model file not found: {e}")
            return False

# Alternative: Direct HTTP Download (kung public ang bucket)
def download_model_http():
    """Alternative download method using HTTP if bucket is public"""
    import requests
    
    model_urls = [
        f"https://{config.SPACES_BUCKET}.{config.SPACES_REGION}.digitaloceanspaces.com/{config.MODEL_FILENAME}",
        f"https://{config.SPACES_BUCKET}.{config.SPACES_REGION}.cdn.digitaloceanspaces.com/{config.MODEL_FILENAME}"
    ]
    
    for url in model_urls:
        try:
            print(f"Trying to download from: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            local_path = "/tmp/best_model.pt"
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}%", end="")
            
            print(f"\n✓ Downloaded from {url}")
            print(f"File size: {downloaded / (1024*1024):.2f} MB")
            return local_path
            
        except Exception as e:
            print(f"✗ Failed from {url}: {e}")
            continue
    
    return None

# Model Architecture (your existing architecture)
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
        if input_size != output_size:
            self.proj = torch.nn.Linear(input_size, output_size)
        else:
            self.proj = torch.nn.Identity()

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        x_proj = self.proj(x)
        skip_out = self.layer_norm(self.dropout(rnn_out + x_proj))
        return skip_out

class AdvancedBERTSkipRNN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.BERT_MODEL)
        
        self.skip_rnn1 = SkipRNNLayer(
            config.HIDDEN_SIZE,
            config.RNN_HIDDEN_SIZE,
            config.RNN_NUM_LAYERS,
            config.RNN_DROPOUT,
            config.BIDIRECTIONAL
        )
        
        self.skip_rnn2 = SkipRNNLayer(
            config.RNN_HIDDEN_SIZE * (2 if config.BIDIRECTIONAL else 1),
            config.RNN_HIDDEN_SIZE // 2,
            max(1, config.RNN_NUM_LAYERS // 2),
            config.RNN_DROPOUT,
            config.BIDIRECTIONAL
        )
        
        rnn_out_size = (config.RNN_HIDDEN_SIZE // 2) * (2 if config.BIDIRECTIONAL else 1)
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=rnn_out_size,
            num_heads=4,
            dropout=config.DROPOUT_RATE,
            batch_first=True
        )
        
        self.dropout = torch.nn.Dropout(config.DROPOUT_RATE)
        
        self.offense_classifier = torch.nn.Sequential(
            torch.nn.Linear(rnn_out_size * 2, config.RNN_HIDDEN_SIZE // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.DROPOUT_RATE),
            torch.nn.Linear(config.RNN_HIDDEN_SIZE // 4, config.NUM_OFFENSE_CLASSES)
        )
        self.severity_classifier = torch.nn.Sequential(
            torch.nn.Linear(rnn_out_size * 2, config.RNN_HIDDEN_SIZE // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.DROPOUT_RATE),
            torch.nn.Linear(config.RNN_HIDDEN_SIZE // 4, config.NUM_SEVERITY_CLASSES)
        )
    
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = bert_out.last_hidden_state
        
        rnn1 = self.skip_rnn1(seq_out)
        rnn2 = self.skip_rnn2(rnn1)
        
        attn_out, _ = self.attention(rnn2, rnn2, rnn2)
        
        max_pool = torch.max(attn_out, dim=1)[0]
        avg_pool = torch.mean(attn_out, dim=1)
        combined = torch.cat([max_pool, avg_pool], dim=1)
        
        logits_offense = self.offense_classifier(self.dropout(combined))
        logits_severity = self.severity_classifier(self.dropout(combined))
        return logits_offense, logits_severity

# Load Model with retry mechanism
def load_model_with_retry(max_retries=3):
    """Load model with retry mechanism"""
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL)
    model = AdvancedBERTSkipRNN(config)
    
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries} to download and load model...")
            
            # Try HTTP download first (if public)
            model_path = download_model_http()
            
            if not model_path:
                # Try boto3 download (if authenticated)
                downloader = DigitalOceanSpacesDownloader()
                if downloader.check_file_exists():
                    model_path = "/tmp/best_model.pt"
                    downloader.download_model(model_path)
                else:
                    raise Exception("Model not found in Spaces")
            
            # Load the model
            print(f"Loading model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=config.DEVICE)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✓ Loaded from full checkpoint format")
            else:
                model.load_state_dict(checkpoint)
                print("✓ Loaded from state dict format")
            
            model.to(config.DEVICE)
            model.eval()
            
            print(f"✓ Model loaded successfully on {config.DEVICE}")
            return tokenizer, model
            
        except Exception as e:
            print(f"✗ Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            import time
            time.sleep(5)  # Wait before retry

# Initialize Flask app
print("=" * 50)
print("Initializing Harassment Detection Service")
print("=" * 50)

# Load tokenizer and model
tokenizer, model = load_model_with_retry(max_retries=3)

# Labels
OFFENSE_LABELS = ['Physical Harassment', 'Verbal Harassment', 'Non-Verbal Harassment', 
                  'Not Harassment', 'Cyber Sexual Harassment']
SEVERITY_LABELS = ['Light', 'Less Grave', 'Grave', 'None']

# Flask App
app = Flask(_name_)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({'error': 'Missing description field'}), 400
        
        description = data['description']
        
        inputs = tokenizer(
            description,
            max_length=config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            offense_logits, severity_logits = model(
                inputs['input_ids'].to(config.DEVICE), 
                inputs['attention_mask'].to(config.DEVICE)
            )
            
            offense_probs = torch.softmax(offense_logits, dim=1).cpu().numpy()[0]
            severity_probs = torch.softmax(severity_logits, dim=1).cpu().numpy()[0]
            
            offense_idx = np.argmax(offense_probs)
            severity_idx = np.argmax(severity_probs)
            
            # Override severity for Not Harassment
            if OFFENSE_LABELS[offense_idx] == 'Not Harassment':
                severity_idx = 3
                severity_probs = [0.0, 0.0, 0.0, 1.0]
        
        return jsonify({
            'offense': {
                'label': OFFENSE_LABELS[offense_idx],
                'confidence': float(offense_probs[offense_idx]),
                'probabilities': {label: float(offense_probs[i]) for i, label in enumerate(OFFENSE_LABELS)}
            },
            'severity': {
                'label': SEVERITY_LABELS[severity_idx],
                'confidence': float(severity_probs[severity_idx]),
                'probabilities': {label: float(severity_probs[i]) for i, label in enumerate(SEVERITY_LABELS)}
            }
        })
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'device': str(config.DEVICE),
        'model_loaded': True
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Harassment Detection Microservice',
        'status': 'running',
        'endpoints': ['/predict', '/health']
    })

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=8080, debug=False)