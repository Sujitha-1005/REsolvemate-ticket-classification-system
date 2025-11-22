"""
app.py - Flask backend for ResolveMate Customer Support Ticket Classification System

This Flask app integrates with the trained ML model from ticket_pipeline_using_llm.py
and provides web interface for ticket classification with Firebase authentication.
"""

import os
import json
import joblib
import torch
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import numpy as np
from transformers import AutoTokenizer
import sys
from pathlib import Path

# Import your model classes and functions
# Make sure ticket_pipeline_using_llm.py is in the same directory
try:
    from ticket_pipeline_using_llm import (
        MultiTaskModel, 
        hybrid_infer, 
        find_first_pattern, 
        has_keyword,
        INVOICE_PATTERNS,
        REFUND_KEYWORDS
    )
except ImportError:
    print("❌ Error: Could not import from ticket_pipeline_using_llm.py")
    print("Make sure ticket_pipeline_using_llm.py is in the same directory as app.py")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
MODEL_DIR = "ticket_model"
TEMPLATE_CACHE_PATH = "template_cache.joblib"

# Global variables to store model components
model = None
tokenizer = None
label_encoders = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_components():
    """Load the trained model, tokenizer, and label encoders."""
    global model, tokenizer, label_encoders
    
    try:
        print(f"🔄 Loading model from {MODEL_DIR}...")
        
        # Check if model directory exists
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Model directory {MODEL_DIR} not found. Please train the model first.")
        
        # Load label encoders
        label_encoders_path = os.path.join(MODEL_DIR, 'label_encoders.joblib')
        if not os.path.exists(label_encoders_path):
            raise FileNotFoundError(f"Label encoders not found at {label_encoders_path}")
        
        label_encoders = joblib.load(label_encoders_path)
        print(f"✅ Loaded label encoders: {list(label_encoders.keys())}")
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
            print("✅ Loaded tokenizer from model directory")
        except:
            # Fallback to default model name
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
            print("⚠️ Loaded default tokenizer (distilbert-base-uncased)")
        
        # Initialize model architecture
        model = MultiTaskModel(
            model_name="distilbert-base-uncased",  # Base model name
            n_category=len(label_encoders['category'].classes_),
            n_subcategory=len(label_encoders['subcategory'].classes_),
            n_urgency=len(label_encoders['urgency'].classes_),
            n_sentiment=len(label_encoders['sentiment'].classes_),
            dropout=0.2
        )
        
        # Load trained weights
        model_path = os.path.join(MODEL_DIR, 'best_model.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_DIR, 'final_model.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found in {MODEL_DIR}")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ Loaded model from {model_path}")
        
        print("✅ All model components loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model components: {e}")
        return False

@app.route('/')
def index():
    """Home page route."""
    return render_template('index.html')

@app.route('/login')
def login():
    """Login page route."""
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page route."""
    return render_template('dashboard.html')

@app.route('/results')
def results():
    """Results page route."""
    return render_template('results.html')

@app.route('/history')
def history():
    """History page route."""
    return render_template('history.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for ticket classification."""
    try:
        # Check if model is loaded
        if model is None or tokenizer is None or label_encoders is None:
            return jsonify({
                'error': 'Model not loaded. Please check server logs.',
                'success': False
            }), 500
        
        # Get input data
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided in request',
                'success': False
            }), 400
        
        input_text = data['text'].strip()
        if not input_text:
            return jsonify({
                'error': 'Empty text provided',
                'success': False
            }), 400
        
        print(f"🔍 Classifying: {input_text[:100]}...")
        
        # Use the hybrid inference pipeline
        results = hybrid_infer(
            texts=[input_text],
            tokenizer=tokenizer,
            model=model,
            device=device,
            label_encoders=label_encoders,
            cache_path=TEMPLATE_CACHE_PATH
        )
        
        if not results:
            return jsonify({
                'error': 'No results from model',
                'success': False
            }), 500
        
        result = results[0]  # Get first (and only) result
        
        # Format response
        response = {
            'success': True,
            'category': result['predicted_category'],
            'subcategory': result['predicted_subcategory'],
            'urgency': result['predicted_urgency'],
            'sentiment': result['predicted_sentiment'],
            'invoice': result['invoice_detected'] if result['invoice_detected'] else 'Not found',
            'template': result['suggested_template'],
            'overridden_by_rule': result['overridden_by_rule'],
            'template_from_cache': result['template_from_cache']
        }
        
        print(f"✅ Classification complete: {result['predicted_category']} -> {result['predicted_subcategory']}")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        }), 500

@app.route('/model-status')
def model_status():
    """API endpoint to check model loading status."""
    status = {
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None,
        'label_encoders_loaded': label_encoders is not None,
        'device': str(device),
        'model_dir_exists': os.path.exists(MODEL_DIR)
    }
    
    if label_encoders is not None:
        status['categories'] = {
            'category': list(label_encoders['category'].classes_),
            'subcategory': list(label_encoders['subcategory'].classes_),
            'urgency': list(label_encoders['urgency'].classes_),
            'sentiment': list(label_encoders['sentiment'].classes_)
        }
    
    return jsonify(status)

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': str(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
    })

@app.errorhandler(404)
def not_found(error):
    """404 error handler."""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting ResolveMate Flask Application...")
    print(f"📱 Device: {device}")
    
    # Load model components on startup
    if load_model_components():
        print("🎉 Model loaded successfully! Starting server...")
    else:
        print("⚠️ Model loading failed! Server will start but predictions won't work.")
        print("   Make sure you have trained the model using:")
        print("   python ticket_pipeline_using_llm.py --do_train --csv your_dataset.csv")
    
    # Run the Flask app
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )