# RESOLVEMATE – AI-Powered Ticket Classification & Automated Support System

RESOLVEMATE is an end-to-end **AI-driven customer support automation system** that classifies support tickets, predicts urgency and sentiment, and generates automated responses using **NLP models + OpenAI API**.  
It includes a **web interface**, authentication, and real-time analytics for efficient ticket handling.

---

## 🚀 Key Features

### 🧠 Advanced NLP Models  
- Fine-tuned **DistilBERT** transformer achieving **91% accuracy** across **5 categories and 25 subcategories**.  
- Built a **multi-task model** predicting:  
  - Ticket **category**  
  - **Urgency** (80% F1-score)  
  - **Sentiment** (85% F1-score)  
- Ticket classification improves triaging and routing accuracy significantly.

### 🤖 Automated Response Generation  
- Integrated **OpenAI API** for intelligent ticket reply generation.  
- Reduces manual ticket handling time by **60%**.  
- Produces consistent, context-aware replies.

### 🌐 Web Application (Flask + Firebase)  
- Secure, scalable, responsive UI for real-time ticket submission.  
- Integrated **Firebase Authentication** for login/signup.  
- Ticket analytics dashboard with prediction + AI-generated message.  
- Reduced support backlog by **30%** through automated workflows.

---

## 📁 Project Structure

```
RESOLVEMATE/
│── __pycache__/
│── templates/
│     ├── dashboard.html
│     ├── history.html
│     ├── index.html
│     ├── login.html
│     └── results.html
│── ticket_model/
│     ├── best_model.pt
│     ├── final_model.pt
│     ├── label_encoders.joblib
│     ├── special_tokens_map.json
│     ├── tokenizer_config.json
│     ├── tokenizer.json
│     └── vocab.txt
│── app.py
│── ticket_pipeline_using_llm.py
│── customer_support_dataset.csv
│── template_cache.joblib
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/Sujitha-1005/REsolvemate-ticket-classification-system.git
cd RESOLVEMATE
```

### 2️⃣ Create and activate a virtual environment  
**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

### **Start the Flask server**
```bash
python app.py
```

App runs at:

```
http://127.0.0.1:5000/
```

---

## 📦 Requirements

```
Flask
torch
transformers
pandas
numpy
firebase-admin
scikit-learn
joblib
openai
```

(Add exact versions if needed.)

---

## 🤖 Model Training

To retrain the DistilBERT model:

```bash
python ticket_pipeline_using_llm.py
```

Outputs generated:

- `final_model.pt`  
- `best_model.pt`  
- Tokenizer files  
- Label encoder mappings  

---

## 🔥 OpenAI Integration

Used for **automated reply generation** based on ticket description + model predictions.

Add your OpenAI API key in environment variable:

```bash
export OPENAI_API_KEY="your-key"
```

---

## 🔐 Firebase Authentication Setup

Add your Firebase keys in `app.py`:

```python
firebaseConfig = {
  "apiKey": "YOUR_API_KEY",
  "authDomain": "YOUR_PROJECT.firebaseapp.com",
  "projectId": "YOUR_PROJECT",
  "storageBucket": "YOUR_PROJECT.appspot.com",
  "messagingSenderId": "YOUR_SENDER_ID",
  "appId": "YOUR_APP_ID"
}
```

Features enabled:
- Login / Signup  
- User-based ticket history  
- Dashboard tracking  

---


---

## 🤝 Contributing

Pull requests are welcome!  
Feel free to create issues or suggest improvements.

---

## 📜 License

This project is licensed under the **MIT License**.


