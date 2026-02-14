```markdown
# 📘 Academic Abstract Classifier  
*A Machine Learning Project for Automated Research Field Classification*

---

## 📝 Overview  
The **Academic Abstract Classifier** is an end-to-end Machine Learning application designed to automatically predict the academic research field of any given abstract.  
It uses a **fine-tuned DistilBERT transformer model** to classify abstracts into:

- Artificial Intelligence (AI)
- Business Research
- Healthcare Research
- Environmental Sciences

This project includes data collection, preprocessing, training, evaluation, and deployment through a **Flask backend API** and a clean **HTML/CSS frontend**.

---

## 🚀 Key Features  
- Fine-tuned transformer model  
- Weighted loss for class imbalance  
- Clean, modern web UI  
- HuggingFace inference pipeline  
- Modular project architecture  
- Reproducible ML training workflow  
- Fast, lightweight inference  

---

## 📁 Project Folder Structure


Academic-Classifier/
│
├── models/
│   └── abstract_classifier/            # Trained ML model
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── model.safetensors
│       ├── vocab.txt
│       ├── special_tokens_map.json
│       ├── label_map.json
│
├── src/
│   ├── flask_app.py                    # Flask backend for inference
│   ├── infer_local.py                  # CLI testing script
│   └── __init__.py
│
├── templates/
│   └── index.html                      # Web UI
│
├── static/
│   └── style.css                       # CSS styling
│
├── requirements.txt                    # Dependencies
└── README.md                           # Documentation
```

---

## 🔧 Installation & Setup Guide  

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/Academic-Classifier.git
cd Academic-Classifier
```

### 2️⃣ Create & Activate Virtual Environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Add Your Trained Model
Place your model files in:
```
models/abstract_classifier/
```

**Required files:**
- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `vocab.txt`
- `special_tokens_map.json`
- `tokenizer_config.json`
- `label_map.json`

---

## ▶️ Running the Application

### Start Flask Server
```bash
cd src
python flask_app.py
```

Visit the app in a browser:
```
http://127.0.0.1:5000/
```

---

## 🎨 Frontend Overview

The HTML/CSS interface includes:
- Title banner
- Text area for abstract input
- Classify & Clear buttons
- Prediction + confidence output
- Animated confidence progress bar
- Fully responsive design

---

## 🧠 Model Architecture

**✔ Base Model**
- DistilBERT: lightweight, fast, transformer-based model

**✔ Training Steps**
1. Dataset preparation (8000 samples)
2. Label mapping for 4 classes
3. Tokenization using DistilBERT tokenizer
4. Apply weighted loss for class imbalance
5. Fine-tuning (3 epochs)
6. Evaluation using accuracy + F1 macro
7. Save trained model + tokenizer

**✔ Typical Metrics**
- Accuracy: ~78%
- F1 Macro: ~78%

---

## 🧪 Example Input and Output

**Input Abstract:**
> This study proposes a deep reinforcement learning framework for autonomous robotic navigation in complex and dynamic environments. Various policy gradient methods are evaluated.

**Expected Output:**
```
Predicted Field: Artificial Intelligence
Confidence: 92.1%
```

---

## 📦 Backend Overview

The backend (`flask_app.py`) performs:
- Loading the fine-tuned transformer model
- Loading tokenizer & label map
- Exposing `/predict` endpoint
- Converting `LABEL_0` → Actual class name
- Returning prediction + confidence
- Handling empty input gracefully

---

## 📊 Dataset Summary

**Total Samples:** 8000 (via ArXiv API)

| Category | Samples |
|----------|---------|
| AI | ~4000 |
| Business | ~1800 |
| Healthcare | ~1200 |
| Environmental Science | ~1000 |

**Dataset files include:**
- `train.csv`
- `val.csv`
- `test.csv`

---

## 📘 ML Workflow Summary

**Cell A — Tokenization & Class Weights**
- Load dataset
- Tokenize abstracts
- Convert labels → IDs
- Compute class weights
- Save `label_map.json`

**Cell B — TrainingArguments**
- LR, batch size, epochs
- Save best model
- Use F1 macro as metric

**Cell C — WeightedTrainer**
- Custom loss function
- Override `compute_loss`
- Balanced gradient updates

**Cell D — Training & Evaluation**
- Train model
- Save model & tokenizer
- Validate performance
- Generate classification report

---

## 🛠 Technologies Used

| Component | Technology |
|-----------|------------|
| Backend | Flask |
| Model | DistilBERT (HuggingFace) |
| ML Tools | PyTorch, Datasets, Evaluate |
| Frontend | HTML, CSS |
| Dataset | ArXiv API |
| Training | Google Colab GPU |

---

## 📌 Future Enhancements

- Add more scientific categories
- Deploy model as cloud microservice
- Convert frontend to React
- Add PDF upload + text extraction
- Train larger models (RoBERTa / LLaMA)

---

## ✨ Conclusion

The Academic Abstract Classifier demonstrates a complete NLP machine-learning workflow — from dataset creation to training, evaluation, deployment, and user interaction through a web interface.

It can support academic indexing, research portals, and automated literature analysis with high accuracy and speed.

---

## 👤 Author

Ansuj Kumar Meher
2025 — Academic Abstract Classifier Project
```

