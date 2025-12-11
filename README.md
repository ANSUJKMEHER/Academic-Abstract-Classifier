


---

📘 Academic Abstract Classifier

A Machine Learning Project for Automated Research Field Classification


---

📝 Overview

The Academic Abstract Classifier is an end-to-end Machine Learning application designed to automatically predict the academic research field of any given abstract.
It uses a fine-tuned DistilBERT transformer model to classify abstracts into:

Artificial Intelligence (AI)

Business Research

Healthcare Research

Environmental Science


This project combines dataset collection, preprocessing, model training, evaluation, and deployment into a clean, modular architecture.
A Flask backend API powers the model inference, while a minimal, user-friendly HTML/CSS frontend delivers the results.


---

🚀 Key Features

Fine-tuned transformer model

Weighted loss to handle class imbalance

Modern and simple web UI

HuggingFace inference pipeline

Clean directory structure

Reproducible training pipeline

Lightweight, fast inference



---

📁 Project Folder Structure

Academic-Classifier/
│
├── models/
│   └── abstract_classifier/            # Place your trained ML model here
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
│   └── style.css                       # UI styling
│
├── requirements.txt                    # Python dependencies
└── README.md                           # Documentation


---

🔧 Installation & Setup Guide

1️⃣ Clone the Repository

git clone https://github.com/<your-username>/Academic-Classifier.git
cd Academic-Classifier

2️⃣ Create & Activate a Virtual Environment

Windows

python -m venv .venv
.venv\Scripts\activate

Mac/Linux

python3 -m venv .venv
source .venv/bin/activate

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Add Your Trained Model

Place your model files inside:

models/abstract_classifier/

Required files:

model.safetensors

config.json

tokenizer.json

vocab.txt

special_tokens_map.json

tokenizer_config.json

label_map.json



---

▶️ Running the Application

Start the Flask Server

cd src
python flask_app.py

Open in browser:

http://127.0.0.1:5000/


---

🎨 Frontend Overview

The web interface includes:

Clean banner with project title

Large text area for abstracts

Classify & Clear buttons

Dynamic prediction result

Animated confidence bar

Responsive layout


Built using HTML + CSS only.


---

🧠 Model Architecture

✔ Base Model

DistilBERT, optimized for speed and accuracy


✔ Training Steps

1. Dataset preparation (8000 samples)


2. Label mapping for 4 classes


3. DistilBERT tokenization


4. Weighted loss for imbalance


5. Fine-tuning (3 epochs)


6. Evaluation (accuracy + F1-macro)


7. Save final model & tokenizer



✔ Typical Metrics

Accuracy: ~78%

F1-Macro: ~78%



---

🧪 Example Input & Output

Input Abstract:

> This study proposes a deep reinforcement learning framework for autonomous robotic navigation in complex and dynamic environments. Various policy gradient methods are evaluated.



Expected Output:

Predicted Field: Artificial Intelligence
Confidence: 92.1%


---

📦 Backend Overview

The backend (flask_app.py) handles:

Loading the fine-tuned transformer model

Loading tokenizer & label mapping

Exposing /predict API

Converting HuggingFace labels (LABEL_0 → Class Name)

Returning prediction + confidence score

Error handling



---

📊 Dataset Summary

Total Samples: 8000 (from ArXiv API)

Category	Samples

AI	~4000
Business	~1800
Healthcare	~1200
Environmental Science	~1000


Preprocessed dataset files:

train.csv

val.csv

test.csv



---

📘 ML Workflow Summary

Cell A — Tokenization & Class Weights

Load dataset

Tokenize with DistilBERT

Map labels → IDs

Compute class weights

Save label_map.json


Cell B — TrainingArguments

Learning rate

Epochs

Batch size

Save best model

F1-macro as metric


Cell C — WeightedTrainer

Custom loss

Override compute_loss

Balanced gradient updates


Cell D — Training & Evaluation

Train

Save model & tokenizer

Compute accuracy + F1

Generate classification report



---

🛠 Technologies Used

Component	Technology

Backend	Flask
Model	DistilBERT (HuggingFace)
ML Tools	PyTorch, Datasets, Evaluate
Frontend	HTML, CSS
Dataset	ArXiv API
Training	Google Colab GPU



---

📌 Future Enhancements

Add more scientific categories

Deploy as cloud API

Convert UI to React

Add PDF upload + text extraction

Fine-tune RoBERTa / LLaMA for higher accuracy



---

✨ Conclusion

The Academic Abstract Classifier demonstrates a full machine-learning workflow — from dataset extraction to training, evaluation, saving, and deployment via a web interface.

This system can greatly support research indexing, academic portals, and automated literature analysis.


---

👤 Author

Ansuj Kumar Meher
2025 — Academic Abstract Classifier Project


---

