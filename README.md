

---

# 📘 Academic Abstract Classifier  
*A Machine Learning Project for Automated Research Field Classification*

---

## 📝 Overview  
The **Academic Abstract Classifier** is an end-to-end Machine Learning application designed to automatically predict the academic research field of any given abstract.  
By leveraging a **fine-tuned DistilBERT transformer model**, the system can classify abstracts into:

- **Artificial Intelligence (AI)**
- **Business Research**
- **Healthcare Research**
- **Environmental Science**

This project integrates **dataset collection, preprocessing, model training, evaluation, and deployment** into a clean, modular structure. The final output is served through a **Flask-based backend API** and a **beautiful, minimalistic HTML/CSS frontend**.

This classifier is useful for:
- Academic indexing  
- Research library organization  
- Automated literature survey tools  
- University/college project submissions  
- Research recommendation engines  

---

## 🚀 Key Features  
- Custom-trained transformer model  
- Balanced dataset using class weights  
- Weighted loss function for improved accuracy  
- Interactive web-based UI  
- Fast inference via HuggingFace pipeline  
- Clean backend architecture  
- Fully reproducible training workflow  

---
```
## 📁 Project Folder Structure  
Below is the **exact folder structure**, fully formatted for README.md:
Academic-Classifier/
│
├── models/
│   └── abstract_classifier/
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── model.safetensors
│       ├── vocab.txt
│       ├── special_tokens_map.json
│       ├── label_map.json
│
├── src/
│   ├── flask_app.py
│   ├── infer_local.py
│   └── __init__.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── requirements.txt
└── README.md

---

## 🔧 Installation & Setup Guide  

### 1️⃣ Clone the Repository
```bash
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

Place your model folder here:

models/abstract_classifier/

Mandatory files include:

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

Then open your browser and visit:

http://127.0.0.1:5000/


---

🎨 Frontend Overview

The web interface is clean and modern, built using HTML + CSS.

Features include:

Banner with project title

Large text box for pasting abstract

Two buttons: Classify & Clear

Prediction result section

Animated confidence bar

Gradient backgrounds & card layout

Fully responsive UI



---

🧠 Model Architecture

✔ Base Model

The classifier uses DistilBERT, a lightweight version of BERT optimized for speed.

✔ Training Steps

1. Dataset preparation (8000 samples)


2. Label mapping (AI, Business, Healthcare, Environmental Science)


3. Tokenization using DistilBERT tokenizer


4. Weighted loss to handle class imbalance


5. Fine-tuning for 3 epochs


6. Evaluation using accuracy & F1-macro


7. Saving trained model + tokenizer



✔ Metrics

Typical validation results:

Accuracy: ~78%

F1 Macro: ~78%

Balanced class performance due to weighted loss



---

🧪 Example Input

Paste this into the textbox:

This study proposes a deep reinforcement learning framework for autonomous robotic navigation in complex and dynamic environments. Various policy gradient methods are evaluated.

Expected Output:

Predicted Field: Artificial Intelligence
Confidence: 92.1%


---

📦 Backend Overview

The backend (flask_app.py):

Loads your trained HuggingFace model

Loads tokenizer & label mapping

Exposes /predict API for inference

Supports CPU/GPU inference

Converts LABEL_0 → Actual Label

Returns label + confidence

Handles empty input errors



---

📊 Dataset Summary

Dataset Size: 8000 abstracts
Collected from ArXiv using Python API.

Categories:

AI – ~4000 samples

Business – ~1800 samples

Healthcare – ~1200 samples

Environmental Science – ~1000 samples


After preprocessing:

Cleaned, labeled, and saved as:

train.csv

val.csv

test.csv




---

📘 How the ML Workflow Was Implemented

Cell A — Tokenization, Model Setup & Class Weights

Loads dataset

Tokenizes text with DistilBERT

Converts labels to integer IDs

Computes class weights

Saves label_map.json


Cell B — Define TrainingArguments

Learning rate, batch size, epochs

Saves best model

Uses F1-macro as evaluation metric

Handles transformers version differences


Cell C — WeightedTrainer

Custom loss function with class weights

Overrides the default trainer’s compute_loss

Ensures balanced gradients


Cell D — Training & Evaluation

Starts training loop

Saves the fine-tuned model & tokenizer

Generates validation metrics

Creates classification_report & confusion matrix



---

🛠 Technologies Used

Component	Technology

Backend	Flask
Model	HuggingFace Transformers (DistilBERT)
ML Tools	PyTorch, Datasets, Evaluate
Frontend	HTML, CSS, JavaScript
Dataset	ArXiv API
Training	Google Colab GPU



---

📌 Future Enhancements

Add more scientific categories

Deploy as a cloud-hosted API

Convert frontend to React

Add PDF upload & auto-extraction

Improve accuracy with RoBERTa/LLaMA fine-tuning



---

✨ Conclusion

This project demonstrates the complete lifecycle of an NLP-based machine learning system—from dataset creation to model training, evaluation, and final deployment.
Using modern transformer models and a clean development pipeline, the Academic Abstract Classifier provides fast and accurate predictions, making it valuable for academic organizations, research students, and digital libraries.


---

👤 Author

Your Name
2025 — Academic Abstract Classifier Project

---

If you want:

✅ Even more detailed README  
✅ A short README for GitHub summary  
✅ A professional project banner  
✅ A documentation PDF  

Just tell me!
