

---

# Academic Abstract Classifier

A machine learning project that automatically classifies research abstracts into four academic domains:
**Artificial Intelligence**, **Business Research**, **Healthcare Research**, and **Environmental Science**.  
The system uses a fine-tuned **DistilBERT Transformer model**, a **Flask backend**, and a clean **web interface** for easy testing.

---

## 1. Project Overview
This project aims to simplify academic text categorization by leveraging Natural Language Processing (NLP).  
Users can paste a research abstract, and the model predicts the most likely research domain along with a confidence score.

The pipeline includes:
- Dataset collection from **ArXiv API**
- Preprocessing & structured dataset creation
- Tokenization and label encoding
- Transformer model fine-tuning (DistilBERT)
- Weighted training for class imbalance
- Flask inference API
- HTML/CSS frontend for interaction

---

## 2. Features
- End-to-end ML pipeline (data → model → deployment)
- Balanced training with class-weights
- Saved model with label map for reproducible inference
- Lightweight Flask API
- Modern interface with gradient theme and confidence bar
- Reproducible code structure

---

## 3. Folder Structure

Academic-Classifier/ │ ├── models/ │     └── abstract_classifier/     ← (Add your trained model here) │ ├── src/ │     ├── flask_app.py             ← Backend API │     ├── infer_local.py           ← CLI testing script │     └── init.py │ ├── templates/ │     └── index.html               ← Frontend UI │ ├── static/ │     └── style.css                ← UI styling │ ├── data/ (optional) │ ├── requirements.txt └── README.md

---

## 4. Installation & Setup

### Step 1 — Clone Repository
```bash
git clone https://github.com/<your-username>/Academic-Classifier.git
cd Academic-Classifier

Step 2 — Create Virtual Environment

python -m venv .venv

Activate:

Windows:

.venv\Scripts\activate

Mac/Linux:

source .venv/bin/activate

Step 3 — Install Required Libraries

pip install -r requirements.txt

Step 4 — Add Model Files

Place your trained model inside:

models/abstract_classifier/

This folder must include:

model.safetensors / pytorch_model.bin

tokenizer.json

config.json

vocab.txt

label_map.json

special_tokens_map.json



---

5. Running the Application

Start the Flask server:

cd src
python flask_app.py

Open the app in a browser:

http://127.0.0.1:5000

Paste an abstract → click Classify → get prediction & confidence.


---

6. Training Summary (Model Building)

The project fine-tuned DistilBERT using the following cells:

Cell A: Load dataset, tokenize text, map labels, compute class weights

Cell B: Configure TrainingArguments (batch size, learning rate, epochs, evaluation strategy)

Cell C: Build a custom WeightedTrainer to apply weighted loss

Cell D: Train model, evaluate, save checkpoints, generate reports


Final validation accuracy achieved: ~78%


---

7. Tech Stack

Machine Learning

Hugging Face Transformers

Tokenizers

PyTorch

Evaluate (accuracy, F1)


Backend

Flask

JSON API


Frontend

HTML5

CSS3

Fetch API (JavaScript)



---

8. How to Test the Model

Example input:

This work introduces a transformer-based framework to improve few-shot learning
performance across multiple AI benchmark datasets.

Expected output:

Predicted Field: Artificial Intelligence
Confidence: ~92%


---

9. Notes

Do not push large model files to GitHub.

The repository includes the entire pipeline, UI, and backend but excludes the trained model.

Works on both CPU and GPU (GPU strongly recommended for training).



---

10. License

This project is intended for academic and educational use.
Modify or extend it as needed for your course or research.


---

