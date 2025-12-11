


---

Academic Abstract Classifier

A full-stack Machine Learning project that classifies research abstracts into four major academic fields:

Artificial Intelligence

Business Research

Healthcare Research

Environmental Science


This project includes:

A custom-trained DistilBERT model (Transformer-based)

A Flask backend API

A clean, modern HTML/CSS frontend

A complete dataset pipeline using ArXiv API

Fully reproducible training code with tokenization, class-balancing, weighted loss, evaluation, and inference



---

🚀 Features

Classifies any academic abstract into 1 of 4 categories

Modern UI with confidence bar

Flask API backend that loads model locally

Custom tokenizer + model from Hugging Face Transformers

Trained on 8000+ ArXiv abstracts

Balanced training using class weights

Evaluation metrics: Accuracy, F1-Score, Confusion Matrix



---

📁 Project Structure

Academic-Classifier/
│
├── models/
│     └── abstract_classifier/     ← Your trained model folder (not uploaded to GitHub)
│
├── src/
│     ├── flask_app.py             ← Flask backend API
│     ├── infer_local.py           ← Local testing script
│     └── __init__.py
│
├── templates/
│     └── index.html               ← Frontend HTML
│
├── static/
│     └── style.css                ← Frontend CSS
│
├── requirements.txt               ← Python dependencies
└── README.md


---

📦 Installation

1️⃣ Clone the repository

git clone https://github.com/yourusername/Academic-Classifier.git
cd Academic-Classifier

2️⃣ Create Virtual Environment

python -m venv .venv
source .venv/Scripts/activate        # Windows
# OR
source .venv/bin/activate           # Mac/Linux

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Add Your Model

Place your trained model folder here:

Academic-Classifier/models/abstract_classifier/

Must contain:

config.json

model.safetensors

tokenizer.json

tokenizer_config.json

vocab.txt

special_tokens_map.json

label_map.json



---

⚙️ Running the Backend (Flask API)

cd src
python flask_app.py

Server starts at:

http://127.0.0.1:5000/


---

🖥️ Frontend Usage

The frontend contains:

A banner section

A text-area to paste abstracts

A "Classify" button

A "Clear" button

A confidence progress bar

Clean gradient background


Just open:

http://localhost:5000

Paste an academic abstract → click Classify → result appears instantly.


---

📘 Dataset Collection

Dataset is collected using the ArXiv API:

2000 abstracts for AI

2000 for Healthcare

2000 for Business

2000 for Environmental Science


All combined into:

arxiv_combined_8000.csv

Each record contains:

title

abstract

categories

field (label)



---

🔧 Model Training Pipeline

Training involves four main stages:

1️⃣ Tokenization & Label Encoding

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized = ds.map(tokenize_fn)

2️⃣ Model Initialization

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels)
)

3️⃣ Weighted Training (Handling Class Imbalance)

loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)

4️⃣ Evaluation

Metrics used:

Accuracy

Macro F1 Score

Confusion Matrix



---

📊 Model Results

Metric	Score

Validation Accuracy	~78%
Macro F1-score	~0.78
Train Loss	~0.43


The model performs reliably across all four academic domains.


---

🧪 Local Testing Script

Run inference without UI:

python src/infer_local.py


---

🧠 Example Test Input

This research proposes a transformer-based deep learning method for improving computer vision tasks such as object detection and semantic segmentation.

Output:

Predicted Field: Artificial Intelligence
Confidence: 92.7%


---

📌 Future Improvements

Deploy on Hugging Face Spaces / Render / AWS

Add more categories (Physics, Finance, Biology, etc.)

Improve accuracy with RoBERTa / BERT-large

Add dataset visualization dashboard



---

📄 License

This project is for academic and educational use.


---

🤝 Contributing

Pull requests and improvements are welcome!


---

