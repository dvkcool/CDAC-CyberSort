# **CDAC-CyberSort: Few-Shot Text Classification Using Transformers**

This repository demonstrates training and evaluating separate models for **category** and **subcategory** classification using the Hugging Face Transformers library. The workflow involves preprocessing data, training models, and evaluating them for accuracy, precision, recall, and F1 scores.

## Steps Involved

### 1. Data Cleaning
We cleaned the provided dataset using scripts from the `data_cleaning` folder to ensure consistency and quality.  
The cleaning process involves:
1. Removing blank spaces from the text.
2. Ensuring the complaint length (after space removal) is at least **30 characters**.
3. Mapping complaints to the correct **category** and **subcategory** using the provided mapping flow.

```mermaid
graph TD
  A[Start] --> B[Remove Blank Spaces]
  B --> C[Check Complaint Length >= 30]
  C -->|Yes| D[Map Correct Category and Subcategory]
  C -->|No| E[Discard Complaint]
  D --> F[Cleaned Dataset]
  E --> F
```

---

### 2. Data Preprocessing
We used the `dataArranger.py` script to create two separate datasets for further analysis:
- **`category.csv`**: Contains the complaint text and its corresponding category.
- **`subcategory.csv`**: Contains the complaint text and its corresponding subcategory.

```mermaid
graph TD
  A[Start] --> B[Run dataArranger.py]
  B --> C[Split Dataset into Two Files]
  C --> D[category.csv]
  C --> E[subcategory.csv]
  D --> F[Text & Category]
  E --> G[Text & Subcategory]
```

---

### 3. ML Model Selection
Several models were evaluated for categorization tasks:
1. **Random Forest**  
2. **Complete Random Forest**  
3. **IFDF (Improved Feature Decision Forest)**  

Each of these models achieved approximately **50% accuracy**.  Detailed report can be found in the Reports folder.
A **few-shot classification** approach using **Facebook's BART** model yielded a much better accuracy of **90%**.

```mermaid
graph TD
  A[Start] --> B[Model Comparison]
  B --> C[Random Forest ~50% Accuracy]
  B --> D[Complete Random Forest ~50% Accuracy]
  B --> E[IFDF ~50% Accuracy]
  B --> F[Facebook BART Few-Shot Classification]
  F --> G[90% Accuracy Achieved]
  C --> H[Low Accuracy]
  D --> H
  E --> H
  H --> F
```

---

## Folder Structure
```
├── Data Cleaning
│   ├── cleaning.py
│   ├── mapping.py
│   ├── merge.py
│   ├── split.py
│   └── unique.py
├── FewshotsTraining
│   ├── category.csv
│   ├── dataArranger.py
│   ├── results
│   ├── subcategory.csv
│   └── trainer.py
├── GGUF_Model
│   ├── buildfile.sh
│   ├── chat.py
│   ├── Dockerfile
│   └── model
├── LICENSE
├── ModelFiles
│   ├── a.modelfile
│   ├── b.modelfile
│   └── original.modelfile
├── PythonTester
│   ├── bert-category-classifier
│   ├── bertClassifier.ipynb
│   ├── bertii.py
│   ├── bert.py
│   ├── categorized_complaints.csv
│   ├── categorized_complaintsiii.csv
│   ├── category_model.pkl
│   ├── cleaned_dataset.csv
│   ├── cyber_complaint_classifier.pkl
│   ├── ifidf.py
│   ├── ollamaiii.py
│   ├── ollamaii.py
│   ├── ollama.py
│   ├── outputs.csv
│   ├── output.txt
│   ├── prompt.txt
│   ├── randomforest.py
│   ├── requirements.txt
│   ├── subcategory_model.pkl
│   ├── test_data_cl.csv
│   ├── test_data.csv
│   ├── test_data_up.csv
│   ├── tester.py
│   ├── tfidf_vectorizer.pkl
│   └── venv
├── README.md
└── Reports
    ├── Bert.md
    ├── completeRandomForest.md
    └── randomForest.md
```

--- 


## **ML Training Workflow Overview**

The project is structured to:
1. Load datasets (`category.csv` and `subcategory.csv`).
2. Preprocess data for tokenization.
3. Train separate models for category and subcategory classification.
4. Evaluate the trained models using metrics.
5. Predict category and subcategory labels for unseen data.

### **Flowchart of few shots training**

```mermaid
flowchart TD
    A[Load Data] --> B[Split into Training and Evaluation Sets]
    B --> C[Preprocess Text Data]
    C --> D[Train Category Model]
    C --> E[Train Subcategory Model]
    D --> F[Evaluate Category Model]
    E --> G[Evaluate Subcategory Model]
    F --> H[Calculate Metrics Accuracy, Precision, Recall, F1]
    G --> H
    H --> I[Predict Labels for New Data]
```

---

## **Prerequisites**

Ensure you have the following installed:
- Python 3.8+
- Transformers (`pip install transformers`)
- Datasets (`pip install datasets`)
- scikit-learn (`pip install scikit-learn`)
- Pandas (`pip install pandas`)

---

## **Data Format**

### **Input Files**
1. **`category.csv`**
2. **`subcategory.csv`**

Each file should have the following headers:
- `text`: The text data to classify.
- `label`: The corresponding label for the text.

Example:

| text                            | label            |
|---------------------------------|------------------|
| "My credit card was hacked."   | "Financial Fraud" |
| "Unauthorized access detected." | "Other Cyber Crime" |

---

## **Key Scripts**

### 1. **Preprocessing**
- Tokenizes the text data using a tokenizer (e.g., `distilbert-base-uncased`).
- Splits data into training and evaluation sets (80/20 split).
  
### 2. **Training**
- Trains separate models for `category` and `subcategory` classification.
- Uses Hugging Face's `Trainer` API with metrics like accuracy, precision, recall, and F1 score.

### 3. **Evaluation**
- Computes metrics for evaluation datasets.
- Predicts labels for unseen data.

---

## **Usage**

### 1. **Clone the Repository**
```bash
git clone https://github.com/dvkcool/CDAC-CyberSort.git
cd CDAC-CyberSort/FewshotsTraining

```

### 2. **Prepare Data**
Place `category.csv` and `subcategory.csv` in the FewShotsTraining directory as per above instructions.

### 3. **Run the Script**
```bash
python trainer.py
```

---

## **Output**

### **Evaluation Metrics**
Printed metrics for both category and subcategory models:
```plaintext
Category Metrics:
{'accuracy': 0.92, 'precision': 0.91, 'recall': 0.92, 'f1': 0.91}

Subcategory Metrics:
{'accuracy': 0.85, 'precision': 0.84, 'recall': 0.85, 'f1': 0.84}
```



---

## **Future Enhancements**

1. Fine-tune models on larger datasets for better accuracy.
2. Implement hyperparameter tuning for optimal performance.
3. Add visualization for evaluation metrics (e.g., confusion matrix).

---

## **Contributors**
- **Divyanshu Kumar** - [dvkcool](https://github.com/dvkcool)
- **Shingamba Pukhramba** - [shingamba](https://github.com/shingamba)
- **Y Ranjan Singh** 

Feel free to open an issue for suggestions or improvements.

--- 
          