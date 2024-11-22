import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Load data
category_data = pd.read_csv("category.csv").head(1000)
subcategory_data = pd.read_csv("subcategory.csv").head(1000)

# Split into training and evaluation datasets (80% train, 20% eval)
from sklearn.model_selection import train_test_split

category_train, category_eval = train_test_split(category_data, test_size=0.2, random_state=42)
subcategory_train, subcategory_eval = train_test_split(subcategory_data, test_size=0.2, random_state=42)

# Load tokenizer
model_name = "distilbert-base-uncased"  # Replace with desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

from datasets import Dataset
def prepare_dataset(df):
    # Convert to Hugging Face Dataset and encode labels as integers
    df["label"] = pd.Categorical(df["label"]).codes
    return Dataset.from_pandas(df).map(preprocess_function, batched=True)

# Prepare datasets
category_train_dataset = prepare_dataset(category_train)
category_eval_dataset = prepare_dataset(category_eval)
subcategory_train_dataset = prepare_dataset(subcategory_train)
subcategory_eval_dataset = prepare_dataset(subcategory_eval)

# Define compute metrics function


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.array(logits)  # Convert logits to a NumPy array
    labels = np.array(labels)  # Convert labels to a NumPy array
    predictions = logits.argmax(axis=-1)  # Get the predicted class
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
)

# Train models
def train_model(tokenized_train_dataset, tokenized_eval_dataset, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return model

# Train category model
num_category_labels = len(category_data["label"].unique())
category_model = train_model(category_train_dataset, category_eval_dataset, num_labels=num_category_labels)

# Train subcategory model
num_subcategory_labels = len(subcategory_data["label"].unique())
subcategory_model = train_model(subcategory_train_dataset, subcategory_eval_dataset, num_labels=num_subcategory_labels)

# Prediction Function
def predict(model, tokenizer, texts):
    if isinstance(texts, str):  # If a single string is passed, convert it to a list
        texts = [texts]
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = logits.argmax(axis=-1).detach().cpu().numpy()  # Ensure it outputs an array
    return predictions

# Evaluate Category Model
category_texts = category_eval["text"].tolist()
category_labels = category_eval["label"].astype("category").cat.codes.tolist()
category_predictions = predict(category_model, tokenizer, category_texts)
category_labels = category_eval["label"].astype("category").cat.codes.tolist()

# Ensure predictions and labels are array-like
category_predictions = list(category_predictions)
category_labels = list(category_labels)

accuracy = accuracy_score(category_labels, category_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(category_labels, category_predictions, average="weighted")

print("Category Metrics:")
print({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})

#category_metrics = compute_metrics((category_predictions, category_labels))

# Evaluate Subcategory Model
subcategory_texts = subcategory_eval["text"].tolist()
subcategory_labels = subcategory_eval["label"].astype("category").cat.codes.tolist()
subcategory_predictions = predict(subcategory_model, tokenizer, subcategory_texts)

# Ensure predictions and labels are array-like
subcategory_predictions = list(subcategory_predictions)
subcategory_labels = list(subcategory_labels)

print("Subcategory Metrics:")
sub_accuracy = accuracy_score(subcategory_labels, subcategory_predictions)
sprecision, srecall, sf1, _ = precision_recall_fscore_support(category_labels, category_predictions, average="weighted")
print({"sub_accuracy": accuracy, "precision": sprecision, "recall": srecall, "f1": sf1})


