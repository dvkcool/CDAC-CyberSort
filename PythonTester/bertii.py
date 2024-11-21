import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset

# Load data
# data = pd.read_csv('/content/drive/MyDrive/Hackathon/cleaned_dataset.csv')  # Update with the path to your CSV file
data = pd.read_csv('cleaned_dataset.csv')
# Encode category and subcategory labels as numeric codes
data['category_label'] = data['true_category'].astype('category').cat.codes
data['subcategory_label'] = data['true_subcategory'].astype('category').cat.codes

# Split data for both category and subcategory
train_texts, val_texts, train_category_labels, val_category_labels = train_test_split(
    data['complaint'], data['category_label'], test_size=0.2, random_state=42
)
_, _, train_subcategory_labels, val_subcategory_labels = train_test_split(
    data['complaint'], data['subcategory_label'], test_size=0.2, random_state=42
)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the complaint texts
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# Dataset class for category and subcategory datasets
class CyberComplaintDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create separate datasets for category and subcategory
category_train_dataset = CyberComplaintDataset(train_encodings, train_category_labels.to_list())
category_val_dataset = CyberComplaintDataset(val_encodings, val_category_labels.to_list())

subcategory_train_dataset = CyberComplaintDataset(train_encodings, train_subcategory_labels.to_list())
subcategory_val_dataset = CyberComplaintDataset(val_encodings, val_subcategory_labels.to_list())

# Compute metrics function for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Function to train and save a model, and print accuracy
def train_model(output_dir, train_dataset, val_dataset, num_labels):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to="none",
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"Model trained and saved at '{output_dir}'")
    print(f"Final accuracy for {output_dir}: {metrics['eval_accuracy']:.4f}")

    trainer.save_model(output_dir)
    return model

# Train and save the category model
num_category_labels = data['category_label'].nunique()
category_model = train_model("bert-category-classifier", category_train_dataset, category_val_dataset, num_category_labels)

# Train and save the subcategory model
num_subcategory_labels = data['subcategory_label'].nunique()
subcategory_model = train_model("bert-subcategory-classifier", subcategory_train_dataset, subcategory_val_dataset, num_subcategory_labels)

# Load models for inference
category_model = BertForSequenceClassification.from_pretrained("bert-category-classifier")
subcategory_model = BertForSequenceClassification.from_pretrained("bert-subcategory-classifier")
tokenizer = BertTokenizer.from_pretrained("bert-category-classifier")  # Tokenizer can be the same

# Inference function
def predict_category_and_subcategory(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Predict category
    category_outputs = category_model(**inputs)
    category_prediction = category_outputs.logits.argmax(dim=-1).item()

    # Predict subcategory
    subcategory_outputs = subcategory_model(**inputs)
    subcategory_prediction = subcategory_outputs.logits.argmax(dim=-1).item()

    return category_prediction, subcategory_prediction

# Example usage
text = "I lost access to my social media account."
category, subcategory = predict_category_and_subcategory(text)
print("Predicted Category:", category)
print("Predicted Subcategory:", subcategory)
