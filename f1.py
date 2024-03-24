import pandas as pd
from datasets import Dataset
from transformers import pipeline, AutoTokenizer
from sklearn.metrics import accuracy_score

# Upload your CSV file
uploaded_file = "ggggg.csv"

# Read the CSV file using pandas
df = pd.read_csv(uploaded_file,encoding='cp1252')

# Convert your pandas DataFrame to a Hugging Face dataset
hf_dataset = Dataset.from_pandas(df)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("knowledgator/comprehend_it-base")
classifier = pipeline("zero-shot-classification", model="knowledgator/comprehend_it-base")

length=len(df.label)
labels=[]
for i in range(length):
  labels.append(df.label[i])

# Tokenize your dataset
tokenized_dataset = tokenizer(hf_dataset["text"], padding=True, truncation=True, return_tensors="pt")

# Perform inference
dic={}
fin=[]
for i in range(length):
  dic= classifier(df.text[i], labels)
  fin.append(dic['labels'][0])
#print(fin)
from sklearn.metrics import accuracy_score, f1_score
predicted_labels = fin

# Ground truth labels for evaluation (assuming your DataFrame has a column named 'labels' containing ground truth labels)
ground_truth_labels = df['label']

# Evaluate model accuracy
accuracy = accuracy_score(ground_truth_labels, predicted_labels)
print("Model Accuracy:", accuracy)
f1 = f1_score(ground_truth_labels, predicted_labels, average='weighted')
print("F1 Score:", f1)
