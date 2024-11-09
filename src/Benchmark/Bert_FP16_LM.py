import torch
import os
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler
def calculate_perplexity(loss):
    loss_tensor = torch.tensor(loss)  # convert float to tensor
    return torch.exp(loss_tensor).item()  # apply exp and convert back to float
# Function to save the model to disk and calculate its size in bytes
def get_model_size(model, filename='temp_model.pt'):
    torch.save(model.state_dict(), filename)
    model_size = os.path.getsize(filename)
    os.remove(filename)  # Clean up after size is measured
    return model_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForSequenceClassification.from_pretrained('bert-large-uncased')
model.to(device)

# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# Load and preprocess the dataset
dataset = load_dataset("imdb")
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
train_loader = DataLoader(dataset['train'], batch_size=8, shuffle=True)
test_loader = DataLoader(dataset['test'], batch_size=8)

# Function to apply quantization to the model
def quantize_to_16bit(model):
    model.eval()
    with torch.no_grad():
        for param in model.parameters():
            param.data = param.data.to(torch.float16)

# Training and evaluation function
def train_eval_model(model, train_loader, test_loader, quantized=False):
    optimizer = Adam(model.parameters(), lr=2e-5)
    criterion = CrossEntropyLoss()
    scaler = GradScaler(enabled=not quantized)  # Disable scaler if model is quantized

    if quantized:
        original_size = get_model_size(model)
        quantize_to_16bit(model)
        quantized_size = get_model_size(model)
        compression_ratio = original_size / quantized_size
    else:
        compression_ratio = 1  # No compression if not quantized

    # Training loop
    model.train()
    for epoch in range(1):  # Training for one epoch only for demonstration
        for batch in train_loader:
            inputs, masks, labels = batch['input_ids'].cuda(), batch['attention_mask'].cuda(), batch['labels'].cuda()
            optimizer.zero_grad()

            with autocast(enabled=not quantized):
                outputs = model(inputs, attention_mask=masks, labels=labels)
                loss = criterion(outputs.logits, labels)

            if quantized:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

    # Evaluation loop
    model.eval()
    correct = 0
    total_loss = 0
    total = 0
    total_prep = 0
    with torch.no_grad(), autocast(enabled=not quantized):
        for batch in test_loader:
            inputs, masks, labels = batch['input_ids'].cuda(), batch['attention_mask'].cuda(), batch['labels'].cuda()
            outputs = model(inputs, attention_mask=masks)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item() * inputs.size(0)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_prep+=inputs.size(0)

    accuracy = correct / total
    average_loss = total_loss / total_prep
    perplexity = calculate_perplexity(average_loss)
    return accuracy, compression_ratio, perplexity

# Evaluate before quantization
accuracy_before, _, perplexity_before = train_eval_model(model, train_loader, test_loader, quantized=False)

# Evaluate after quantization
accuracy_after, compression_ratio, perplexity_after = train_eval_model(model, train_loader, test_loader, quantized=True)

# Print the reduction in accuracy and the compression ratio
print(f"Accuracy before quantization: {accuracy_before}")
print(f"Accuracy after quantization: {accuracy_after}")
print(f"Reduction in accuracy: {accuracy_before - accuracy_after}")
print(f"Compression Ratio: {compression_ratio:.2f}")
print(f"Reduction in Perplexity: { -perplexity_before+perplexity_after}")