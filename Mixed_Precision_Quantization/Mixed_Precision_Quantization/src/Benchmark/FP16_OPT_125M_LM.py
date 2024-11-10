import torch
import os
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim import Adam
from transformers import AutoTokenizer, OPTForCausalLM
from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler

# Function to save the model to disk and calculate its size in bytes
def get_model_size(model, filename='temp_model.pt'):
    torch.save(model.state_dict(), filename)
    model_size = os.path.getsize(filename)
    os.remove(filename)  # Clean up after size is measured
    return model_size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# Load tokenizer and model
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = OPTForCausalLM.from_pretrained(model_name)
# Load the dataset
dataset = load_dataset("imdb")
###### for imdbp
dataset = dataset.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128), batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
###### for imdb
train_dataloader = DataLoader(dataset['train'], batch_size=16, shuffle=True)
test_dataloader = DataLoader(dataset['test'], batch_size=16, shuffle=False)
optimizer = optim.SGD(model.parameters(), lr=0.001)


# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# Function to apply quantization to the model
def quantize_to_16bit(model):
    model.eval()
    with torch.no_grad():
        for param in model.parameters():
            param.data = param.data.to(torch.int8)
# def train_eval_model(model, train_loader, test_loader):
#     criterion = CrossEntropyLoss()
#     model.train()
#     for batch in train_loader:
#         inputs, masks, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
#         optimizer.zero_grad()
#         # Forward pass
#         outputs = model(inputs, attention_mask=masks)
#         logits = outputs.logits

#         # If your task involves predicting a label for each token in the sequence:
#         # Flatten the logits and labels to be compatible with CrossEntropyLoss
#         # Here, we ensure labels are also a sequence of the same length
#         # Assuming labels are already correctly aligned in your dataset
#         logits_flat = logits.view(-1, logits.size(-1))
#         labels_flat = labels.view(-1)

#         # Calculate loss
#         loss = criterion(logits_flat, labels_flat)
#         loss.backward()
#         optimizer.step()

#     model.eval()
#     total = 0
#     correct = 0
#     with torch.no_grad():
#         for batch in test_loader:
#             inputs, masks, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

#             outputs = model(inputs, attention_mask=masks)
#             logits = outputs.logits

#             # Flatten for evaluation
#             logits_flat = logits.view(-1, logits.size(-1))
#             labels_flat = labels.view(-1)

#             predictions = torch.argmax(logits_flat, dim=1)
#             correct += (predictions == labels_flat).sum().item()
#             total += labels_flat.size(0)

#     accuracy = correct / total
#     return accuracy

def train_eval_model(model, train_loader, test_loader, device, optimizer):
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device).train()

    for batch in train_loader:
        inputs, masks, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, attention_mask=masks)
        logits = outputs.logits

        # Correctly flattening logits and labels
        # Check if you need to adjust sequence lengths
        if logits.dim() > 2:  # logits typically [batch_size, seq_length, num_classes]
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
        else:
            logits_flat = logits
            labels_flat = labels

        # Ensure that logits and labels dimensions are aligned
        if logits_flat.size(0) != labels_flat.size(0):
            raise ValueError(f"Flattened logits and labels have mismatched sizes: {logits_flat.size(0)} vs {labels_flat.size(0)}")

        # Calculate loss
        loss = criterion(logits_flat, labels_flat)
        loss.backward()
        optimizer.step()

    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, masks, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

            outputs = model(inputs, attention_mask=masks)
            logits = outputs.logits

            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            predictions = torch.argmax(logits_flat, dim=1)
            correct += (predictions == labels_flat).sum().item()
            total += labels_flat.size(0)

    accuracy = correct / total
    return accuracy


# Training and evaluation function
# def train_eval_model(model, train_loader, test_loader, quantized=False):
#     optimizer = Adam(model.parameters(), lr=2e-5)
#     criterion = CrossEntropyLoss()
#     scaler = GradScaler(enabled=not quantized)  # Disable scaler if model is quantized

#     if quantized:
#         original_size = get_model_size(model)
#         quantize_to_16bit(model)
#         quantized_size = get_model_size(model)
#         compression_ratio = original_size / quantized_size
#     else:
#         compression_ratio = 1  # No compression if not quantized
    # Training loop
    # model.train()
    # for epoch in range(1):  # Training for one epoch only for demonstration
    #     for batch in train_loader:
    #         inputs, masks, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
    #         optimizer.zero_grad()
    #         outputs = model(inputs, attention_mask=masks)
    #         logits = outputs.logits
    #         # Aligning logits and labels
    #         # Removing the last token's logits because there's no next token to predict
    #         shift_logits = logits[..., :-1, :].contiguous()
    #         # Labels should be shifted to align with logits' prediction
    #         shift_labels = labels[..., 1:].contiguous()
    #         # Flatten the logits and labels
    #         # This will align each prediction with each label in a long list
    #         logits_flat = shift_logits.view(-1, shift_logits.size(-1))
    #         labels_flat = shift_labels.view(-1)
    #         loss = criterion(logits_flat, labels_flat)
    #         if quantized:
    #             loss.backward()
    #             optimizer.step()
    #         else:
    #             scaler.scale(loss).backward()
    #             scaler.step(optimizer)
    #             scaler.update()

    # # Evaluation loop
    # model.eval()
    # correct = 0
    # total = 0
    # with torch.no_grad(), autocast(enabled=not quantized):
    #     for batch in test_loader:
    #         inputs, masks, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
    #         outputs = model(inputs, attention_mask=masks)
    #         predictions = torch.argmax(outputs.logits, dim=1)
    #         correct += (predictions == labels).sum().item()
    #         total += labels.size(0)

    # accuracy = correct / total
    # return accuracy, compression_ratio

# Evaluate before quantization
accuracy_before, _ = train_eval_model(model, train_dataloader, test_dataloader, device, optimizer)

# Evaluate after quantization
accuracy_after, compression_ratio = train_eval_model(model, train_dataloader, test_dataloader, quantized=True)

# Print the reduction in accuracy and the compression ratio
print(f"Accuracy before quantization: {accuracy_before}")
print(f"Accuracy after quantization: {accuracy_after}")
print(f"Reduction in accuracy: {accuracy_before - accuracy_after}")
print(f"Compression Ratio: {compression_ratio:.2f}")
