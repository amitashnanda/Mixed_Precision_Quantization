import torch
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cross_decomposition import CCA

# Load model and tokenizer
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# Load the tokenizer and model
CHECKPOINT = "dmis-lab/biobert-v1.1"
model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT, num_labels=2
).to(device)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model.to(device)
model.eval()
#canonical_analysis_sensitivites_bert_large_cased_whole_word_masking
raw_datasets = load_dataset("imdb")
raw_datasets = raw_datasets.shuffle(seed=42)

# remove unnecessary data split
del raw_datasets["unsupervised"]


# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

# Load and tokenize the dataset
dataset = load_dataset("imdb", split='test[:10%]')
dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Create DataLoader
train_population = range(len(raw_datasets["train"]))
test_population = range(len(raw_datasets["test"]))

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets["train"] = tokenized_datasets["train"].select(train_population)
tokenized_datasets["test"] = tokenized_datasets["test"].select(test_population)

tokenized_datasets = tokenized_datasets.remove_columns("text")
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
trainloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=32,
    collate_fn=data_collator,
)

testloader = DataLoader(
    tokenized_datasets["test"], batch_size=32, collate_fn=data_collator
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=32,
    collate_fn=data_collator,
)

# Function to introduce sparsity to a given layer
def apply_sparsity_to_layer(layer, sparsity_level):
    with torch.no_grad():
        for param in layer.parameters():
            flat_param = param.data.view(-1)
            threshold = torch.quantile(torch.abs(flat_param), sparsity_level)
            mask = torch.abs(flat_param) > threshold
            param.data *= mask.float().view(param.data.shape)

# Function to evaluate the model's accuracy
def evaluate_model(model, tokenizer, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy

# Main loop to apply sparsity and evaluate
layer_sensitivity = {}
base_accuracy = evaluate_model(model, tokenizer, testloader, device)
print(f"Base Accuracy: {base_accuracy}")
sparsity_levels = [0.3]
for s in sparsity_levels:
    print("Sparsity level is "+ str(s))
    sensitivities = []
    for layer_num, layer in enumerate(model.bert.encoder.layer):
        layer_name = f"Encoder Layer {layer_num}"
        print(f"Evaluating {layer_name}")
        # Evaluate with increasing sparsity levels
            # Clone the original model for each experiment
        sparse_model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2).to(device)
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        sparse_model.load_state_dict(model.state_dict())

        # Apply sparsity
        apply_sparsity_to_layer(sparse_model.bert.encoder.layer[layer_num], s)

        # Evaluate
        accuracy = evaluate_model(sparse_model, tokenizer, testloader, device)
        sensitivity = base_accuracy - accuracy
        sensitivities.append(sensitivity)
#         print(f"Sparsity Level: {s}, Accuracy: {accuracy}, Sensitivity: {sensitivity}")
        # Clean up to free GPU memory
        del sparse_model
        torch.cuda.empty_cache()
    # layer_sensitivity[layer_name] = sensitivities
    # print("Layer Sensitivity Analysis Completed:")
    # for layer, sensitivity in layer_sensitivity.items():
    #     print(f"{layer}: {sensitivity}")
layer_sensitivitites = {}
layer_idx =0
for sensitivity in sensitivities:
    layer_sensitivitites["layer_"+str(layer_idx)] = sensitivity
    layer_idx+=1

