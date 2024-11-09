import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# Load the tokenizer and model
CHECKPOINT = "dmis-lab/biobert-v1.1"
model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT, num_labels=2
).to(device)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
# tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')
# model = BertModel.from_pretrained('bert-large-cased-whole-word-masking', output_hidden_states=True)
model.to(device)
model.eval()
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

#canonical_analysis_sensitivites_bert_large_cased_whole_word_masking
raw_datasets = load_dataset("imdb")
raw_datasets = raw_datasets.shuffle(seed=42)

# remove unnecessary data split
del raw_datasets["unsupervised"]
train_population = range(len(raw_datasets["train"]))
test_population = range(len(raw_datasets["test"]))

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets["train"] = tokenized_datasets["train"].select(train_population)
tokenized_datasets["test"] = tokenized_datasets["test"].select(test_population)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns("text")
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
trainloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=32,
    collate_fn=data_collator,
)
dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=32,
    collate_fn=data_collator,
)

testloader = DataLoader(
    tokenized_datasets["test"], batch_size=32, collate_fn=data_collator
)

def taylor_sensitivity(model, dataloader, device):
    sensitivities = []
    model.eval()
    for index, layer in enumerate(model.bert.encoder.layer):
        # Store original state
        original_state_dict = {name: param.clone() for name, param in layer.named_parameters()}

        # Generate perturbation and apply it to the first parameter
        first_param_name, first_param = next(iter(layer.named_parameters()))
        perturbation = torch.randn_like(first_param, device=device)
        first_param.data.add_(perturbation)  # Apply perturbation in-place

        total_variation = 0.0
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss if outputs.loss is not None else outputs.logits.sum()
            total_variation += loss.item()

        # Restore original layer state
        layer.load_state_dict(original_state_dict)

        # Calculate normalized variation
        normalized_variation = total_variation / len(dataloader.dataset)
        sensitivities.append((index, normalized_variation))

    return sensitivities
sensitivity_scores = taylor_sensitivity(model, dataloader, device)
layer_sensitivitites = {}
for layer_idx, sensitivity in sensitivity_scores:
    print(f"Layer {layer_idx}: Sensitivity = {sensitivity}")
    layer_sensitivitites["layer_"+str(layer_idx)]=sensitivity
