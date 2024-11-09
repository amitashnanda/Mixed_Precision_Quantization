from datasets import load_dataset
from transformers import BertTokenizer, BertModel, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cross_decomposition import CCA

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

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=32,
    collate_fn=data_collator,
)
# Extract layer outputs
def extract_layer_outputs(dataloader, model, device):
    # Ensure the model is set to return hidden states
    model.config.output_hidden_states = True

    model.eval()
    layer_outputs = {f'layer_{i}': [] for i in range(model.config.num_hidden_layers + 1)}
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(device) for key, val in batch.items() if key in ['input_ids', 'attention_mask']}
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

            # Check if hidden_states is None
            if hidden_states is None:
                raise ValueError("Hidden states are None. Check model configuration to ensure that hidden states are enabled.")

            for i, state in enumerate(hidden_states):
                layer_outputs[f'layer_{i}'].append(state.mean(dim=1).cpu().numpy())

    for key in layer_outputs:
        layer_outputs[key] = np.vstack(layer_outputs[key])
    return layer_outputs

# Perform CCA to determine the sensitivity of a specific layer
def compute_layer_sensitivity(layer_outputs, target_layer_index):
    sensitivities = []
    target_layer_key = f'layer_{target_layer_index}'
    cca = CCA(n_components=1)
    target_data = layer_outputs[target_layer_key]
    # print("target_layer_index")
    # print(target_layer_index)
    for key, data in layer_outputs.items():
        if key != target_layer_key:
            cca.fit(target_data, data)
            X_c, Y_c = cca.transform(target_data, data)
            correlation = np.corrcoef(X_c.T, Y_c.T)[0, 1]
            sensitivities.append(correlation)
            print(correlation)

    # Return the average sensitivity for the target layer
    average_sensitivity = 1 - np.mean(sensitivities)  # Lower correlation implies higher sensitivity
    return target_layer_key, average_sensitivity

# Main execution
layer_outputs = extract_layer_outputs(dataloader, model, device)
target_layer_index = 0  # Specify the layer index you are interested in
layer_sensitivitites = {}
num_layers = model.config.num_hidden_layers

while target_layer_index< num_layers:
    layer_key, layer_sensitivity = compute_layer_sensitivity(layer_outputs, target_layer_index)
    print("layer sensitivity")
    print(layer_sensitivity)
    # {'layer_0': 0.1, 'layer_1': 0.5, 'layer_2': 0.9}
    layer_sensitivitites["layer_"+str(target_layer_index)]=layer_sensitivity
    print(f"Sensitivity of {layer_key}: {layer_sensitivity:.3f}")
    target_layer_index+=1
