from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cross_decomposition import CCA
from tqdm import tqdm
import json
# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# Load the mps device
# Load the mps device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"device : {device}")

model.to(device)
model.eval()

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

# Load and tokenize the dataset
dataset = load_dataset("imdb", split='test[:10%]')
dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Extract layer outputs
def extract_layer_outputs(dataloader, model, device):
    model.eval()
    layer_outputs = {f'layer_{i}': [] for i in range(model.config.num_hidden_layers + 1)}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = {key: val.to(device) for key, val in batch.items() if key in ['input_ids', 'attention_mask']}
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
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
            # print(correlation)

    # Return the average sensitivity for the target layer
    average_sensitivity = 1 - np.mean(sensitivities)  # Lower correlation implies higher sensitivity
    return target_layer_key, average_sensitivity

# Main execution
layer_outputs = extract_layer_outputs(dataloader, model, device)
target_layer_index = 0  # Specify the layer index you are interested in
layer_sensitivitites = {}
num_layers = model.config.num_hidden_layers

# Use a for loop to iterate over the layer indices
for target_layer_index in tqdm(range(num_layers)):
    layer_key, layer_sensitivity = compute_layer_sensitivity(layer_outputs, target_layer_index)
    layer_sensitivitites["layer_"+str(target_layer_index)]=layer_sensitivity
    print(f"Sensitivity of {layer_key}: {layer_sensitivity:.3f}")

# save the layer sensitivitites list to a json file
with open('saved_data/layer_sensitivitites_bert_large_uncased.json', 'w') as f:
    json.dump(layer_sensitivitites, f)

print("Saved the layer sensitivitites to layer_sensitivitites_bert_large_uncased.json")