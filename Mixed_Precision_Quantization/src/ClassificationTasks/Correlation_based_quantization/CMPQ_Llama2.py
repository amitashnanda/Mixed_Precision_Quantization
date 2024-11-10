from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
accelerator = Accelerator()
import torch
from huggingface_hub import login
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cross_decomposition import CCA

from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

# Get the token from an environment variable
token = os.getenv('HF_TOKEN')
if token:
    login(token=token)
else:
    print("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

model_id = 'meta-llama/Llama-2-7b-chat-hf'

if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map='auto',
        output_hidden_states=True
    )
else:
    model = None
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model, tokenizer = accelerator.prepare(model, tokenizer)

model.eval()

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

dataset = load_dataset("imdb", split='test[:10%]')
dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
dataloader = accelerator.prepare(dataloader)
def extract_layer_outputs(dataloader, model):
    model.eval()
    layer_outputs = {f'layer_{i}': [] for i in range(model.config.num_hidden_layers + 1)}
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(accelerator.device) for key, val in batch.items() if key in ['input_ids', 'attention_mask']}
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            for i, state in enumerate(hidden_states):
                layer_outputs[f'layer_{i}'].append(accelerator.gather(state).mean(dim=1).cpu().numpy())
                print(layer_outputs)
    for key in layer_outputs:
        layer_outputs[key] = np.vstack(layer_outputs[key])
    return layer_outputs

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
layer_outputs = extract_layer_outputs(dataloader, model)
target_layer_index = 0  # Specify the layer index you are interested in
layer_sensitivitites = {}
num_layers = model.config.num_hidden_layers

print("testttt")

while target_layer_index< num_layers:
    print("enterrrrr")
    layer_key, layer_sensitivity = compute_layer_sensitivity(layer_outputs, target_layer_index)
    print("layer sensitivity")
    print(layer_sensitivity)
    # {'layer_0': 0.1, 'layer_1': 0.5, 'layer_2': 0.9}
    layer_sensitivitites["layer_"+str(target_layer_index)]=layer_sensitivity
    print(f"Sensitivity of {layer_key}: {layer_sensitivity:.3f}")
    target_layer_index+=1
