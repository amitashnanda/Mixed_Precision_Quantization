import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, OPTForCausalLM
# Load model and tokenizer

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

# Load the tokenizer and model
model_id = "facebook/opt-1.3b"  # Change this to 'facebook/opt-2.7b' for OPT-2
if torch.cuda.is_available():
    model = OPTForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map='auto',
        output_hidden_states=True
    )
else:
    model = None
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model, tokenizer = accelerator.prepare(model, tokenizer)

# Load IMDB dataset
dataset = load_dataset('imdb')
test_dataset = dataset['test'].map(
    lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=512), batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_loader = DataLoader(test_dataset, batch_size=16)
test_loader = accelerator.prepare(test_loader)

# Function to introduce sparsity to a given layer
def apply_sparsity_to_layer(layer, sparsity_level):
    with torch.no_grad():
        for param in layer.parameters():
            flat_param = param.data.view(-1)
            threshold = torch.quantile(torch.abs(flat_param), sparsity_level)
            mask = torch.abs(flat_param) > threshold
            param.data *= mask.float().view(param.data.shape)

# Function to evaluate the model's accuracy
def evaluate_model(model):
    model.eval()
    total_accuracy = 0
    for batch in tqdm(test_loader, desc="Evaluating"):
        inputs = {k: v.to(accelerator.device) for k, v in batch.items() if k != 'label'}
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            total_accuracy += (predictions == batch['label'].to(accelerator.device)).float().mean().item()
    return total_accuracy / len(test_loader)

# Main loop to apply sparsity and evaluate
layer_sensitivity = {}
base_accuracy = evaluate_model(model)
sparsity_levels = [0.3]
for s in sparsity_levels:
    sensitivities = []
    for layer_num, layer in enumerate(model.model.decoder.layers):
        layer_name = f"Encoder Layer {layer_num}"
        # Evaluate with increasing sparsity levels
            # Clone the original model for each experiment
        sparse_model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=2).to(accelerator.device)
        sparse_model.load_state_dict(model.state_dict())
        # Apply sparsity
        apply_sparsity_to_layer(sparse_model.model.decoder.layers[layer_num], s)
        # Evaluate
        accuracy = evaluate_model(sparse_model)
        sensitivity = base_accuracy - accuracy
        sensitivities.append(sensitivity)
        # Clean up to free GPU memory
        del sparse_model
        torch.cuda.empty_cache()
        layer_sensitivity[layer_name] = sensitivities
    for layer, sensitivity in layer_sensitivity.items():
        print(f"{layer}: {sensitivity}")

