import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
accelerator = Accelerator()

# Load the tokenizer and model
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()


# Load the tokenizer and model
model_id = "facebook/opt-1.3b"  # Change this to 'facebook/opt-2.7b' for OPT-2
if torch.cuda.is_available():
    model = AutoModelForSequenceClassification.from_pretrained(
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
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

# Load and tokenize the dataset
dataset = load_dataset("imdb", split='test[:10%]')
dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load IMDB dataset
dataset = load_dataset('imdb')
test_dataset = dataset['test'].map(
    lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=512), batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_loader = DataLoader(test_dataset, batch_size=16)
def taylor_sensitivity(model, dataloader):
    sensitivities = []
    model.eval()
    for index, layer in enumerate(model.model.decoder.layers):  # Adjusted to access the correct decoder attribute
        # Store original state
        original_state_dict = {name: param.clone() for name, param in layer.named_parameters()}
        # Generate perturbation and apply it to the first parameter
        first_param_name, first_param = next(iter(layer.named_parameters()))
        perturbation = torch.randn_like(first_param, device=accelerator.device)
        first_param.data.add_(perturbation)  # Apply perturbation in-place

        total_variation = 0.0
        for batch in dataloader:
            inputs = {k: v.to(accelerator.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            with torch.no_grad():
                outputs = model(**inputs)
                if 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    logits = outputs['logits']
                    # Assuming you have labels to calculate loss since they are not shown how to be processed here.
                    labels = batch['label'].to(accelerator.device)  # Make sure labels are included in the batch
                    loss = torch.nn.functional.cross_entropy(logits, labels)
            total_variation += loss.item()

        # Restore original layer state
        layer.load_state_dict(original_state_dict)
        # Calculate normalized variation
        normalized_variation = total_variation / len(dataloader.dataset)
        sensitivities.append((index, normalized_variation))
    return sensitivities
sensitivity_scores = taylor_sensitivity(model, dataloader)
layer_sensitivitites = {}
for layer_idx, sensitivity in sensitivity_scores:
    print(f"Layer {layer_idx}: Sensitivity = {sensitivity}")
    layer_sensitivitites["layer_"+str(layer_idx)]=sensitivity
