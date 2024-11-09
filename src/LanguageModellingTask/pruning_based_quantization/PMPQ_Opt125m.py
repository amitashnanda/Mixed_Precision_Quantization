import torch
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import OPTForCausalLM, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
import torch.nn as nn
from sklearn.cluster import KMeans
from tqdm import tqdm

# Load model and tokenizer
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = OPTForCausalLM.from_pretrained(model_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
# Load IMDB dataset
def encode(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

# Load and tokenize the Wikitext dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Create DataLoader
dataloader = DataLoader(dataset['train'], batch_size=1, shuffle=True)  # Assuming using 'train' split

# Function to introduce sparsity to a given layer
def apply_sparsity_to_layer(layer, sparsity_level):
    with torch.no_grad():
        for param in layer.parameters():
            flat_param = param.data.view(-1)
            threshold = torch.quantile(torch.abs(flat_param), sparsity_level)
            mask = torch.abs(flat_param) > threshold
            param.data *= mask.float().view(param.data.shape)

# Function to evaluate the model's accuracy
# def evaluate_model(model):
import torch
from torch import nn
from tqdm import tqdm
from datasets import load_dataset

def evaluate_model(model, tokenizer):
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    # Load and prepare dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    data = data.input_ids.to(model.device)
    seqlen = 512
    model = model.eval()
    n_samples = data.numel() // seqlen
    nlls = []

    with tqdm(range(n_samples), desc="Perplexity Evaluation") as progress_bar:
        for i in progress_bar:
            start_index = i * seqlen
            end_index = (i + 1) * seqlen
            batch = data[:, start_index:end_index].to(model.device)
            with torch.no_grad():
                output = model(batch)
                logits = output.logits if hasattr(output, 'logits') else output[0]

            # Ensure logits are 3-dimensional
            if logits.dim() == 2:  # Model output is (batch_size, num_classes)
                logits = logits.unsqueeze(1)  # Add sequence length dimension

            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = data[:, start_index:end_index][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

            curr_ppl = _perplexity(nlls, i + 1, seqlen)
            progress_bar.set_description(f"Perplexity {curr_ppl:.3f}")

    ppl = _perplexity(nlls, n_samples, seqlen)
    return ppl.item()


def load_dataset_and_tokenizer(dataset_name, model_name, max_length=128):
    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(dataset_name, 'wikitext-2-raw-v1')

    # Tokenize the data
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return tokenized_datasets, tokenizer
def create_data_loader(tokenized_datasets, tokenizer):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, mlm_probability=0.15)
    dataloader = DataLoader(tokenized_datasets['test'], batch_size=4, collate_fn=data_collator)
    return dataloader

dataset_name = 'wikitext'
model_name = "facebook/opt-125m"
tokenized_datasets, tokenizer = load_dataset_and_tokenizer(dataset_name, model_name)
dataloader = create_data_loader(tokenized_datasets, tokenizer)
model = OPTForCausalLM.from_pretrained(model_name)
# Main loop to apply sparsity and evaluate
layer_sensitivity = {}
base_accuracy = evaluate_model(model,tokenizer)
print(f"Base Accuracy: {base_accuracy}")
sparsity_levels = [0.3]
for s in sparsity_levels:
    print("Sparsity level is "+ str(s))
    sensitivities = []
    print(model)
    for layer_num, layer in enumerate(model.model.decoder.layers):
        layer_name = f"Encoder Layer {layer_num}"
        print(f"Evaluating {layer_name}")
        # Evaluate with increasing sparsity levels
        # Clone the original model for each experiment
        sparse_model = OPTForCausalLM.from_pretrained(model_name, num_labels=2).to(device)
        sparse_model.load_state_dict(model.state_dict())
        # Apply sparsity
        apply_sparsity_to_layer(sparse_model.model.decoder.layers[layer_num], s)
        # Evaluate
        accuracy = evaluate_model(sparse_model,tokenizer)
        sensitivity = base_accuracy - accuracy
        sensitivities.append(sensitivity)
#         print(f"Sparsity Level: {s}, Accuracy: {accuracy}, Sensitivity: {sensitivity}")
        # Clean up to free GPU memory
        del sparse_model
        torch.cuda.empty_cache()
        layer_sensitivity[layer_name] = sensitivity
        print("Layer Sensitivity Analysis Completed:")
        for layer, sensitivity in layer_sensitivity.items():
            print(f"{layer}: {sensitivity}")
layer_sensitivitites = {}
layer_idx =0
for sensitivity in list(layer_sensitivity.items()):
    layer_sensitivitites["layer_"+str(layer_idx)] = sensitivity
    layer_idx+=1

