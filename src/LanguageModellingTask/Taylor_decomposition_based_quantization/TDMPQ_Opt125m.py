import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, OPTForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
dataset_name = 'wikitext'
model_name = "facebook/opt-125m"

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

tokenized_datasets, tokenizer = load_dataset_and_tokenizer(dataset_name, model_name)
dataloader = create_data_loader(tokenized_datasets, tokenizer)
model = OPTForCausalLM.from_pretrained(model_name)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
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
def taylor_sensitivity(model, dataloader, device):
    sensitivities = []
    model.eval()
    print(model)
    for index, layer in enumerate(model.model.decoder.layers):
        print("going in")
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

print("Sd1")

sensitivity_scores = taylor_sensitivity(model, dataloader, device)
layer_sensitivitites = {}
for layer_idx, sensitivity in sensitivity_scores:
    print(f"Layer {layer_idx}: Sensitivity = {sensitivity}")
    layer_sensitivitites["layer_"+str(layer_idx)]=sensitivity
