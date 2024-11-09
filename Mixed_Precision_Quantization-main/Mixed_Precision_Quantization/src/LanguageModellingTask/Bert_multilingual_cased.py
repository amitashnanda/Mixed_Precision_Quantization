import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.cluster import KMeans
from tqdm import tqdm
import re
from transformers import BertForMaskedLM, DataCollatorForLanguageModeling
# from correlation_based_quantization.canonical_analysis_sensitivites_wikitextData2 import layer_sensitivitites
# from pruning_based_quantization.DifferentpercentsparsitypruniningSensitivies_bert_base_wikitextData2 import layer_sensitivitites
from Taylor_decomposition_based_quantization.TaylorSensitivies_bert_base_uncased_wikitextData2 import layer_sensitivitites
print(layer_sensitivitites)
# Adjust the path for your actual module for sensitivity import
# layer_sensitivitites = {'layer_0': 0.019878246857284565, 'layer_1': 0.017031907655393197, 'layer_2': 0.01419698592795493, 'layer_3': 0.011416636859833296, 'layer_4': 0.00971312281852077, 'layer_5': 0.008231076891791145, 'layer_6': 0.008750632762504207, 'layer_7': 0.008948706220364255, 'layer_8': 0.009496784614152154, 'layer_9': 0.010138825645658378, 'layer_10': 0.010947287551763107, 'layer_11': 0.01233985319543962}
class LinearLSQ(nn.Module):
    def __init__(self, original_linear, nbits_w):
        super(LinearLSQ, self).__init__()
        self.original_linear = original_linear
        self.nbits_w = nbits_w
        self.original_weight = original_linear.weight.detach().clone()

    def calculate_memory_reduction(self):
        original_mem = self.original_weight.nelement() * 32  # Assuming original weights are 32-bit floats
        quantized_mem = self.original_weight.nelement() * self.nbits_w
        reduction_percent = 100 * (1 - quantized_mem / original_mem)
        compression_ratio = original_mem / quantized_mem
        return original_mem, quantized_mem, reduction_percent, compression_ratio

    def quantize(self, x, nbits):
        qmin = -(2 ** (nbits - 1))
        qmax = (2 ** (nbits - 1)) - 1
        min_val, max_val = x.min(), x.max()
        scale = (max_val - min_val) / (qmax - qmin)
        scale = max(scale, 1e-8)
        zero_point = qmin - min_val / scale
        q_x = torch.round(x / scale + zero_point)
        q_x.clamp_(qmin, qmax)
        q_x = (q_x - zero_point) * scale
        return q_x

    def forward(self, x):
        quantized_weight = self.quantize(self.original_linear.weight, self.nbits_w)
        self.original_linear.weight = nn.Parameter(quantized_weight)
        output = self.original_linear(x)
        self.original_linear.weight = nn.Parameter(self.original_weight)
        return output

def evaluate_perplexity(model, tokenizer): #Function for evaluating the model
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    # load and prepare dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    data = data.input_ids.to(model.device)

    seqlen = 512
    model = model.eval()
    n_samples = data.numel() // seqlen

    nlls = []

    with tqdm(range(n_samples), desc="Perplexity -") as progress_bar:
        for i in progress_bar:
            start_index = i * seqlen
            end_index = (i + 1) * seqlen
            batch = data[:, start_index:end_index].to(model.device)
            with torch.no_grad():
                logits = model(batch).logits
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
    tokenizer = BertTokenizer.from_pretrained(model_name)
    dataset = load_dataset(dataset_name, 'wikitext-2-raw-v1')

    # Tokenize the data
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    return tokenized_datasets, tokenizer

def create_data_loader(tokenized_datasets, tokenizer):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    dataloader = DataLoader(tokenized_datasets['test'], batch_size=4, collate_fn=data_collator)
    return dataloader

def evaluate_model_perplexity(model, dataloader, device):
    model.eval()
    model.to(device)
    total_loss = 0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs['labels']
            
            # Calculate loss
            loss_fct = torch.nn.CrossEntropyLoss()  # We might ignore index -100 (masked values)
            mask = labels != -100
            loss = loss_fct(logits.view(-1, model.config.vocab_size), labels.view(-1))
            total_loss += loss.item() * batch['input_ids'].size(0)  # Multiply by batch size
            total_examples += batch['input_ids'].size(0)

    mean_loss = total_loss / total_examples
    perplexity = torch.exp(torch.tensor(mean_loss))
    return perplexity.item()


def replace_layers_and_calculate_memory(model, layer_precision_map):
    total_original_mem = 0
    total_quantized_mem = 0
    # First, collect all layers in a list to avoid mutating the OrderedDict during iteration
    layers = []
    step = 0
    for name, module in model.named_modules():
        layers.append([step,name, module])
        step+=1
    for step,name, module in layers:
        layer_number= re.findall(r'\d+', name)
        if len(layer_number)>0:
            layer_number = int(layer_number[0])
        else:
            layer_number = 0
        if isinstance(module, nn.Linear):
            bits = layer_precision_map.get(layer_number, 8)  # Default to 32 bits if not specified in the map
            quant_layer = LinearLSQ(module, bits)
            setattr(model, name, quant_layer)  # Replace the original layer with the quantized version
            model._modules[name] = quant_layer # Ensure the module is updated in the model's dictionary of modules
            # After replacing the layer, calculate the memory change
            orig_mem, quant_mem, reduction, compression_ratio = quant_layer.calculate_memory_reduction()
            total_original_mem += orig_mem
            total_quantized_mem += quant_mem
        if isinstance(module, nn.LayerNorm):
            bits = layer_precision_map.get(layer_number, 8)  # Default to 32 bits if not specified in the map
            quant_layer = LinearLSQ(module, bits)
            setattr(model, name, quant_layer)  # Replace the original layer with the quantized version
            model._modules[name] = quant_layer  # Ensure the module is updated in the model's dictionary of modules
            # After replacing the layer, calculate the memory change
            orig_mem, quant_mem, reduction, compression_ratio = quant_layer.calculate_memory_reduction()
            total_original_mem += orig_mem
            total_quantized_mem += quant_mem
        if isinstance(module, nn.MultiheadAttention):
            # Quantize the linear components within the attention module
            for attn_component_name in ['in_proj_weight', 'in_proj_bias', 'out_proj.weight', 'out_proj.bias']:
                param = getattr(module, attn_component_name)
                if param is not None:
                    bits = layer_precision_map.get(layer_number, 8)
                    quant_layer = LinearLSQ(param, bits)
                    setattr(module, attn_component_name, quant_layer.param)
                    model._modules[name] = quant_layer  # Ensure the module is updated in the model's dictionary of modules
                    orig_mem = param.nelement() * 32  # Assuming original is FP32
                    quant_mem = quant_layer.param.nelement() * bits
                    total_original_mem += orig_mem
                    total_quantized_mem += quant_mem
                    reduction = (orig_mem - quant_mem) / orig_mem * 100
                    # print(f"Attention {name} {attn_component_name}: Original Memory = {orig_mem} bits, Quantized Memory = {quant_mem} bits, Reduction = {reduction:.2f}%")

    return model, total_original_mem, total_quantized_mem, compression_ratio
def main():
    # Initial setup
    model_name = 'bert-large-uncased'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # Load and prepare the Wikitext dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    dataset = dataset.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Create DataLoader
    dataloader = DataLoader(dataset['test'], batch_size=32)

    # NOTE: Since Wikitext does not have labels, we cannot calculate accuracy or perform tasks requiring labels.
    # Additional steps would be needed to adapt the task to something applicable, such as masked language modeling.

    # Placeholder for calculating layer sensitivity and clustering
    layer_sensitivity = layer_sensitivitites
    sensitivities = np.array(list(layer_sensitivity.values())).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(sensitivities)
    clusters = kmeans.labels_

    layer_precision_map = {}
    for i, (layer_name, _) in enumerate(layer_sensitivity.items()):
        if clusters[i] == 0:
            layer_precision_map[i] = 16  # Higher precision for more sensitive layers
        elif clusters[i] == 1:
            layer_precision_map[i] = 8  # Moderate precision
        else:
            layer_precision_map[i] = 4   # Lower precision for less sensitive layers

    # Assume other necessary methods like replace_layers_and_calculate_memory are defined similarly
    # Total memory changes would be calculated here but note the model used doesn

    # Apply quantization and print results
    dataset_name = 'wikitext'
    model_name = 'bert-large-uncased'
    tokenized_datasets, tokenizer = load_dataset_and_tokenizer(dataset_name, model_name)
    dataloader = create_data_loader(tokenized_datasets, tokenizer)
    model = BertForMaskedLM.from_pretrained(model_name)
    perplexity_before = evaluate_perplexity(model, tokenizer)
    print(f"Perplexity Before quantization: {perplexity_before}")

    model, total_orig_mem, total_quant_mem, compression_ratio = replace_layers_and_calculate_memory(model, layer_precision_map)
    total_reduction_percent = 100 * (1 - total_quant_mem / total_orig_mem)
    print(f"Total Memory Reduction: Original Memory = {total_orig_mem} bits, Quantized Memory = {total_quant_mem} bits, Reduction = {total_reduction_percent:.2f}%")
    print(f"Compression Ratio: {total_orig_mem/total_quant_mem}")
    # Evaluate after quantization
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    perplexity_after = evaluate_perplexity(model, tokenizer)
    print(f"Perplexity after quantization: {perplexity_after}")
    print(f"Perplexity Drop : { perplexity_before - perplexity_after}")

if __name__ == "__main__":
    main()

