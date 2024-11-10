import torch
import numpy as np
from transformers import AutoTokenizer,OPTForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.cluster import KMeans
from tqdm import tqdm
import re
import torch.optim as optim
import torch
from accelerate import Accelerator
accelerator = Accelerator()
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)
# 
import os

def train(model, train_loader, optimizer):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss().to(accelerator.device)  # Ensure the loss function is on the correct device

    for epoch in range(3):  # Number of epochs can be adjusted
        for i, batch in enumerate(train_loader):
            # Move all data in batch to the correct device
            inputs = {key: val.to(accelerator.device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(accelerator.device)
            if labels.dim() > 1:
                raise ValueError("Labels must be 1D, but got a tensor with shape: {}".format(labels.shape))

            optimizer.zero_grad()
            model = model.to(device)
            outputs = model(**inputs)

            logits = outputs.logits[:, 0, :]  # Assuming this is correct for your model's output shape
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Epoch {epoch}, Batch {i}: Loss {loss.item()}")

    print(f"Training completed for {epoch+1} epochs.")

# Assume main function and other necessary parts are correctly defined elsewhere

from transformers import BertForMaskedLM, DataCollatorForLanguageModeling
# from correlation_based_quantization.cmpq_opt125m import layer_sensitivitites
# from pruning_based_quantization.pmpq_opt125m import layer_sensitivitites
# from Taylor_decomposition_based_quantization.tdmpq_opt125m import layer_sensitivitites
# print(layer_sensitivitites)
# print(layer_sesitivitites)
# Adjust the path for your actual module for sensitivity import
# layer_sensitivitites = {'layer_0': 0.019878246857284565, 'layer_1': 0.017031907655393197, 'layer_2': 0.01419698592795493, 'layer_3': 0.011416636859833296, 'layer_4': 0.00971312281852077, 'layer_5': 0.008231076891791145, 'layer_6': 0.008750632762504207, 'layer_7': 0.008948706220364255, 'layer_8': 0.009496784614152154, 'layer_9': 0.010138825645658378, 'layer_10': 0.010947287551763107, 'layer_11': 0.01233985319543962}
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
# l1 = {'layer_0': 0.02, 'layer_1': 0.013, 'layer_2': 0.012, 'layer_3': 0.013, 'layer_4': 0.01, 'layer_5': 0.011, 'layer_6': 0.01, 'layer_7': 0.01, 'layer_8': 0.009, 'layer_9': 0.009, 'layer_10': 0.01, 'layer_11': 0.011}
l1 = {
      'layer_0': 24.7633, 'layer_1': 6.11525, 'layer_2': 13.711, 'layer_3': 8.0195, 'layer_4': 9.721,
      'layer_5': 8.83439, 'layer_6': 11.3716, 'layer_7': 13.2819, 'layer_8': 7.9037, 'layer_9': 7.48585,
      'layer_10': 12.9749, 'layer_11': 10.09528, 'layer_12': 6.424831, 'layer_13': 6.07216392,
      'layer_14': 7.21301, 'layer_15': 7.65826, 'layer_16': 6.4356, 'layer_17': 13.9727, 'layer_18': 4.89462,
      'layer_19': 5.42218, 'layer_20': 8.652, 'layer_21': 6.09658, 'layer_22': 6.468, 'layer_23': 6.4442
  }
l2 = {'layer_0': -0.0600207933461292, 'layer_1': -0.0600207933461292, 'layer_2': -0.0600207933461292, 'layer_3': -0.0600207933461292, 'layer_4': -0.0600207933461292, 'layer_5': -0.05970089571337173, 'layer_6': -0.0600207933461292, 'layer_7': -0.05970089571337173, 'layer_8': -0.05970089571337173, 'layer_9': -0.0600207933461292, 'layer_10': -0.05970089571337173, 'layer_11': -0.05970089571337173, 'layer_12': -0.05970089571337173, 'layer_13': -0.0600207933461292, 'layer_14': -0.05970089571337173, 'layer_15': -0.05970089571337173, 'layer_16': -0.05970089571337173, 'layer_17': -0.05970089571337173, 'layer_18': -0.05970089571337173, 'layer_19': -0.05970089571337173, 'layer_20': -0.0600207933461292, 'layer_21': -0.0600207933461292, 'layer_22': -0.0600207933461292, 'layer_23': -0.0600207933461292}
l3 = {'layer_0': 1.0995205301344395, 'layer_1': 0.6137791457176208, 'layer_2': 0.8564823066711426, 'layer_3': 0.7497436543524265, 'layer_4': 0.6475968232750893, 'layer_5': 0.6266153248429298, 'layer_6': 0.9870001959234476, 'layer_7': 0.5624912996560335, 'layer_8': 0.6649266678482294, 'layer_9': 0.4315337785124779, 'layer_10': -0.15606744532883168, 'layer_11': 0.2810993002653122, 'layer_12': 0.2104423563480377, 'layer_13': 0.6686014277130365, 'layer_14': 0.005536769729852676, 'layer_15': 0.5386015409171582, 'layer_16': 0.8978167580962181, 'layer_17': 0.5282053351044654, 'layer_18': 0.5278098473191262, 'layer_19': 0.3938675766825676, 'layer_20': 0.8167414792835712, 'layer_21': 1.3145552425444127, 'layer_22': 0.7458991473317146, 'layer_23': 2.0543314699053763}
layer_sensitivitites_array = [l1, l2, l3]

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
    data = data.input_ids.to(accelerator.device)

    seqlen = 512
    model = model.eval()
    n_samples = data.numel() // seqlen

    nlls = []

    with tqdm(range(n_samples), desc="Perplexity -") as progress_bar:
        for i in progress_bar:
            start_index = i * seqlen
            end_index = (i + 1) * seqlen
            batch = data[:, start_index:end_index].to(accelerator.device)
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

# def evaluate_model_perplexity(model, dataloader):
#     model.eval()
#     total_loss = 0
#     total_examples = 0

#     with torch.no_grad():
#         for batch in dataloader:
#             inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
#             outputs = model(**inputs)
#             logits = outputs.logits
#             labels = inputs['labels']
            
#             # Calculate loss
#             loss_fct = torch.nn.CrossEntropyLoss()  # We might ignore index -100 (masked values)
#             mask = labels != -100
#             loss = loss_fct(logits.view(-1, model.config.vocab_size), labels.view(-1))
#             total_loss += loss.item() * batch['input_ids'].size(0)  # Multiply by batch size
#             total_examples += batch['input_ids'].size(0)

#     mean_loss = total_loss / total_examples
#     perplexity = torch.exp(torch.tensor(mean_loss))
#     return perplexity.item()


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
    # Load and prepare the Wikitext dataset
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = OPTForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map='auto',
            output_hidden_states=True
        )
    else:
        model = None
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = dataset.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Create DataLoader
    dataloader = DataLoader(dataset['test'], batch_size=32)
    # NOTE: Since Wikitext does not have labels, we cannot calculate accuracy or perform tasks requiring labels.
    # Additional steps would be needed to adapt the task to something applicable, such as masked language modeling.
    for i in range(3):
        print("###############################################################################")
        # Placeholder for calculating layer sensitivity and clustering
        layer_sensitivity = layer_sensitivitites_array[i]
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
        # accuracy_before = evaluate_model(model, tokenizer, dataloader, device)
        # print(f"Perplexity before quantization: {accuracy_before:.5f}")
        dataset_name = 'wikitext'
        # model_name = 'bert-base-multilingual-cased'
        model_name = "facebook/opt-1.3b"
        tokenized_datasets, tokenizer = load_dataset_and_tokenizer(dataset_name, model_name)
        dataloader = create_data_loader(tokenized_datasets, tokenizer)
        model = OPTForCausalLM.from_pretrained(model_name)
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        train(model, dataloader, optimizer)
        perplexity_before = evaluate_perplexity(model, tokenizer)
        print(f"Perplexity Before quantization: {perplexity_before}")
        model, total_orig_mem, total_quant_mem, compression_ratio = replace_layers_and_calculate_memory(model, layer_precision_map)
        total_reduction_percent = 100 * (1 - total_quant_mem / total_orig_mem)
        print(f"Total Memory Reduction: Original Memory = {total_orig_mem} bits, Quantized Memory = {total_quant_mem} bits, Reduction = {total_reduction_percent:.2f}%")
        # print(f"Compression ratio = {compression_ratio:.2f}")
        compression_ratio = total_orig_mem/total_quant_mem
        print(f"Compression ratio = {compression_ratio:.2f}")
        perplexity_after = evaluate_perplexity(model, tokenizer)
        print(f"Perplexity after quantization: {perplexity_after}")
        print(f"Perplexity Drop : { perplexity_before - perplexity_after}")

if __name__ == "__main__":
    main()

