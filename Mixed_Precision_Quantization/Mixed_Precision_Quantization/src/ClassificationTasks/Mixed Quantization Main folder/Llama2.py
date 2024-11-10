import torch
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.cluster import KMeans
from tqdm import tqdm

from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from Correlation_based_quantization.canonical_analysis_sensitivites_llama2 import layer_sensitivitites
#from Taylor_decomposition_based_quantization.TaylorSensitivies_bert_large_uncased import layer_sensitivitites
# from pruning_based_quantization.DifferentpercentsparsitypruniningSensitivies_bert_base_uncased import layer_sensitivitites
#pruning Encoder Layer 11: [0.015075175943698005, 0.0017994241842610448, -0.0010396673064619333, -0.0005998080614203483, 0.009077095329494522, -0.0023592450415866884, -0.004158669225847733, -0.03882757517594371, -0.004358605246321201, -0.004278630838131792, 0.009476967370441458, 0.0027591170825335687]
# print("layer_sensitivitites")
# print(layeensitivitites)

# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

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
def evaluate_model(model, tokenizer, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v for k, v in batch.items() if k != 'labels'}
            # labels = batch['labels'].to(device)
            labels = batch['labels']
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy
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
            # print(f"Layer {name}: Original Memory = {orig_mem} bits, Quantized Memory = {quant_mem} bits, Reduction = {reduction:.2f}%")
        if isinstance(module, nn.LayerNorm):
            bits = layer_precision_map.get(layer_number, 8)  # Default to 32 bits if not specified in the map
            quant_layer = LinearLSQ(module, bits)
            setattr(model, name, quant_layer)  # Replace the original layer with the quantized version
            model._modules[name] = quant_layer  # Ensure the module is updated in the model's dictionary of modules
            # After replacing the layer, calculate the memory change
            orig_mem, quant_mem, reduction, compression_ratio = quant_layer.calculate_memory_reduction()
            total_original_mem += orig_mem
            total_quantized_mem += quant_mem
            # print(f"Layer {name}: Original Memory = {orig_mem} bits, Quantized Memory = {quant_mem} bits, Reduction = {reduction:.2f}%")
        if isinstance(module, nn.MultiheadAttention):
            # Quantize the linear components within the attention module
            # print("camamamaam")
            for attn_component_name in ['in_proj_weight', 'in_proj_bias', 'out_proj.weight', 'out_proj.bias']:
                param = getattr(module, attn_component_name)
                if param is not None:
                    bits = layer_precision_map.get(layer_number, 8)
                    quant_layer = LinearLSQ(param, bits)
                    setattr(module, attn_component_name, quant_layer.param)
                    # Calculate memory change
                    orig_mem = param.nelement() * 32  # Assuming original is FP32
                    quant_mem = quant_layer.param.nelement() * bits
                    total_original_mem += orig_mem
                    total_quantized_mem += quant_mem
                    reduction = (orig_mem - quant_mem) / orig_mem * 100
                    # print(f"Attention {name} {attn_component_name}: Original Memory = {orig_mem} bits, Quantized Memory = {quant_mem} bits, Reduction = {reduction:.2f}%")

    return model, total_original_mem, total_quantized_mem, compression_ratio


def main():

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

    # Load the dataset
    dataset = load_dataset("imdb")

# Example to adjust the dataset for use with LLaMA if necessary (e.g., if using for classification)
# This part would need customization based on what you're trying to achieve with the LLaMA model on IMDB
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
###### for imdb
    dataset = dataset.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128), batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
###### for imdb

######### wikitext
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # # Tokenize the text
    # dataset = dataset.map(lambda examples: tokenizer(examples['text'], 
    #                                                 padding="max_length", 
    #                                                 truncation=True, 
    #                                                 max_length=128), batched=True)

    # # Assuming you are preparing this dataset for a task that needs 'labels'
    # # Normally, you'd prepare this part based on your specific task requirements

    # # Set the format for PyTorch
    # dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])



######### wikitext
    dataloader = DataLoader(dataset['test'], batch_size=32)
    accuracy_before = evaluate_model(model, tokenizer, dataloader)
    print(f"Accuracy before quantization: {accuracy_before:.2f}")

    # Placeholder for calculating layer sensitivity and clustering
    layer_sensitivity = layer_sensitivitites#{'layer_0': 0.1, 'layer_1': 0.5, 'layer_2': 0.9}  # Example sensitivities
    # print("Sree")
    # print(layer_sensitivity)
    sensitivities = np.array(list(layer_sensitivity.values())).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(sensitivities)
    clusters = kmeans.labels_

    layer_precision_map = {}
    for i, (layer_name, _) in enumerate(layer_sensitivity.items()):
        if clusters[i] == 0:
            layer_precision_map[i] = 16  # fp32 for most sensitive
        elif clusters[i] == 1:
            layer_precision_map[i] = 8  # int16 for medium
        else:
            layer_precision_map[i] = 8   # int8 for least sensitive

    # Apply quantization and print results
    
    model, total_orig_mem, total_quant_mem, compression_ratio = replace_layers_and_calculate_memory(model, layer_precision_map)
    total_reduction_percent = 100 * (1 - total_quant_mem / total_orig_mem)
    print(f"Total Memory Reduction: Original Memory = {total_orig_mem} bits, Quantized Memory = {total_quant_mem} bits, Reduction = {total_reduction_percent:.2f}%")
    print(f"Compression ratio = {compression_ratio:.2f}")
    # Evaluate after quantization
    accuracy_after = evaluate_model(model, tokenizer, dataloader)
    print(f"Accuracy after quantization: {accuracy_after:.5f}")

if __name__ == "__main__":
    main()
