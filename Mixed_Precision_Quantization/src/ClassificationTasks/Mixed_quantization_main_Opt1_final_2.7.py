import torch
import re
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, OPTForCausalLM
from datasets import load_dataset
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from datasets import load_dataset
from accelerate import Accelerator
accelerator = Accelerator()
import os

# Load environment variables from .env file
# load_dotenv()
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)
# from Correlation_based_quantization.cmpq_opt import layer_sensitivitites
# print(layer_sensitivitites)
# print(layer_sensiivitites)
# from Taylor_decomposition_based_quantization.pmpq_opt_125m import layer_sensitivitites
# from pruning_based_quantization.pmpq_opt_125m import layer_sensitivitites


l1 = {
      'layer_0': 24.7633, 'layer_1': 6.11525, 'layer_2': 13.711, 'layer_3': 8.0195, 'layer_4': 9.721,
      'layer_5': 8.83439, 'layer_6': 11.3716, 'layer_7': 13.2819, 'layer_8': 7.9037, 'layer_9': 7.48585,
      'layer_10': 12.9749, 'layer_11': 10.09528, 'layer_12': 6.424831, 'layer_13': 6.07216392,
      'layer_14': 7.21301, 'layer_15': 7.65826, 'layer_16': 6.4356, 'layer_17': 13.9727, 'layer_18': 4.89462,
      'layer_19': 5.42218, 'layer_20': 8.652, 'layer_21': 6.09658, 'layer_22': 6.468, 'layer_23': 6.4442, 'layer_24': 6.4356, 'layer_25': 13.9727, 'layer_26': 4.89462,
      'layer_27': 5.42218, 'layer_28': 8.652, 'layer_29': 6.09658, 'layer_30': 6.468, 'layer_31': 6.4442
  }
l2 = {'layer_0': -0.0600207933461292, 'layer_1': -0.0600207933461292, 'layer_2': -0.0600207933461292, 'layer_3': -0.0600207933461292, 'layer_4': -0.0600207933461292, 'layer_5': -0.05970089571337173, 'layer_6': -0.0600207933461292, 'layer_7': -0.05970089571337173, 'layer_8': -0.05970089571337173, 'layer_9': -0.0600207933461292, 'layer_10': -0.05970089571337173, 'layer_11': -0.05970089571337173, 'layer_12': -0.05970089571337173, 'layer_13': -0.0600207933461292, 'layer_14': -0.05970089571337173, 'layer_15': -0.05970089571337173, 'layer_16': -0.05970089571337173, 'layer_17': -0.05970089571337173, 'layer_18': -0.05970089571337173, 'layer_19': -0.05970089571337173, 'layer_20': -0.0600207933461292, 'layer_21': -0.0600207933461292, 'layer_22': -0.0600207933461292, 'layer_23': -0.0600207933461292, 'layer_24': -0.05970089571337173, 'layer_25': -0.05970089571337173, 'layer_26': -0.05970089571337173, 'layer_27': -0.05970089571337173, 'layer_28': -0.05970089571337173, 'layer_29': -0.0600207933461292, 'layer_30': -0.0600207933461292, 'layer_31': -0.0600207933461292}
l3 = {'layer_0': 1.0995205301344395, 'layer_1': 0.6137791457176208, 'layer_2': 0.8564823066711426, 'layer_3': 0.7497436543524265, 'layer_4': 0.6475968232750893, 'layer_5': 0.6266153248429298, 'layer_6': 0.9870001959234476, 'layer_7': 0.5624912996560335, 'layer_8': 0.6649266678482294, 'layer_9': 0.4315337785124779, 'layer_10': -0.15606744532883168, 'layer_11': 0.2810993002653122, 'layer_12': 0.2104423563480377, 'layer_13': 0.6686014277130365, 'layer_14': 0.005536769729852676, 'layer_15': 0.5386015409171582, 'layer_16': 0.8978167580962181, 'layer_17': 0.5282053351044654, 'layer_18': 0.5278098473191262, 'layer_19': 0.3938675766825676, 'layer_20': 0.8167414792835712, 'layer_21': 1.3145552425444127, 'layer_22': 0.7458991473317146, 'layer_23': 2.0543314699053763, 'layer_24': 0.8978167580962181, 'layer_25': 0.5282053351044654, 'layer_26': 0.5278098473191262, 'layer_27': 0.3938675766825676, 'layer_28': 0.8167414792835712, 'layer_29': 1.3145552425444127, 'layer_30': 0.7458991473317146, 'layer_31': 2.0543314699053763}
layer_sensitivitites_array = [l1, l2 ,l3] #cmpq, pmpq, tdmpq
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
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
    # model.to(accelerator.device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(accelerator.device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(accelerator.device)
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, dim=2)  # max over num_classes

            # Expand labels if necessary
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)  # make it [32, 1]
                labels = labels.expand(-1, predicted.shape[1])  # expand to match predicted shape [32, 128]

            correct += (predicted == labels).sum().item()
            total += labels.numel()

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
def train(model, train_loader, optimizer):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss().to(accelerator.device)  # Ensure the loss function is on the correct device

    for epoch in range(5):  # Number of epochs can be adjusted
        for i, batch in enumerate(train_loader):
            # Move all data in batch to the correct device
            inputs = {key: val.to(accelerator.device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(accelerator.device)

            # Make sure to pass attention_mask if it exists
            if 'attention_mask' not in inputs:
                print("Warning: 'attention_mask' not found in inputs, which is usually required for transformer models.")

            optimizer.zero_grad()  # Clear previous gradients
            # model = model.to(accelerator.device)
            outputs = model(**inputs)

            # Reshape logits and labels to match
            logits = outputs.logits[:, 0, :]  # Selecting logits for the first token in each sequence for classification

            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            # Optionally print out loss every few batches or epochs
            if i % 50 == 0:
                print(f"Epoch {epoch}, Batch {i}: Loss {loss.item()}")

    print(f"Training completed for {epoch+1} epochs.")

def main():
    # Initial setup
    # Replace the paths with the specific LLaMA 2 model you are using
    model_name = "facebook/opt-2.7b"
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
    # model = OPTForCausalLM.from_pretrained(model_name)
    # Load the dataset
    dataset = load_dataset("imdb")
    model, tokenizer = accelerator.prepare(model, tokenizer)
# Example to adjust the dataset for use with LLaMA if necessary (e.g., if using for classification)
# This part would need customization based on what you're trying to achieve with the LLaMA model on IMDB
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
###### for imdb
    dataset = dataset.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128), batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
###### for imdb
    train_dataloader = DataLoader(dataset['train'], batch_size=16, shuffle=True)
    test_dataloader = DataLoader(dataset['test'], batch_size=16, shuffle=False)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train_dataloader = accelerator.prepare(train_dataloader)
    test_dataloader = accelerator.prepare(test_dataloader)

    train(model, train_dataloader, optimizer)
    # dataloader = DataLoader(dataset['test'], batch_size=32)
    accuracy_before = evaluate_model(model, tokenizer, test_dataloader)
    print(f"Accuracy before quantization: {accuracy_before:.2f}")
    for i in range(3):
    # Placeholder for calculating layer sensitivity and clustering
        layer_sensitivity = layer_sensitivitites_array[i]#{'layer_0': 0.1, 'layer_1': 0.5, 'layer_2': 0.9}  # Example sensitivities
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
                layer_precision_map[i] = 4   # int8 for least sensitive

        # Apply quantization and print results
        
        model, total_orig_mem, total_quant_mem, compression_ratio = replace_layers_and_calculate_memory(model, layer_precision_map)
        total_reduction_percent = 100 * (1 - total_quant_mem / total_orig_mem)
        print(f"Total Memory Reduction: Original Memory = {total_orig_mem} bits, Quantized Memory = {total_quant_mem} bits, Reduction = {total_reduction_percent:.2f}%")
        compression_ratio = total_orig_mem/total_quant_mem
        print(f"Compression ratio = {compression_ratio:.2f}")
        # Evaluate after quantization
        accuracy_after = evaluate_model(model, tokenizer, test_dataloader)
        print(f"Accuracy after quantization: {accuracy_after:.5f}")

if __name__ == "__main__":
    main()
